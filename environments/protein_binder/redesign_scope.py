from __future__ import annotations

import json
import random
from typing import Any

from synthetic_dataset import (
    AMINO_ACIDS,
    CLASS_RESIDUES,
    analyze_sequence_profile,
    build_target_spec,
    classify_counts,
    net_charge,
    normalize_sequence,
    public_target_spec,
    score_sequence,
)

STRATEGIES = ["balanced", "anchor", "composition", "charge", "explore"]
DEFAULT_BUDGET = 12.0
DEFAULT_COSTS = {
    "design": 1.5,
    "quick": 0.75,
    "full": 2.5,
}


def candidate_id(index: int) -> str:
    return f"C{index:04d}"


def parse_candidate_id(text: str) -> int | None:
    normalized = (text or "").strip().upper()
    if normalized.startswith("C") and normalized[1:].isdigit():
        return int(normalized[1:])
    if normalized.isdigit():
        return int(normalized)
    return None


def _pick_residue_for_class(
    class_name: str,
    forbidden_residues: list[str],
    rng: random.Random,
    exclude: str | None = None,
) -> str:
    allowed = [
        residue
        for residue in CLASS_RESIDUES.get(class_name, "")
        if residue not in forbidden_residues and residue != exclude
    ]
    if not allowed:
        allowed = [residue for residue in AMINO_ACIDS if residue not in forbidden_residues and residue != exclude]
    return rng.choice(allowed)


def _mutate_reference_to_seed(rng: random.Random, reference_sequence: str, spec: dict[str, Any]) -> str:
    public_spec = public_target_spec(spec)
    residues = list(reference_sequence)
    mutable_positions = list(range(spec["length"]))

    for _ in range(64):
        working = residues[:]
        num_mutations = min(len(working), rng.randint(2, 4))
        for idx in rng.sample(mutable_positions, k=num_mutations):
            slot = spec["position_preferences"][idx]
            if rng.random() < 0.5:
                replacement_pool = [
                    residue
                    for residue in AMINO_ACIDS
                    if residue not in CLASS_RESIDUES[slot["primary"]] and residue not in spec["forbidden_residues"]
                ]
            else:
                replacement_pool = [
                    residue
                    for residue in AMINO_ACIDS
                    if residue not in spec["forbidden_residues"]
                ]
            if replacement_pool:
                working[idx] = rng.choice(replacement_pool)
        candidate = "".join(working)
        candidate_score = score_sequence(candidate, public_spec)["reward"]
        if 0.25 <= candidate_score <= 0.8:
            return candidate
    return reference_sequence


def _format_position_preferences(spec: dict[str, Any]) -> str:
    return "\n".join(
        f"  {slot['position']}: prefer {slot['primary']} (backup {slot['secondary']})"
        for slot in spec["position_preferences"]
    )


def make_scope05_prompt(spec: dict[str, Any]) -> str:
    seed_profile = analyze_sequence_profile(spec["seed_sequence"])
    return (
        "Optimize a peptide binder by iteratively redesigning a weak seed candidate under a fixed budget.\n"
        "You must create and screen candidates with tools, then submit the best screened candidate together with its sequence.\n\n"
        f"Target ID: {spec['target_id']}\n"
        f"Binder length: {spec['length']}\n"
        f"Desired net charge: {spec['target_charge']:+d}\n"
        f"Anchor positions: {', '.join(str(position) for position in spec['anchor_positions'])}\n"
        f"Avoid residues: {', '.join(spec['forbidden_residues']) if spec['forbidden_residues'] else 'none'}\n"
        f"Composition targets: {', '.join(f'{name}={count}' for name, count in spec['desired_counts'].items())}\n"
        f"Budget: {spec['budget']} total units\n"
        f"Tool costs: design_variants={spec['costs']['design']}, quick_screen={spec['costs']['quick']} per candidate, full_screen={spec['costs']['full']} per candidate\n\n"
        "Strategy guide:\n"
        "- balanced: small mixed improvements across anchors, composition, and charge\n"
        "- anchor: focus on hotspot / anchor positions\n"
        "- composition: rebalance residue-class counts\n"
        "- charge: fix net charge mismatch\n"
        "- explore: take diverse local search steps\n\n"
        "Position preferences:\n"
        f"{_format_position_preferences(spec)}\n\n"
        "Initial candidate table:\n"
        f"- C0000 sequence: {spec['seed_sequence']}\n"
        f"- C0000 profile: length={seed_profile['length']}, net_charge={seed_profile['net_charge']:+d}, "
        f"special_residue_count={seed_profile['special_residue_count']}\n\n"
        "Goal: submit the single best screened candidate using both tags, for example: "
        "<answer>C0003</answer><sequence>ACDEFGHIK</sequence>. "
        "The sequence must exactly match the submitted candidate ID. Reserve one turn for the final answer and do not spend your last turn on another tool call."
    )


def build_scope05_rows(num_examples: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for example_idx in range(num_examples):
        spec = build_target_spec(rng, split=split, example_idx=example_idx)
        seed_sequence = _mutate_reference_to_seed(rng, spec["reference_sequence"], spec)
        public_spec = public_target_spec(spec)
        seed_score = score_sequence(seed_sequence, public_spec)["reward"]
        task_info = {
            **spec,
            "seed_sequence": seed_sequence,
            "seed_score": round(seed_score, 3),
            "budget": DEFAULT_BUDGET,
            "costs": DEFAULT_COSTS,
            "task": "protein-binder-scope-0.5",
        }
        rows.append(
            {
                "question": make_scope05_prompt(task_info),
                "answer": spec["reference_sequence"],
                "info": task_info,
                "task": "protein-binder-scope-0.5",
            }
        )
    return rows


def summarize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    profile = analyze_sequence_profile(candidate["sequence"])
    return {
        "candidate_id": candidate["candidate_id"],
        "parent_id": candidate["parent_id"],
        "sequence": candidate["sequence"],
        "mutations": candidate["mutations"],
        "net_charge": profile["net_charge"],
        "quick_score": candidate.get("quick_score"),
        "full_score": candidate.get("full_score"),
        "screened": candidate.get("screened", False),
    }


def format_candidate_table(candidates: dict[str, dict[str, Any]]) -> str:
    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.get("full_score")
            if candidate.get("full_score") is not None
            else candidate.get("quick_score")
            if candidate.get("quick_score") is not None
            else -1.0,
            candidate["candidate_id"],
        ),
        reverse=True,
    )
    return json.dumps([summarize_candidate(candidate) for candidate in ordered], indent=2)


def _candidate_slots_by_priority(sequence: str, spec: dict[str, Any]) -> list[int]:
    sequence = normalize_sequence(sequence)
    scored_positions: list[tuple[float, int]] = []
    for idx, residue in enumerate(sequence):
        slot = spec["position_preferences"][idx]
        position_score = 1.0 if residue in CLASS_RESIDUES[slot["primary"]] else 0.5 if residue in CLASS_RESIDUES[slot["secondary"]] else 0.0
        anchor_bonus = 0.4 if slot["position"] in spec["anchor_positions"] else 0.0
        scored_positions.append((position_score + anchor_bonus, idx))
    scored_positions.sort(key=lambda item: item[0])
    return [idx for _, idx in scored_positions]


def _apply_anchor_strategy(residues: list[str], spec: dict[str, Any], rng: random.Random) -> None:
    for position in spec["anchor_positions"]:
        idx = position - 1
        slot = spec["position_preferences"][idx]
        residues[idx] = _pick_residue_for_class(slot["primary"], spec["forbidden_residues"], rng, exclude=residues[idx])


def _apply_charge_strategy(residues: list[str], spec: dict[str, Any], rng: random.Random) -> None:
    current_charge = net_charge("".join(residues))
    charge_gap = spec["target_charge"] - current_charge
    if charge_gap == 0:
        idx = rng.choice(_candidate_slots_by_priority("".join(residues), spec))
        slot = spec["position_preferences"][idx]
        residues[idx] = _pick_residue_for_class(slot["primary"], spec["forbidden_residues"], rng, exclude=residues[idx])
        return

    candidate_positions = _candidate_slots_by_priority("".join(residues), spec)
    for idx in candidate_positions:
        if charge_gap > 0:
            residues[idx] = _pick_residue_for_class("basic", spec["forbidden_residues"], rng, exclude=residues[idx])
            return
        residues[idx] = _pick_residue_for_class("acidic", spec["forbidden_residues"], rng, exclude=residues[idx])
        return


def _apply_composition_strategy(residues: list[str], spec: dict[str, Any], rng: random.Random) -> None:
    sequence = "".join(residues)
    counts = classify_counts(sequence)
    deficits = [
        class_name
        for class_name, target in spec["desired_counts"].items()
        if counts[class_name] < target
    ]
    if not deficits:
        deficits = [slot["primary"] for slot in spec["position_preferences"]]
    target_class = rng.choice(deficits)
    idx = rng.choice(_candidate_slots_by_priority(sequence, spec))
    residues[idx] = _pick_residue_for_class(target_class, spec["forbidden_residues"], rng, exclude=residues[idx])


def _apply_explore_strategy(residues: list[str], spec: dict[str, Any], rng: random.Random) -> None:
    idx = rng.choice(range(len(residues)))
    slot = spec["position_preferences"][idx]
    preferred_class = slot["primary"] if rng.random() < 0.65 else slot["secondary"]
    residues[idx] = _pick_residue_for_class(preferred_class, spec["forbidden_residues"], rng, exclude=residues[idx])


def _apply_balanced_strategy(residues: list[str], spec: dict[str, Any], rng: random.Random) -> None:
    _apply_anchor_strategy(residues, spec, rng)
    _apply_composition_strategy(residues, spec, rng)
    _apply_charge_strategy(residues, spec, rng)


def _mutation_summary(parent: str, child: str) -> list[str]:
    return [f"{idx + 1}{before}>{after}" for idx, (before, after) in enumerate(zip(parent, child, strict=True)) if before != after]


def design_variants_from_strategy(
    parent_sequence: str,
    strategy: str,
    num_variants: int,
    spec: dict[str, Any],
    rng: random.Random,
    existing_sequences: set[str],
) -> list[dict[str, Any]]:
    strategy = (strategy or "balanced").strip().lower()
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Expected one of: {', '.join(STRATEGIES)}")

    builder_map = {
        "balanced": _apply_balanced_strategy,
        "anchor": _apply_anchor_strategy,
        "composition": _apply_composition_strategy,
        "charge": _apply_charge_strategy,
        "explore": _apply_explore_strategy,
    }

    created: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = max(16, num_variants * 10)
    public_spec = public_target_spec(spec)

    while len(created) < num_variants and attempts < max_attempts:
        attempts += 1
        working = list(parent_sequence)
        builder_map[strategy](working, spec, rng)
        if strategy in {"balanced", "explore"} and rng.random() < 0.6:
            builder_map[rng.choice(["composition", "charge", "anchor"])](working, spec, rng)
        candidate_sequence = normalize_sequence("".join(working))
        if candidate_sequence in existing_sequences or candidate_sequence == parent_sequence:
            continue
        existing_sequences.add(candidate_sequence)
        created.append(
            {
                "sequence": candidate_sequence,
                "mutations": _mutation_summary(parent_sequence, candidate_sequence),
                "true_score": round(score_sequence(candidate_sequence, public_spec)["reward"], 4),
            }
        )
    return created


def make_quick_screen_report(sequence: str, spec: dict[str, Any], rng: random.Random) -> dict[str, float]:
    true_report = score_sequence(sequence, public_target_spec(spec))
    coarse_signal = (
        0.55 * true_report["position"]
        + 0.25 * true_report["anchor"]
        + 0.2 * true_report["charge"]
    )
    noisy_score = max(0.0, min(1.0, coarse_signal + rng.uniform(-0.08, 0.08)))
    return {
        "score": round(noisy_score, 3),
        "anchor_fit": round(true_report["anchor"], 3),
        "charge_fit": round(true_report["charge"], 3),
        "position_fit": round(true_report["position"], 3),
    }


def make_full_screen_report(sequence: str, spec: dict[str, Any], rng: random.Random) -> dict[str, float]:
    true_report = score_sequence(sequence, public_target_spec(spec))
    noisy_score = max(0.0, min(1.0, true_report["reward"] + rng.uniform(-0.03, 0.03)))
    return {
        "score": round(noisy_score, 3),
        "position": round(true_report["position"], 3),
        "composition": round(true_report["composition"], 3),
        "charge": round(true_report["charge"], 3),
        "anchor": round(true_report["anchor"], 3),
        "stability": round(true_report["stability"], 3),
        "valid": round(true_report["valid"], 3),
    }
