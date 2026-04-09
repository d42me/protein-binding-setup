from __future__ import annotations

import json
import random
from typing import Any

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
COMPOSITION_CLASSES = ["aliphatic", "aromatic", "polar", "basic", "acidic"]

CLASS_RESIDUES = {
    "aliphatic": "AVLIM",
    "aromatic": "FWYH",
    "polar": "STNQG",
    "basic": "KR",
    "acidic": "DE",
    "special": "CP",
    "small": "AGST",
}

PAIR_LIBRARY = [
    ("aliphatic", "small"),
    ("polar", "small"),
    ("basic", "polar"),
    ("aromatic", "aliphatic"),
    ("acidic", "polar"),
    ("small", "polar"),
    ("aliphatic", "polar"),
    ("aromatic", "polar"),
]

DEFAULT_WEIGHTS = {
    "position": 0.45,
    "composition": 0.2,
    "charge": 0.1,
    "anchor": 0.15,
    "stability": 0.1,
}

PROPERTIES = {
    residue: {
        "classes": [name for name, members in CLASS_RESIDUES.items() if residue in members],
        "charge": 1 if residue in CLASS_RESIDUES["basic"] else -1 if residue in CLASS_RESIDUES["acidic"] else 0,
        "hydropathy_hint": (
            "hydrophobic"
            if residue in (CLASS_RESIDUES["aliphatic"] + "FWY")
            else "charged"
            if residue in (CLASS_RESIDUES["basic"] + CLASS_RESIDUES["acidic"])
            else "polar"
        ),
    }
    for residue in AMINO_ACIDS
}


def net_charge(sequence: str) -> int:
    return sum(PROPERTIES.get(residue, {}).get("charge", 0) for residue in sequence)


def classify_counts(sequence: str) -> dict[str, int]:
    counts = {name: 0 for name in COMPOSITION_CLASSES}
    for residue in sequence:
        for name in COMPOSITION_CLASSES:
            if residue in CLASS_RESIDUES[name]:
                counts[name] += 1
    return counts


def longest_run(sequence: str) -> int:
    if not sequence:
        return 0
    best = 1
    current = 1
    for index in range(1, len(sequence)):
        if sequence[index] == sequence[index - 1]:
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def slot_score(residue: str, slot: dict[str, Any]) -> float:
    primary_members = CLASS_RESIDUES[slot["primary"]]
    secondary_members = CLASS_RESIDUES[slot["secondary"]]
    if residue in primary_members:
        return 1.0
    if residue in secondary_members:
        return 0.65
    if slot["primary"] in {"basic", "acidic"} and residue in (CLASS_RESIDUES["basic"] + CLASS_RESIDUES["acidic"]):
        return 0.2
    return 0.0


def normalize_sequence(text: str | None) -> str:
    if not text:
        return ""
    cleaned = "".join(ch for ch in text.upper() if ch.isalpha())
    return cleaned


def analyze_sequence_profile(sequence: str) -> dict[str, Any]:
    normalized = normalize_sequence(sequence)
    invalid_residues = sorted({residue for residue in normalized if residue not in AMINO_ACIDS})
    counts = classify_counts(normalized)
    return {
        "sequence": normalized,
        "length": len(normalized),
        "net_charge": net_charge(normalized),
        "composition": counts,
        "longest_repeat_run": longest_run(normalized),
        "special_residue_count": sum(1 for residue in normalized if residue in CLASS_RESIDUES["special"]),
        "invalid_residues": invalid_residues,
    }


def score_sequence(sequence: str, spec: dict[str, Any]) -> dict[str, float]:
    normalized = normalize_sequence(sequence)
    if len(normalized) != spec["length"]:
        return {
            "reward": 0.0,
            "valid": 0.0,
            "position": 0.0,
            "composition": 0.0,
            "charge": 0.0,
            "anchor": 0.0,
            "stability": 0.0,
            "forbidden": 0.0,
        }
    if any(residue not in AMINO_ACIDS for residue in normalized):
        return {
            "reward": 0.0,
            "valid": 0.0,
            "position": 0.0,
            "composition": 0.0,
            "charge": 0.0,
            "anchor": 0.0,
            "stability": 0.0,
            "forbidden": 0.0,
        }

    forbidden_hits = sum(1 for residue in normalized if residue in spec["forbidden_residues"])
    position_score_value = sum(
        slot_score(residue, slot)
        for residue, slot in zip(normalized, spec["position_preferences"], strict=True)
    ) / spec["length"]

    observed_counts = classify_counts(normalized)
    composition_terms = []
    for class_name, target in spec["desired_counts"].items():
        tolerance = max(1, target)
        gap = abs(observed_counts[class_name] - target)
        composition_terms.append(max(0.0, 1.0 - (gap / tolerance)))
    composition_score_value = sum(composition_terms) / len(composition_terms)

    charge_gap = abs(net_charge(normalized) - spec["target_charge"])
    charge_score_value = max(0.0, 1.0 - (charge_gap / max(2, spec["length"] // 2)))

    anchor_score_value = sum(
        slot_score(normalized[position - 1], spec["position_preferences"][position - 1])
        for position in spec["anchor_positions"]
    ) / len(spec["anchor_positions"])

    run_score = max(0.0, 1.0 - 0.35 * max(0, longest_run(normalized) - 2))
    special_budget = spec["max_special_residues"]
    special_count = sum(1 for residue in normalized if residue in CLASS_RESIDUES["special"])
    special_score = max(0.0, 1.0 - 0.5 * max(0, special_count - special_budget))
    forbidden_score = max(0.0, 1.0 - 0.7 * forbidden_hits)
    stability_score_value = (run_score + special_score + forbidden_score) / 3

    weights = spec.get("weights", DEFAULT_WEIGHTS)
    reward = (
        weights["position"] * position_score_value
        + weights["composition"] * composition_score_value
        + weights["charge"] * charge_score_value
        + weights["anchor"] * anchor_score_value
        + weights["stability"] * stability_score_value
    )
    if forbidden_hits:
        reward *= max(0.0, 1.0 - 0.2 * forbidden_hits)

    return {
        "reward": max(0.0, min(1.0, reward)),
        "valid": 1.0,
        "position": position_score_value,
        "composition": composition_score_value,
        "charge": charge_score_value,
        "anchor": anchor_score_value,
        "stability": stability_score_value,
        "forbidden": forbidden_score,
    }


def _sample_sequence(rng: random.Random, spec: dict[str, Any]) -> str:
    residues: list[str] = []
    for slot in spec["position_preferences"]:
        primary_pool = [r for r in CLASS_RESIDUES[slot["primary"]] if r not in spec["forbidden_residues"]]
        secondary_pool = [r for r in CLASS_RESIDUES[slot["secondary"]] if r not in spec["forbidden_residues"]]
        if not primary_pool:
            primary_pool = [r for r in AMINO_ACIDS if r not in spec["forbidden_residues"]]
        if not secondary_pool:
            secondary_pool = primary_pool
        use_primary = rng.random() < 0.7
        pool = primary_pool if use_primary else secondary_pool
        residues.append(rng.choice(pool))
    return "".join(residues)


def _mutate_sequence(rng: random.Random, sequence: str, spec: dict[str, Any]) -> str:
    residues = list(sequence)
    num_mutations = rng.randint(1, max(1, len(sequence) // 3))
    positions = rng.sample(range(len(sequence)), k=num_mutations)
    for idx in positions:
        slot = spec["position_preferences"][idx]
        candidate_pool = list(
            {
                *[r for r in CLASS_RESIDUES[slot["primary"]] if r not in spec["forbidden_residues"]],
                *[r for r in CLASS_RESIDUES[slot["secondary"]] if r not in spec["forbidden_residues"]],
            }
        )
        if not candidate_pool:
            candidate_pool = [r for r in AMINO_ACIDS if r not in spec["forbidden_residues"]]
        residues[idx] = rng.choice(candidate_pool)
    return "".join(residues)


def build_target_spec(rng: random.Random, split: str, example_idx: int) -> dict[str, Any]:
    length = rng.choice([8, 9, 10, 11, 12])
    position_preferences = [
        {"position": idx + 1, "primary": primary, "secondary": secondary}
        for idx, (primary, secondary) in enumerate(rng.choices(PAIR_LIBRARY, k=length))
    ]

    default_anchors = [1, max(2, length // 2), length]
    anchor_positions = sorted(set(default_anchors))
    forbidden_pool = ["C", "P", "M"]
    forbidden_residues = sorted(rng.sample(forbidden_pool, k=rng.choice([0, 1, 1, 2])))

    latent_spec = {
        "target_id": f"{split.upper()}-{example_idx:03d}",
        "length": length,
        "position_preferences": position_preferences,
        "anchor_positions": anchor_positions,
        "forbidden_residues": forbidden_residues,
        "max_special_residues": 1,
        "weights": DEFAULT_WEIGHTS,
    }

    best_sequence = ""
    best_score = -1.0
    candidate = _sample_sequence(rng, latent_spec)
    for _ in range(300):
        if best_sequence and rng.random() < 0.6:
            candidate = _mutate_sequence(rng, best_sequence, latent_spec)
        else:
            candidate = _sample_sequence(rng, latent_spec)
        score = score_sequence(candidate, {**latent_spec, "desired_counts": classify_counts(candidate), "target_charge": net_charge(candidate)})["reward"]
        if score > best_score:
            best_sequence = candidate
            best_score = score

    desired_counts = classify_counts(best_sequence)
    spec = {
        **latent_spec,
        "desired_counts": desired_counts,
        "target_charge": net_charge(best_sequence),
        "reference_sequence": best_sequence,
        "reference_score": round(best_score, 3),
    }
    spec["prompt_json"] = json.dumps(public_target_spec(spec), indent=2)
    return spec


def public_target_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_id": spec["target_id"],
        "length": spec["length"],
        "anchor_positions": spec["anchor_positions"],
        "forbidden_residues": spec["forbidden_residues"],
        "target_charge": spec["target_charge"],
        "desired_counts": spec["desired_counts"],
        "position_preferences": spec["position_preferences"],
        "max_special_residues": spec["max_special_residues"],
    }


def make_prompt(spec: dict[str, Any]) -> str:
    bullet_lines = [
        f"- Binder length: {spec['length']} residues",
        f"- Desired net charge: {spec['target_charge']:+d}",
        f"- Anchor positions: {', '.join(str(position) for position in spec['anchor_positions'])}",
        f"- Avoid residues: {', '.join(spec['forbidden_residues']) if spec['forbidden_residues'] else 'none'}",
        f"- Keep special residues (C/P) at or below {spec['max_special_residues']}",
        "- Approximate composition targets: "
        + ", ".join(f"{name}={count}" for name, count in spec["desired_counts"].items()),
    ]

    slot_lines = [
        f"  {slot['position']}: prefer {slot['primary']} (backup {slot['secondary']})"
        for slot in spec["position_preferences"]
    ]

    return (
        "Design a short peptide binder for the synthetic target pocket below.\n"
        "You may use tools to inspect amino-acid classes or analyze candidate sequences.\n"
        "Return exactly one final candidate sequence inside <sequence> tags.\n\n"
        "Target summary:\n"
        + "\n".join(bullet_lines)
        + "\n\nPosition preferences:\n"
        + "\n".join(slot_lines)
        + "\n\nTarget JSON for tools:\n"
        + spec["prompt_json"]
    )


def generate_rows(num_examples: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for example_idx in range(num_examples):
        spec = build_target_spec(rng, split=split, example_idx=example_idx)
        rows.append(
            {
                "question": make_prompt(spec),
                "answer": spec["reference_sequence"],
                "info": spec,
                "task": "protein-binder",
            }
        )
    return rows


def sequence_similarity_to_reference(sequence: str, reference: str) -> float:
    normalized_sequence = normalize_sequence(sequence)
    normalized_reference = normalize_sequence(reference)
    if not normalized_sequence or len(normalized_sequence) != len(normalized_reference):
        return 0.0
    matches = sum(
        1
        for lhs, rhs in zip(normalized_sequence, normalized_reference, strict=True)
        if lhs == rhs
    )
    return matches / len(normalized_reference)


def render_sequence_profile(sequence: str) -> str:
    return json.dumps(analyze_sequence_profile(sequence), indent=2, sort_keys=True)


def render_target_comparison(sequence: str, target_spec_json: str) -> str:
    spec = json.loads(target_spec_json)
    report = score_sequence(sequence, spec)
    profile = analyze_sequence_profile(sequence)
    combined = {
        "sequence_profile": profile,
        "target_score_breakdown": report,
    }
    return json.dumps(combined, indent=2, sort_keys=True)


def residue_reference(residue: str) -> str:
    residue = (residue or "").strip().upper()
    if residue not in PROPERTIES:
        return json.dumps({"error": f"Unknown residue '{residue}'"}, indent=2)
    payload = {
        "residue": residue,
        "classes": PROPERTIES[residue]["classes"],
        "charge": PROPERTIES[residue]["charge"],
        "hydropathy_hint": PROPERTIES[residue]["hydropathy_hint"],
    }
    return json.dumps(payload, indent=2, sort_keys=True)
