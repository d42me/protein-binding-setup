#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from datasets import load_dataset

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
INTERFACE_RESIDUE_PRIORITY = {
    "W": 10,
    "Y": 10,
    "F": 9,
    "H": 8,
    "R": 8,
    "K": 8,
    "D": 7,
    "E": 7,
    "Q": 6,
    "N": 6,
    "S": 5,
    "T": 5,
    "L": 4,
    "I": 4,
    "V": 4,
    "M": 4,
    "C": 4,
    "A": 3,
    "P": 2,
    "G": 1,
}


def clean_sequence(value: Any) -> str:
    return re.sub(r"\s+", "", str(value or "").upper())


def valid_sequence(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= AMINO_ACIDS


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def likely_homodimer(receptor_sequence: str, ligand_sequence: str) -> bool:
    if receptor_sequence == ligand_sequence:
        return True
    length_ratio = min(len(receptor_sequence), len(ligand_sequence)) / max(len(receptor_sequence), len(ligand_sequence))
    if length_ratio < 0.85:
        return False
    return SequenceMatcher(None, receptor_sequence, ligand_sequence, autojunk=False).ratio() >= 0.78


def row_passes_filters(row: dict[str, Any]) -> bool:
    receptor_sequence = clean_sequence(row.get("receptor_sequence"))
    ligand_sequence = clean_sequence(row.get("ligand_sequence"))
    if not valid_sequence(receptor_sequence) or not valid_sequence(ligand_sequence):
        return False
    if not (60 <= len(receptor_sequence) <= 220):
        return False
    if not (35 <= len(ligand_sequence) <= 80):
        return False
    if likely_homodimer(receptor_sequence, ligand_sequence):
        return False
    if safe_float(row.get("probability")) < 0.55:
        return False
    if safe_int(row.get("intermolecular_contacts")) < 20:
        return False
    if safe_float(row.get("buried_sasa")) < 600.0:
        return False
    return True


def stable_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return slug[:56] or "unknown"


def choose_segment_hotspot(sequence: str, start: int, end: int, jitter: random.Random) -> int:
    candidates = []
    for zero_index in range(max(0, start), min(len(sequence), end)):
        residue = sequence[zero_index]
        priority = INTERFACE_RESIDUE_PRIORITY.get(residue, 0)
        terminus_penalty = 2 if zero_index < 4 or zero_index >= len(sequence) - 4 else 0
        candidates.append((priority - terminus_penalty, jitter.random(), zero_index + 1))
    if not candidates:
        return max(1, min(len(sequence), (start + end) // 2 + 1))
    return max(candidates)[2]


def derive_hotspots(sequence: str, source_id: str, count: int = 3, chain: str = "A") -> list[str]:
    jitter = random.Random(f"pinder-hotspots:{source_id}")
    usable_start = 4 if len(sequence) > 12 else 0
    usable_end = len(sequence) - 4 if len(sequence) > 12 else len(sequence)
    span = max(1, usable_end - usable_start)
    positions: list[int] = []
    for segment_index in range(count):
        start = usable_start + int(span * segment_index / count)
        end = usable_start + int(span * (segment_index + 1) / count)
        position = choose_segment_hotspot(sequence, start, end, jitter)
        if position not in positions:
            positions.append(position)
    while len(positions) < count:
        fallback = 1 + int((len(sequence) - 1) * (len(positions) + 1) / (count + 1))
        if fallback not in positions:
            positions.append(fallback)
        else:
            break
    return [f"{chain}{position}" for position in sorted(positions[:count])]


def sequence_profile(sequence: str) -> dict[str, Any]:
    charged = sum(sequence.count(residue) for residue in "RHKDE")
    polar = sum(sequence.count(residue) for residue in "STNQYC")
    hydrophobic = sum(sequence.count(residue) for residue in "AILMFWV")
    return {
        "length": len(sequence),
        "charged_fraction": round(charged / len(sequence), 3),
        "polar_fraction": round(polar / len(sequence), 3),
        "hydrophobic_fraction": round(hydrophobic / len(sequence), 3),
    }


def task_from_row(row: dict[str, Any], *, source_split: str, index: int) -> dict[str, Any]:
    receptor_sequence = clean_sequence(row["receptor_sequence"])
    ligand_sequence = clean_sequence(row["ligand_sequence"])
    ligand_length = len(ligand_sequence)
    source_id = str(row["id"])
    binder_length_min = max(32, ligand_length - 8)
    binder_length_max = min(88, ligand_length + 8)
    contacts = safe_int(row.get("intermolecular_contacts"))
    min_contacts = max(8, min(16, round(contacts / 6)))
    return {
        "task": "protein-binder-monomer-real",
        "target_id": f"pinder-{source_split}-{index:03d}-{stable_slug(source_id)}",
        "target_sequence": receptor_sequence,
        "target_chain": "A",
        "hotspots": derive_hotspots(receptor_sequence, source_id),
        "binder_length_min": binder_length_min,
        "binder_length_max": binder_length_max,
        "num_designs": 4,
        "num_seqs_per_backbone": 4,
        "candidate_batch_size": 8,
        "dataset_source": f"Synthyra/PINDER split={source_split} id={source_id} | hotspots=sequence-derived-target-anchors",
        "quality_gate": {
            "min_target_mean_plddt": 70.0,
            "min_binder_mean_plddt": 78.0,
            "max_binder_distance_rmse": 1.75,
            "min_hotspot_fraction": 0.33,
            "min_interface_residue_contacts": min_contacts,
            "score_threshold": 0.68,
        },
        "source_pinder_split": source_split,
        "source_pinder_id": source_id,
        "source_receptor_length": len(receptor_sequence),
        "source_ligand_length": ligand_length,
        "source_ligand_sequence": ligand_sequence,
        "source_ligand_profile": sequence_profile(ligand_sequence),
        "pinder_probability": round(safe_float(row.get("probability")), 4),
        "pinder_link_density": round(safe_float(row.get("link_density")), 4),
        "pinder_planarity": round(safe_float(row.get("planarity")), 4),
        "pinder_n_residue_pairs": safe_int(row.get("n_residue_pairs")),
        "pinder_n_residues": safe_int(row.get("n_residues")),
        "pinder_buried_sasa": round(safe_float(row.get("buried_sasa")), 3),
        "pinder_intermolecular_contacts": contacts,
        "pinder_contact_mix": {
            "charged_charged": safe_int(row.get("charged_charged_contacts")),
            "charged_polar": safe_int(row.get("charged_polar_contacts")),
            "charged_apolar": safe_int(row.get("charged_apolar_contacts")),
            "polar_polar": safe_int(row.get("polar_polar_contacts")),
            "apolar_polar": safe_int(row.get("apolar_polar_contacts")),
            "apolar_apolar": safe_int(row.get("apolar_apolar_contacts")),
        },
        "pinder_structure_flags": {
            "predicted_R": bool(row.get("predicted_R")),
            "predicted_L": bool(row.get("predicted_L")),
            "apo_R": bool(row.get("apo_R")),
            "apo_L": bool(row.get("apo_L")),
            "holo_R": bool(row.get("holo_R")),
            "holo_L": bool(row.get("holo_L")),
        },
        "hotspot_derivation": "Three receptor anchor residues were chosen from sequence thirds using interface-prior residue classes because the HF PINDER table exposes global PPI metadata, not residue-level contact maps.",
    }


def iter_rows(split: str, *, streaming: bool, seed: int) -> Iterable[dict[str, Any]]:
    dataset = load_dataset("Synthyra/PINDER", split=split, streaming=streaming)
    if streaming:
        dataset = dataset.shuffle(seed=seed, buffer_size=20_000)
    else:
        dataset = dataset.shuffle(seed=seed)
    yield from dataset


def collect_tasks(split: str, *, count: int, streaming: bool, seed: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for row in iter_rows(split, streaming=streaming, seed=seed):
        row = dict(row)
        if not row_passes_filters(row):
            continue
        tasks.append(task_from_row(row, source_split=split, index=len(tasks)))
        if len(tasks) >= count:
            return tasks
    raise RuntimeError(f"Only found {len(tasks)} PINDER tasks for split={split}; requested {count}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate compact PINDER target tasks for protein-binder-monomer-real.")
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--eval-count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("environments/protein_binder_monomer_real/protein_binder_monomer_real/data/pinder_curated_tasks.json"),
    )
    args = parser.parse_args()

    train_tasks = collect_tasks("train", count=args.train_count, streaming=True, seed=args.seed)
    valid_count = args.eval_count // 2
    test_count = args.eval_count - valid_count
    eval_tasks = [
        *collect_tasks("valid", count=valid_count, streaming=False, seed=args.seed + 1),
        *collect_tasks("test", count=test_count, streaming=False, seed=args.seed + 2),
    ]
    payload = {
        "dataset": "Synthyra/PINDER",
        "version": 1,
        "curation": {
            "seed": args.seed,
            "filters": {
                "receptor_length": [60, 220],
                "ligand_length": [35, 80],
                "probability_min": 0.55,
                "intermolecular_contacts_min": 20,
                "buried_sasa_min": 600.0,
                "exclude_exact_or_high_identity_homodimers": True,
            },
            "notes": "HF PINDER rows provide receptor/ligand sequences and global PPI metadata. Residue-level interface contacts are not present, so target hotspots are deterministic sequence-derived anchor hypotheses.",
        },
        "train": train_tasks,
        "eval": eval_tasks,
    }
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "train": len(train_tasks), "eval": len(eval_tasks)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
