from __future__ import annotations

import argparse
import json
import statistics
import urllib.request
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from datasets import load_dataset


DEFAULT_DATASET_ID = "ronig/protein_binding_sequences"
DEFAULT_LENGTH_WINDOWS = [(20, 50), (25, 55), (30, 50), (35, 55), (40, 55)]


@dataclass(frozen=True)
class FilterConfig:
    min_peptide_length: int = 30
    max_peptide_length: int = 50
    min_receptor_length: int = 50
    max_receptor_length: int = 800
    max_sequence_similarity: float = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scout ronig/protein_binding_sequences for monomer-harness-compatible binder tasks."
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default="train")
    parser.add_argument("--min-peptide-length", type=int, default=30)
    parser.add_argument("--max-peptide-length", type=int, default=50)
    parser.add_argument("--min-receptor-length", type=int, default=50)
    parser.add_argument("--max-receptor-length", type=int, default=800)
    parser.add_argument("--max-sequence-similarity", type=float, default=0.8)
    parser.add_argument(
        "--structure-sample-size",
        type=int,
        default=0,
        help="Download a random sample of PDB files and estimate interface-hotspot viability.",
    )
    parser.add_argument("--structure-sample-seed", type=int, default=0)
    parser.add_argument("--contact-cutoff", type=float, default=6.0)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Optional path to write the filtered candidate manifest.",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=10,
        help="Number of filtered examples to print as a preview.",
    )
    return parser.parse_args()


def canonical_pair_key(row: dict) -> tuple[str, str, str]:
    pdb_id = row["protein_pdb_name"].lower()
    chains = tuple(sorted((row["protein_pdb_chain"], row["peptide_pdb_chain"])))
    return (pdb_id, *chains)


def candidate_manifest_row(row: dict, dataset_id: str) -> dict:
    return {
        "candidate_id": f"{row['protein_pdb_name'].lower()}:{row['protein_pdb_chain']}:{row['peptide_pdb_chain']}",
        "pdb_id": row["protein_pdb_name"].lower(),
        "receptor_chain": row["protein_pdb_chain"],
        "peptide_chain": row["peptide_pdb_chain"],
        "train_part": row["train_part"],
        "receptor_sequence": row["receptor"],
        "peptide_sequence": row["peptide"],
        "receptor_length": len(row["receptor"]),
        "peptide_length": len(row["peptide"]),
        "dataset_source": dataset_id,
    }


def is_self_like_pair(row: dict, max_sequence_similarity: float) -> bool:
    peptide = row["peptide"]
    receptor = row["receptor"]
    if peptide == receptor:
        return True
    if peptide in receptor or receptor in peptide:
        return True
    return SequenceMatcher(None, peptide, receptor).ratio() >= max_sequence_similarity


def passes_length_filters(row: dict, config: FilterConfig) -> bool:
    peptide_length = len(row["peptide"])
    receptor_length = len(row["receptor"])
    return (
        config.min_peptide_length <= peptide_length <= config.max_peptide_length
        and config.min_receptor_length <= receptor_length <= config.max_receptor_length
    )


def apply_filters(rows: list[dict], config: FilterConfig) -> tuple[list[tuple[str, int]], list[dict]]:
    stage_counts: list[tuple[str, int]] = [("loaded_rows", len(rows))]

    filtered = [row for row in rows if passes_length_filters(row, config)]
    stage_counts.append(("after_length_windows", len(filtered)))

    filtered = [row for row in filtered if len(row["peptide"]) < len(row["receptor"])]
    stage_counts.append(("after_peptide_shorter_than_receptor", len(filtered)))

    filtered = [row for row in filtered if not is_self_like_pair(row, config.max_sequence_similarity)]
    stage_counts.append(("after_remove_self_like_pairs", len(filtered)))

    deduped: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in filtered:
        key = canonical_pair_key(row)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(row)
    stage_counts.append(("after_dedupe_unordered_chain_pairs", len(deduped)))
    return stage_counts, deduped


def count_length_window_candidates(rows: list[dict], config: FilterConfig, windows: list[tuple[int, int]]) -> list[dict]:
    counts: list[dict] = []
    for min_len, max_len in windows:
        window_config = FilterConfig(
            min_peptide_length=min_len,
            max_peptide_length=max_len,
            min_receptor_length=config.min_receptor_length,
            max_receptor_length=config.max_receptor_length,
            max_sequence_similarity=config.max_sequence_similarity,
        )
        _, filtered = apply_filters(rows, window_config)
        counts.append(
            {
                "peptide_length_window": f"{min_len}-{max_len}",
                "candidate_count": len(filtered),
            }
        )
    return counts


def print_preview(rows: list[dict], limit: int, dataset_id: str) -> None:
    preview = [candidate_manifest_row(row, dataset_id) for row in rows[:limit]]
    print("preview_candidates=")
    print(json.dumps(preview, indent=2))


def write_manifest(path: Path, rows: list[dict], dataset_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(candidate_manifest_row(row, dataset_id)) + "\n")


def fetch_pdb_text(pdb_id: str, cache: dict[str, str]) -> str:
    if pdb_id not in cache:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        cache[pdb_id] = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", errors="ignore")
    return cache[pdb_id]


def parse_atom_lines(pdb_text: str) -> list[tuple[str, str, str, float, float, float]]:
    atoms: list[tuple[str, str, str, float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        chain_id = line[21].strip() or " "
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        atoms.append((chain_id, residue_number, insertion_code, x, y, z))
    return atoms


def residue_label(chain_id: str, residue_number: str, insertion_code: str) -> str:
    return f"{chain_id}{residue_number}{insertion_code}".strip()


def estimate_interface_residue_counts(
    rows: list[dict],
    sample_size: int,
    seed: int,
    contact_cutoff: float,
) -> dict | None:
    if sample_size <= 0 or not rows:
        return None

    import random

    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    cutoff_squared = contact_cutoff * contact_cutoff
    cache: dict[str, str] = {}
    interface_counts: list[int] = []
    examples: list[dict] = []

    for row in sample:
        pdb_text = fetch_pdb_text(row["protein_pdb_name"].lower(), cache)
        atoms = parse_atom_lines(pdb_text)
        receptor_atoms = [atom for atom in atoms if atom[0] == row["protein_pdb_chain"]]
        peptide_atoms = [atom for atom in atoms if atom[0] == row["peptide_pdb_chain"]]
        contacting_residues: set[tuple[str, str, str]] = set()

        for receptor_atom in receptor_atoms:
            for peptide_atom in peptide_atoms:
                dx = receptor_atom[3] - peptide_atom[3]
                dy = receptor_atom[4] - peptide_atom[4]
                dz = receptor_atom[5] - peptide_atom[5]
                if dx * dx + dy * dy + dz * dz <= cutoff_squared:
                    contacting_residues.add((receptor_atom[0], receptor_atom[1], receptor_atom[2]))

        interface_count = len(contacting_residues)
        interface_counts.append(interface_count)
        examples.append(
            {
                "candidate_id": f"{row['protein_pdb_name'].lower()}:{row['protein_pdb_chain']}:{row['peptide_pdb_chain']}",
                "interface_residue_count": interface_count,
                "example_hotspots": [
                    residue_label(chain_id, residue_number, insertion_code)
                    for chain_id, residue_number, insertion_code in sorted(contacting_residues)[:5]
                ],
            }
        )

    return {
        "sample_size": len(sample),
        "contact_cutoff": contact_cutoff,
        "min_interface_residue_count": min(interface_counts),
        "median_interface_residue_count": statistics.median(interface_counts),
        "max_interface_residue_count": max(interface_counts),
        "fraction_with_at_least_3_interface_residues": sum(count >= 3 for count in interface_counts) / len(interface_counts),
        "fraction_with_at_least_5_interface_residues": sum(count >= 5 for count in interface_counts) / len(interface_counts),
        "examples": examples[:10],
    }


def main() -> None:
    args = parse_args()
    config = FilterConfig(
        min_peptide_length=args.min_peptide_length,
        max_peptide_length=args.max_peptide_length,
        min_receptor_length=args.min_receptor_length,
        max_receptor_length=args.max_receptor_length,
        max_sequence_similarity=args.max_sequence_similarity,
    )

    rows = list(load_dataset(args.dataset_id, split=args.split))
    stage_counts, filtered = apply_filters(rows, config)
    length_window_counts = count_length_window_candidates(rows, config, DEFAULT_LENGTH_WINDOWS)
    structure_sample_summary = estimate_interface_residue_counts(
        filtered,
        sample_size=args.structure_sample_size,
        seed=args.structure_sample_seed,
        contact_cutoff=args.contact_cutoff,
    )

    summary = {
        "dataset_id": args.dataset_id,
        "split": args.split,
        "filter_config": {
            "min_peptide_length": config.min_peptide_length,
            "max_peptide_length": config.max_peptide_length,
            "min_receptor_length": config.min_receptor_length,
            "max_receptor_length": config.max_receptor_length,
            "max_sequence_similarity": config.max_sequence_similarity,
        },
        "stage_counts": [{"stage": stage, "count": count} for stage, count in stage_counts],
        "length_window_counts": length_window_counts,
        "train_part_counts": {
            key: sum(row["train_part"] == key for row in filtered)
            for key in sorted({row["train_part"] for row in filtered})
        },
        "structure_sample_summary": structure_sample_summary,
    }

    print(json.dumps(summary, indent=2))
    print_preview(filtered, args.preview_limit, args.dataset_id)

    if args.output_jsonl is not None:
        write_manifest(args.output_jsonl, filtered, args.dataset_id)
        print(f"wrote_manifest={args.output_jsonl}")


if __name__ == "__main__":
    main()
