from __future__ import annotations

import argparse
import json
import random
import urllib.error
import urllib.request
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_ID = "ronig/protein_binding_sequences"
STANDARD_RESIDUES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
HYDROPHOBIC = set("AVILMFWYCG")
DEFAULT_QUALITY_GATE = {
    "min_target_mean_plddt": 80.0,
    "min_binder_mean_plddt": 80.0,
    "max_binder_distance_rmse": 1.5,
    "min_hotspot_fraction": 0.33,
    "min_interface_residue_contacts": 10,
    "score_threshold": 0.72,
}


@dataclass(frozen=True)
class FilterConfig:
    min_peptide_length: int = 30
    max_peptide_length: int = 50
    min_receptor_length: int = 50
    max_receptor_length: int = 400
    max_sequence_similarity: float = 0.8
    max_receptor_hydrophobic_fraction: float = 0.48
    max_receptor_hydrophobic_run: int = 13
    max_peptide_hydrophobic_fraction: float = 0.72
    max_peptide_hydrophobic_run: int = 12
    min_interface_residues: int = 5
    num_hotspots: int = 3
    hotspot_contact_cutoff: float = 6.0
    max_tasks: int = 100
    max_tasks_per_receptor_sequence: int = 2
    max_tasks_per_peptide_sequence: int = 2
    num_designs: int = 4
    num_seqs_per_backbone: int = 4
    candidate_batch_size: int = 8
    selection_seed: int = 7
    max_workers: int = 12
    pdb_cache_dir: Path = Path(".cache/ronig_pdbs")


@dataclass(frozen=True)
class ProcessedCandidate:
    task: dict[str, Any]
    receptor_sequence: str
    peptide_sequence: str


@dataclass(frozen=True)
class Rejection:
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate ronig/protein_binding_sequences into bundled monomer env tasks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--preview-limit", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=7)
    parser.add_argument("--max-workers", type=int, default=12)
    return parser.parse_args()


def canonical_pair_key(row: dict) -> tuple[str, str, str]:
    pdb_id = row["protein_pdb_name"].lower()
    chains = tuple(sorted((row["protein_pdb_chain"], row["peptide_pdb_chain"])))
    return (pdb_id, *chains)


def is_self_like_pair(row: dict, max_sequence_similarity: float) -> bool:
    peptide = row["peptide"]
    receptor = row["receptor"]
    if peptide == receptor:
        return True
    if peptide in receptor or receptor in peptide:
        return True
    return SequenceMatcher(None, peptide, receptor).ratio() >= max_sequence_similarity


def passes_first_pass(row: dict, config: FilterConfig) -> bool:
    peptide_length = len(row["peptide"])
    receptor_length = len(row["receptor"])
    return (
        config.min_peptide_length <= peptide_length <= config.max_peptide_length
        and config.min_receptor_length <= receptor_length <= config.max_receptor_length
        and peptide_length < receptor_length
        and not is_self_like_pair(row, config.max_sequence_similarity)
    )


def first_pass_rows(config: FilterConfig) -> list[dict]:
    rows = list(load_dataset(DATASET_ID, split="train"))
    filtered: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        if not passes_first_pass(row, config):
            continue
        key = canonical_pair_key(row)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(row)
    random.Random(config.selection_seed).shuffle(filtered)
    return filtered


def fetch_pdb_text(pdb_id: str, cache_dir: Path) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{pdb_id.lower()}.pdb"
    if cache_path.exists():
        return cache_path.read_text()

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    last_error: Exception | None = None
    for _ in range(3):
        try:
            text = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", errors="ignore")
            cache_path.write_text(text)
            return text
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def parse_structure(pdb_text: str) -> dict[str, list[dict[str, Any]]]:
    chains: dict[str, OrderedDict[tuple[str, str], dict[str, Any]]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc not in ("", "A"):
            continue
        resname = line[17:20].strip()
        residue_code = STANDARD_RESIDUES.get(resname)
        if residue_code is None:
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

        chain_residues = chains.setdefault(chain_id, OrderedDict())
        residue_key = (residue_number, insertion_code)
        residue = chain_residues.setdefault(
            residue_key,
            {
                "residue_number": residue_number,
                "insertion_code": insertion_code,
                "resname": resname,
                "code": residue_code,
                "atoms": [],
            },
        )
        residue["atoms"].append((x, y, z))
    return {chain_id: list(residues.values()) for chain_id, residues in chains.items()}


def chain_sequence(residues: list[dict[str, Any]]) -> str:
    return "".join(residue["code"] for residue in residues)


def hydrophobic_fraction(sequence: str) -> float:
    if not sequence:
        return 1.0
    return sum(residue in HYDROPHOBIC for residue in sequence) / len(sequence)


def longest_hydrophobic_run(sequence: str) -> int:
    best = 0
    current = 0
    for residue in sequence:
        if residue in HYDROPHOBIC:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def compute_interface_contacts(receptor_residues: list[dict[str, Any]], peptide_residues: list[dict[str, Any]], cutoff: float) -> list[tuple[int, int]]:
    cutoff_squared = cutoff * cutoff
    contacts: list[tuple[int, int]] = []
    for residue_index, receptor_residue in enumerate(receptor_residues, start=1):
        contact_count = 0
        for peptide_residue in peptide_residues:
            residue_pair_in_contact = False
            for receptor_atom in receptor_residue["atoms"]:
                for peptide_atom in peptide_residue["atoms"]:
                    dx = receptor_atom[0] - peptide_atom[0]
                    dy = receptor_atom[1] - peptide_atom[1]
                    dz = receptor_atom[2] - peptide_atom[2]
                    if dx * dx + dy * dy + dz * dz <= cutoff_squared:
                        residue_pair_in_contact = True
                        break
                if residue_pair_in_contact:
                    break
            if residue_pair_in_contact:
                contact_count += 1
        if contact_count:
            contacts.append((residue_index, contact_count))
    contacts.sort(key=lambda item: (-item[1], item[0]))
    return contacts


def binder_length_window(peptide_length: int) -> tuple[int, int]:
    minimum = max(30, peptide_length - 2)
    maximum = min(55, peptide_length + 2)
    return minimum, max(minimum, maximum)


def build_task(
    row: dict,
    receptor_sequence: str,
    peptide_sequence: str,
    hotspot_indices: list[int],
    interface_contacts: list[tuple[int, int]],
    config: FilterConfig,
) -> dict[str, Any]:
    binder_length_min, binder_length_max = binder_length_window(len(peptide_sequence))
    source_pdb = row["protein_pdb_name"].lower()
    source_receptor_chain = row["protein_pdb_chain"]
    source_peptide_chain = row["peptide_pdb_chain"]
    return {
        "task": "protein-binder-monomer-real",
        "target_id": f"ronig-{source_pdb}-{source_receptor_chain.lower()}-{source_peptide_chain.lower()}",
        "target_sequence": receptor_sequence,
        "target_chain": "A",
        "hotspots": [f"A{index}" for index in hotspot_indices],
        "binder_length_min": binder_length_min,
        "binder_length_max": binder_length_max,
        "num_designs": config.num_designs,
        "num_seqs_per_backbone": config.num_seqs_per_backbone,
        "candidate_batch_size": config.candidate_batch_size,
        "dataset_source": f"{DATASET_ID} | pdb={source_pdb} receptor_chain={source_receptor_chain} peptide_chain={source_peptide_chain} | hotspots=atom-contact-top{config.num_hotspots}",
        "quality_gate": dict(DEFAULT_QUALITY_GATE),
        "source_pdb_id": source_pdb,
        "source_receptor_chain": source_receptor_chain,
        "source_peptide_chain": source_peptide_chain,
        "source_peptide_sequence": peptide_sequence,
        "source_peptide_length": len(peptide_sequence),
        "source_train_part": row.get("train_part"),
        "interface_residue_count": len(interface_contacts),
        "top_interface_contact_count": interface_contacts[0][1],
    }


def process_row(row: dict, config: FilterConfig) -> ProcessedCandidate | Rejection:
    pdb_id = row["protein_pdb_name"].lower()
    try:
        structure = parse_structure(fetch_pdb_text(pdb_id, config.pdb_cache_dir))
    except Exception:
        return Rejection("pdb_fetch_failed")

    receptor_residues = structure.get(row["protein_pdb_chain"])
    peptide_residues = structure.get(row["peptide_pdb_chain"])
    if not receptor_residues or not peptide_residues:
        return Rejection("missing_chain")

    receptor_sequence = chain_sequence(receptor_residues)
    peptide_sequence = chain_sequence(peptide_residues)
    if receptor_sequence != row["receptor"] or peptide_sequence != row["peptide"]:
        return Rejection("sequence_mismatch")

    receptor_fraction = hydrophobic_fraction(receptor_sequence)
    peptide_fraction = hydrophobic_fraction(peptide_sequence)
    if (
        receptor_fraction > config.max_receptor_hydrophobic_fraction
        or longest_hydrophobic_run(receptor_sequence) > config.max_receptor_hydrophobic_run
        or peptide_fraction > config.max_peptide_hydrophobic_fraction
        or longest_hydrophobic_run(peptide_sequence) > config.max_peptide_hydrophobic_run
    ):
        return Rejection("hydrophobic_filter")

    interface_contacts = compute_interface_contacts(receptor_residues, peptide_residues, config.hotspot_contact_cutoff)
    if len(interface_contacts) < config.min_interface_residues:
        return Rejection("too_few_interface_residues")

    hotspot_indices = [index for index, _ in interface_contacts[: config.num_hotspots]]
    if len(hotspot_indices) < config.num_hotspots:
        return Rejection("too_few_hotspots")

    return ProcessedCandidate(
        task=build_task(row, receptor_sequence, peptide_sequence, hotspot_indices, interface_contacts, config),
        receptor_sequence=receptor_sequence,
        peptide_sequence=peptide_sequence,
    )


def curate_tasks(config: FilterConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates = first_pass_rows(config)
    summary: dict[str, int] = {
        "first_pass_candidates": len(candidates),
        "pdb_fetch_failed": 0,
        "missing_chain": 0,
        "sequence_mismatch": 0,
        "hydrophobic_filter": 0,
        "too_few_interface_residues": 0,
        "too_few_hotspots": 0,
        "duplicate_receptor_sequence_cap": 0,
        "duplicate_peptide_sequence_cap": 0,
    }

    selected: list[dict[str, Any]] = []
    receptor_counts: dict[str, int] = {}
    peptide_counts: dict[str, int] = {}
    batch_size = max(config.max_workers * 4, 32)

    for batch_start in range(0, len(candidates), batch_size):
        if len(selected) >= config.max_tasks:
            break
        batch = candidates[batch_start : batch_start + batch_size]
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            processed_batch = list(executor.map(lambda row: process_row(row, config), batch))

        for processed in processed_batch:
            if len(selected) >= config.max_tasks:
                break
            if isinstance(processed, Rejection):
                summary[processed.reason] = summary.get(processed.reason, 0) + 1
                continue

            receptor_count = receptor_counts.get(processed.receptor_sequence, 0)
            if receptor_count >= config.max_tasks_per_receptor_sequence:
                summary["duplicate_receptor_sequence_cap"] += 1
                continue

            peptide_count = peptide_counts.get(processed.peptide_sequence, 0)
            if peptide_count >= config.max_tasks_per_peptide_sequence:
                summary["duplicate_peptide_sequence_cap"] += 1
                continue

            selected.append(processed.task)
            receptor_counts[processed.receptor_sequence] = receptor_count + 1
            peptide_counts[processed.peptide_sequence] = peptide_count + 1

    summary["selected"] = len(selected)
    summary["unique_receptor_sequences"] = len(receptor_counts)
    summary["unique_peptide_sequences"] = len(peptide_counts)
    return selected, summary


def main() -> None:
    args = parse_args()
    config = FilterConfig(
        max_tasks=args.max_tasks,
        selection_seed=args.selection_seed,
        max_workers=args.max_workers,
    )
    tasks, summary = curate_tasks(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")

    preview = tasks[: args.preview_limit]
    print(json.dumps({"summary": summary, "preview": preview}, indent=2))


if __name__ == "__main__":
    main()
