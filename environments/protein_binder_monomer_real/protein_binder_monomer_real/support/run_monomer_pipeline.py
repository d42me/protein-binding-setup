#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

AA3_TO_AA1 = {
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

COMMANDS = {
    "pipeline",
    "init-run",
    "target-monomer",
    "rfdiffusion",
    "proteinmpnn",
    "binder-monomer",
    "summarize",
}


@dataclass
class ResidueRecord:
    resid: str
    atoms: list[tuple[float, float, float]]
    ca: tuple[float, float, float] | None


@dataclass
class BackboneMetrics:
    backbone_name: str
    backbone_pdb: str
    binder_chain: str
    binder_length: int
    interface_residue_contacts: int
    hotspot_contacts: int
    hotspot_fraction: float


@dataclass
class CandidateInput:
    candidate_id: str
    backbone_name: str
    backbone_pdb: str
    binder_chain: str
    sample_index: int
    sequence: str
    mpnn_score: float | None
    mpnn_global_score: float | None
    seq_recovery: float | None
    binder_length: int
    interface_residue_contacts: int
    hotspot_contacts: int
    hotspot_fraction: float


@dataclass
class CandidateResult:
    candidate_id: str
    backbone_name: str
    sample_index: int
    sequence: str
    mpnn_score: float | None
    mpnn_global_score: float | None
    seq_recovery: float | None
    binder_length: int
    interface_residue_contacts: int
    hotspot_contacts: int
    hotspot_fraction: float
    binder_mean_plddt: float | None
    binder_ptm: float | None
    binder_distance_rmse: float | None
    monomer_plausibility_score: float
    passes_quality_gate: bool
    quality_gate_failures: list[str]
    binder_score_json: str | None
    binder_pdb: str | None


@dataclass
class QualityGate:
    min_target_mean_plddt: float = 80.0
    min_binder_mean_plddt: float = 80.0
    max_binder_distance_rmse: float = 1.5
    min_hotspot_fraction: float = 0.33
    min_interface_residue_contacts: int = 10
    score_threshold: float = 0.72


@dataclass
class RunConfig:
    run_dir: Path
    target_sequence_fasta: Path | None
    target_pdb: Path | None
    target_chain: str
    hotspots: list[str]
    binder_length_min: int
    binder_length_max: int
    num_designs: int
    num_seqs_per_backbone: int
    sampling_temp: str
    target_msa_mode: str
    binder_msa_mode: str
    colabfold_image: str
    colabfold_cache: Path
    rfdiffusion_reproducible_wrapper: Path
    rfdiffusion_repo: Path
    rfdiffusion_python: Path
    rfdiffusion_models: Path
    proteinmpnn_repo: Path
    proteinmpnn_python: Path
    candidate_batch_size: int
    max_concurrent_binder_batches: int
    quality_gate: QualityGate

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "target_sequence_fasta": str(self.target_sequence_fasta) if self.target_sequence_fasta else None,
            "target_pdb": str(self.target_pdb) if self.target_pdb else None,
            "target_chain": self.target_chain,
            "hotspots": self.hotspots,
            "binder_length_min": self.binder_length_min,
            "binder_length_max": self.binder_length_max,
            "num_designs": self.num_designs,
            "num_seqs_per_backbone": self.num_seqs_per_backbone,
            "sampling_temp": self.sampling_temp,
            "target_msa_mode": self.target_msa_mode,
            "binder_msa_mode": self.binder_msa_mode,
            "colabfold_image": self.colabfold_image,
            "colabfold_cache": str(self.colabfold_cache),
            "rfdiffusion_reproducible_wrapper": str(self.rfdiffusion_reproducible_wrapper),
            "rfdiffusion_repo": str(self.rfdiffusion_repo),
            "rfdiffusion_python": str(self.rfdiffusion_python),
            "rfdiffusion_models": str(self.rfdiffusion_models),
            "proteinmpnn_repo": str(self.proteinmpnn_repo),
            "proteinmpnn_python": str(self.proteinmpnn_python),
            "candidate_batch_size": self.candidate_batch_size,
            "max_concurrent_binder_batches": self.max_concurrent_binder_batches,
            "quality_gate": asdict(self.quality_gate),
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, Any]) -> "RunConfig":
        return cls(
            run_dir=Path(payload["run_dir"]),
            target_sequence_fasta=Path(payload["target_sequence_fasta"]).expanduser() if payload.get("target_sequence_fasta") else None,
            target_pdb=Path(payload["target_pdb"]).expanduser() if payload.get("target_pdb") else None,
            target_chain=payload["target_chain"],
            hotspots=list(payload["hotspots"]),
            binder_length_min=int(payload["binder_length_min"]),
            binder_length_max=int(payload["binder_length_max"]),
            num_designs=int(payload["num_designs"]),
            num_seqs_per_backbone=int(payload["num_seqs_per_backbone"]),
            sampling_temp=payload["sampling_temp"],
            target_msa_mode=payload["target_msa_mode"],
            binder_msa_mode=payload["binder_msa_mode"],
            colabfold_image=payload["colabfold_image"],
            colabfold_cache=Path(payload["colabfold_cache"]).expanduser(),
            rfdiffusion_reproducible_wrapper=Path(payload["rfdiffusion_reproducible_wrapper"]).expanduser(),
            rfdiffusion_repo=Path(payload["rfdiffusion_repo"]).expanduser(),
            rfdiffusion_python=Path(payload["rfdiffusion_python"]).expanduser(),
            rfdiffusion_models=Path(payload["rfdiffusion_models"]).expanduser(),
            proteinmpnn_repo=Path(payload["proteinmpnn_repo"]).expanduser(),
            proteinmpnn_python=Path(payload["proteinmpnn_python"]).expanduser(),
            candidate_batch_size=int(payload["candidate_batch_size"]),
            max_concurrent_binder_batches=int(payload["max_concurrent_binder_batches"]),
            quality_gate=QualityGate(**payload["quality_gate"]),
        )


@dataclass
class RunPaths:
    root: Path
    inputs: Path
    target_monomer: Path
    rfdiffusion: Path
    proteinmpnn: Path
    binder_monomer: Path
    summary: Path
    state: Path

    @classmethod
    def from_root(cls, run_dir: Path) -> "RunPaths":
        return cls(
            root=run_dir,
            inputs=run_dir / "inputs",
            target_monomer=run_dir / "target_monomer",
            rfdiffusion=run_dir / "rfdiffusion",
            proteinmpnn=run_dir / "proteinmpnn",
            binder_monomer=run_dir / "binder_monomer",
            summary=run_dir / "summary",
            state=run_dir / "state",
        )

    def all_dirs(self) -> list[Path]:
        return [
            self.root,
            self.inputs,
            self.target_monomer,
            self.rfdiffusion,
            self.proteinmpnn,
            self.binder_monomer,
            self.summary,
            self.state,
        ]


def normalize_argv(argv: list[str]) -> list[str]:
    if len(argv) <= 1:
        return [argv[0], "pipeline"]
    first = argv[1]
    if first in COMMANDS:
        return argv
    if first.startswith("-"):
        return [argv[0], "pipeline", *argv[1:]]
    return [argv[0], "pipeline", *argv[1:]]


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"\n$ {printable}")
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_run_dirs(run_dir: Path, overwrite: bool) -> RunPaths:
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    paths = RunPaths.from_root(run_dir)
    for path in paths.all_dirs():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def parse_float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except (TypeError, ValueError):
        return None


def normalize_residue_id(resseq: str, insertion: str) -> str:
    base = str(int(resseq.strip())) if resseq.strip() else "0"
    suffix = insertion.strip()
    return f"{base}{suffix}" if suffix else base


def extract_chain_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    residues: list[str] = []
    seen: set[str] = set()
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21].strip() != chain_id:
            continue
        resid = normalize_residue_id(line[22:26], line[26])
        if resid in seen:
            continue
        seen.add(resid)
        resname = line[17:20].strip().upper()
        residues.append(AA3_TO_AA1[resname])
    if not residues:
        raise ValueError(f"No residues found for chain {chain_id} in {pdb_path}")
    return "".join(residues)


def first_matching_path(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} in {directory}")
    return matches[0]


def recursive_matching_path(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched {pattern} under {directory}")
    return matches[0]


def load_residues_by_chain(pdb_path: Path) -> dict[str, list[ResidueRecord]]:
    by_chain: dict[str, dict[str, ResidueRecord]] = {}
    for line in pdb_path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        chain_id = line[21].strip() or "_"
        resid = normalize_residue_id(line[22:26], line[26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        atom_name = line[12:16].strip()
        chain_map = by_chain.setdefault(chain_id, {})
        residue = chain_map.setdefault(resid, ResidueRecord(resid=resid, atoms=[], ca=None))
        residue.atoms.append((x, y, z))
        if atom_name == "CA":
            residue.ca = (x, y, z)
    return {chain: list(records.values()) for chain, records in by_chain.items()}


def squared_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def load_first_chain_ca_coords(pdb_path: Path) -> list[tuple[float, float, float]]:
    residues_by_chain = load_residues_by_chain(pdb_path)
    first_chain = sorted(residues_by_chain)[0]
    coords = [residue.ca for residue in residues_by_chain[first_chain] if residue.ca is not None]
    if not coords:
        raise ValueError(f"No CA atoms found in {pdb_path}")
    return coords


def load_chain_ca_coords(pdb_path: Path, chain_id: str) -> list[tuple[float, float, float]]:
    residues_by_chain = load_residues_by_chain(pdb_path)
    coords = [residue.ca for residue in residues_by_chain[chain_id] if residue.ca is not None]
    if not coords:
        raise ValueError(f"No CA atoms found for chain {chain_id} in {pdb_path}")
    return coords


def pairwise_distance_rmse(a_coords: list[tuple[float, float, float]], b_coords: list[tuple[float, float, float]]) -> float | None:
    if len(a_coords) != len(b_coords) or len(a_coords) < 2:
        return None
    squared_errors: list[float] = []
    for i in range(len(a_coords)):
        for j in range(i + 1, len(a_coords)):
            a_distance = math.sqrt(squared_distance(a_coords[i], a_coords[j]))
            b_distance = math.sqrt(squared_distance(b_coords[i], b_coords[j]))
            squared_errors.append((a_distance - b_distance) ** 2)
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_backbone_metrics(pdb_path: Path, target_chain: str, hotspot_tokens: list[str]) -> BackboneMetrics:
    residues_by_chain = load_residues_by_chain(pdb_path)
    binder_chains = [chain for chain in residues_by_chain if chain != target_chain]
    if not binder_chains:
        raise ValueError(f"Could not find a binder chain in {pdb_path}")
    binder_chain = binder_chains[0]

    target_residues = residues_by_chain[target_chain]
    binder_residues = residues_by_chain[binder_chain]
    hotspot_ids = {token[1:] if token.startswith(target_chain) else token for token in hotspot_tokens}

    interface_cutoff_sq = 8.0 ** 2
    hotspot_cutoff_sq = 6.0 ** 2
    contacting_binder_residues: set[str] = set()
    contacted_hotspots: set[str] = set()

    for binder_residue in binder_residues:
        for target_residue in target_residues:
            min_distance_sq = min(
                squared_distance(binder_atom, target_atom)
                for binder_atom in binder_residue.atoms
                for target_atom in target_residue.atoms
            )
            if min_distance_sq <= interface_cutoff_sq:
                contacting_binder_residues.add(binder_residue.resid)
            if target_residue.resid in hotspot_ids and min_distance_sq <= hotspot_cutoff_sq:
                contacted_hotspots.add(target_residue.resid)

    hotspot_fraction = len(contacted_hotspots) / len(hotspot_ids) if hotspot_ids else 0.0
    return BackboneMetrics(
        backbone_name=pdb_path.stem,
        backbone_pdb=str(pdb_path),
        binder_chain=binder_chain,
        binder_length=len(binder_residues),
        interface_residue_contacts=len(contacting_binder_residues),
        hotspot_contacts=len(contacted_hotspots),
        hotspot_fraction=round(hotspot_fraction, 3),
    )


def compute_monomer_plausibility_score(
    *,
    binder_mean_plddt: float | None,
    binder_distance_rmse: float | None,
    interface_residue_contacts: int,
    hotspot_fraction: float,
) -> float:
    plddt_score = clamp((binder_mean_plddt or 0.0) / 100.0)
    distance_score = 0.0 if binder_distance_rmse is None else clamp(1.0 - (binder_distance_rmse / 2.5))
    interface_score = clamp(interface_residue_contacts / 20.0)
    total = 0.4 * plddt_score + 0.25 * distance_score + 0.2 * hotspot_fraction + 0.15 * interface_score
    return round(total, 3)


def evaluate_quality_gate(
    *,
    candidate: CandidateResult,
    target_mean_plddt: float | None,
    gate: QualityGate,
) -> list[str]:
    failures: list[str] = []
    if target_mean_plddt is None or target_mean_plddt < gate.min_target_mean_plddt:
        failures.append(f"target_mean_plddt<{gate.min_target_mean_plddt}")
    if candidate.binder_mean_plddt is None or candidate.binder_mean_plddt < gate.min_binder_mean_plddt:
        failures.append(f"binder_mean_plddt<{gate.min_binder_mean_plddt}")
    if candidate.binder_distance_rmse is None or candidate.binder_distance_rmse > gate.max_binder_distance_rmse:
        failures.append(f"binder_distance_rmse>{gate.max_binder_distance_rmse}")
    if candidate.hotspot_fraction < gate.min_hotspot_fraction:
        failures.append(f"hotspot_fraction<{gate.min_hotspot_fraction}")
    if candidate.interface_residue_contacts < gate.min_interface_residue_contacts:
        failures.append(f"interface_residue_contacts<{gate.min_interface_residue_contacts}")
    if candidate.monomer_plausibility_score < gate.score_threshold:
        failures.append(f"monomer_plausibility_score<{gate.score_threshold}")
    return failures


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def state_file(paths: RunPaths, name: str) -> Path:
    return paths.state / f"{name}.json"


def write_state(paths: RunPaths, name: str, payload: Any) -> None:
    write_json(state_file(paths, name), payload)


def read_state(paths: RunPaths, name: str) -> Any:
    return read_json(state_file(paths, name))


def write_target_fasta(config: RunConfig, inputs_dir: Path) -> dict[str, Any]:
    fasta_path = inputs_dir / "target.fasta"
    if config.target_sequence_fasta:
        text = config.target_sequence_fasta.read_text().strip().splitlines()
        sequence = "".join(line.strip() for line in text if not line.startswith(">"))
        header = text[0] if text and text[0].startswith(">") else ">target"
        fasta_path.write_text(f"{header}\n{sequence}\n")
        return {
            "target_fasta": str(fasta_path),
            "target_sequence": sequence,
            "target_job_name": header.lstrip(">"),
        }

    if config.target_pdb is None:
        raise ValueError("Missing target PDB input.")
    sequence = extract_chain_sequence_from_pdb(config.target_pdb, config.target_chain)
    job_name = f"target_{config.target_chain}"
    fasta_path.write_text(f">{job_name}\n{sequence}\n")
    return {
        "target_fasta": str(fasta_path),
        "target_sequence": sequence,
        "target_job_name": job_name,
    }


def resolve_docker_gpu_request() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "all"
    devices = [token.strip() for token in visible.split(",") if token.strip()]
    if not devices:
        return "all"
    return f"device={','.join(devices)}"


def run_colabfold_monomer(
    *,
    work_root: Path,
    input_fasta: Path,
    output_dir: Path,
    image: str,
    cache_dir: Path,
    msa_mode: str,
) -> None:
    run_command(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            resolve_docker_gpu_request(),
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"NVIDIA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}",
            "-v",
            f"{cache_dir.resolve()}:/cache",
            "-v",
            f"{work_root.resolve()}:/work",
            image,
            "colabfold_batch",
            "--msa-mode",
            msa_mode,
            "--model-type",
            "alphafold2_ptm",
            "--num-models",
            "1",
            "--num-recycle",
            "1",
            "--overwrite-existing-results",
            f"/work/{input_fasta.relative_to(work_root)}",
            f"/work/{output_dir.relative_to(work_root)}",
        ]
    )


def load_colabfold_summary(output_dir: Path, job_name: str) -> tuple[Path, dict[str, Any]]:
    score_json = first_matching_path(output_dir, f"{job_name}_scores_rank_001_*.json")
    pdb_path = first_matching_path(output_dir, f"{job_name}_unrelaxed_rank_001_*.pdb")
    return pdb_path, json.loads(score_json.read_text())


def run_rfdiffusion(
    *,
    helper_script: Path,
    python_path: Path,
    repo_dir: Path,
    model_dir: Path,
    target_pdb: Path,
    output_prefix: Path,
    target_chain: str,
    target_length: int,
    binder_length_min: int,
    binder_length_max: int,
    hotspots: list[str],
    num_designs: int,
) -> None:
    contig = f"[{target_chain}1-{target_length}/0 {binder_length_min}-{binder_length_max}]"
    run_command(
        [
            str(python_path),
            str(helper_script.resolve()),
            f"inference.output_prefix={output_prefix}",
            f"inference.model_directory_path={model_dir}",
            f"inference.input_pdb={target_pdb}",
            f"inference.num_designs={num_designs}",
            f"contigmap.contigs={contig}",
            f"ppi.hotspot_res=[{','.join(hotspots)}]",
            "denoiser.noise_scale_ca=0",
            "denoiser.noise_scale_frame=0",
        ],
        cwd=repo_dir,
    )


def run_proteinmpnn(
    *,
    python_path: Path,
    repo_dir: Path,
    backbone_pdb: Path,
    binder_chain: str,
    output_dir: Path,
    num_sequences: int,
    sampling_temp: str,
) -> Path:
    run_command(
        [
            str(python_path),
            "protein_mpnn_run.py",
            "--pdb_path",
            str(backbone_pdb),
            "--pdb_path_chains",
            binder_chain,
            "--out_folder",
            str(output_dir),
            "--num_seq_per_target",
            str(num_sequences),
            "--batch_size",
            "1",
            "--sampling_temp",
            sampling_temp,
        ],
        cwd=repo_dir,
    )
    return output_dir / "seqs" / f"{backbone_pdb.stem}.fa"


def parse_proteinmpnn_fasta(fasta_path: Path, backbone: BackboneMetrics) -> list[CandidateInput]:
    lines = [line.strip() for line in fasta_path.read_text().splitlines() if line.strip()]
    entries: list[tuple[str, str]] = []
    current_header: str | None = None
    current_sequence: list[str] = []

    for line in lines:
        if line.startswith(">"):
            if current_header is not None:
                entries.append((current_header, "".join(current_sequence)))
            current_header = line[1:]
            current_sequence = []
        else:
            current_sequence.append(line)
    if current_header is not None:
        entries.append((current_header, "".join(current_sequence)))

    candidates: list[CandidateInput] = []
    for header, sequence in entries:
        if "sample=" not in header:
            continue
        fields = {}
        for piece in header.split(","):
            if "=" not in piece:
                continue
            key, value = piece.strip().split("=", 1)
            fields[key.strip()] = value.strip()
        sample_index = int(fields["sample"])
        candidates.append(
            CandidateInput(
                candidate_id=f"{backbone.backbone_name}_sample{sample_index:02d}",
                backbone_name=backbone.backbone_name,
                backbone_pdb=backbone.backbone_pdb,
                binder_chain=backbone.binder_chain,
                sample_index=sample_index,
                sequence=sequence,
                mpnn_score=parse_float(fields.get("score")),
                mpnn_global_score=parse_float(fields.get("global_score")),
                seq_recovery=parse_float(fields.get("seq_recovery")),
                binder_length=backbone.binder_length,
                interface_residue_contacts=backbone.interface_residue_contacts,
                hotspot_contacts=backbone.hotspot_contacts,
                hotspot_fraction=backbone.hotspot_fraction,
            )
        )
    return candidates


def write_candidate_fasta(candidates: list[CandidateInput], fasta_path: Path) -> None:
    lines: list[str] = []
    for candidate in candidates:
        lines.append(f">{candidate.candidate_id}")
        lines.append(candidate.sequence)
    fasta_path.write_text("\n".join(lines) + "\n")


def load_run_config(run_dir: Path) -> RunConfig:
    config_path = RunPaths.from_root(run_dir).state / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run config not found at {config_path}. Run init-run or pipeline first.")
    return RunConfig.from_jsonable(read_json(config_path))


def save_run_config(paths: RunPaths, config: RunConfig) -> None:
    write_json(paths.state / "run_config.json", config.to_jsonable())


def initialize_run(config: RunConfig, *, overwrite: bool) -> RunPaths:
    paths = ensure_run_dirs(config.run_dir, overwrite)
    save_run_config(paths, config)
    target_info = write_target_fasta(config, paths.inputs)
    write_state(paths, "target_info", target_info)
    return paths


def run_target_monomer_stage(config: RunConfig, paths: RunPaths) -> dict[str, Any]:
    target_info = read_state(paths, "target_info")
    reset_output_dir(paths.target_monomer)
    run_colabfold_monomer(
        work_root=paths.root,
        input_fasta=Path(target_info["target_fasta"]),
        output_dir=paths.target_monomer,
        image=config.colabfold_image,
        cache_dir=config.colabfold_cache,
        msa_mode=config.target_msa_mode,
    )
    target_pdb, target_score_payload = load_colabfold_summary(paths.target_monomer, target_info["target_job_name"])
    target_summary = {
        **target_info,
        "target_pdb": str(target_pdb),
        "target_sequence_length": len(target_info["target_sequence"]),
        "target_mean_plddt": round(fmean(target_score_payload.get("plddt", [])), 3) if target_score_payload.get("plddt") else None,
        "target_ptm": parse_float(target_score_payload.get("ptm")),
        "target_score_payload": target_score_payload,
    }
    write_state(paths, "target_summary", target_summary)
    return target_summary


def run_rfdiffusion_stage(config: RunConfig, paths: RunPaths) -> list[BackboneMetrics]:
    target_summary = read_state(paths, "target_summary")
    reset_output_dir(paths.rfdiffusion)
    output_prefix = paths.rfdiffusion / "binder"
    run_rfdiffusion(
        helper_script=config.rfdiffusion_reproducible_wrapper,
        python_path=config.rfdiffusion_python,
        repo_dir=config.rfdiffusion_repo,
        model_dir=config.rfdiffusion_models,
        target_pdb=Path(target_summary["target_pdb"]),
        output_prefix=output_prefix,
        target_chain=config.target_chain,
        target_length=len(target_summary["target_sequence"]),
        binder_length_min=config.binder_length_min,
        binder_length_max=config.binder_length_max,
        hotspots=config.hotspots,
        num_designs=config.num_designs,
    )
    backbone_pdbs = sorted(paths.rfdiffusion.glob("binder_*.pdb"))
    hotspot_tokens = list(config.hotspots)
    backbones = [compute_backbone_metrics(path, config.target_chain, hotspot_tokens) for path in backbone_pdbs]
    write_state(paths, "backbones", [asdict(backbone) for backbone in backbones])
    return backbones


def run_proteinmpnn_stage(config: RunConfig, paths: RunPaths) -> list[CandidateInput]:
    backbones_payload = read_state(paths, "backbones")
    backbones = [BackboneMetrics(**payload) for payload in backbones_payload]
    reset_output_dir(paths.proteinmpnn)
    candidates: list[CandidateInput] = []
    for backbone in backbones:
        mpnn_dir = paths.proteinmpnn / backbone.backbone_name
        fasta_path = run_proteinmpnn(
            python_path=config.proteinmpnn_python,
            repo_dir=config.proteinmpnn_repo,
            backbone_pdb=Path(backbone.backbone_pdb),
            binder_chain=backbone.binder_chain,
            output_dir=mpnn_dir,
            num_sequences=config.num_seqs_per_backbone,
            sampling_temp=config.sampling_temp,
        )
        candidates.extend(parse_proteinmpnn_fasta(fasta_path, backbone))
    if not candidates:
        raise RuntimeError("ProteinMPNN did not produce any sampled candidate sequences.")
    write_state(paths, "candidates", [asdict(candidate) for candidate in candidates])
    return candidates


def _run_binder_batch(
    *,
    batch_index: int,
    batch_fasta: Path,
    output_dir: Path,
    config: RunConfig,
    work_root: Path,
) -> dict[str, Any]:
    print(f"\nScoring binder batch {batch_index} -> {output_dir.name}")
    run_colabfold_monomer(
        work_root=work_root,
        input_fasta=batch_fasta,
        output_dir=output_dir,
        image=config.colabfold_image,
        cache_dir=config.colabfold_cache,
        msa_mode=config.binder_msa_mode,
    )
    return {
        "batch_index": batch_index,
        "batch_fasta": str(batch_fasta),
        "output_dir": str(output_dir),
    }


def run_binder_monomer_stage(config: RunConfig, paths: RunPaths) -> list[dict[str, Any]]:
    candidates_payload = read_state(paths, "candidates")
    candidates = [CandidateInput(**payload) for payload in candidates_payload]
    if not candidates:
        raise RuntimeError("No candidates available for binder monomer scoring.")

    reset_output_dir(paths.binder_monomer)
    batch_inputs_dir = paths.inputs / "binder_candidate_batches"
    reset_output_dir(batch_inputs_dir)

    batch_specs: list[tuple[int, Path, Path]] = []
    for batch_index, batch_candidates in enumerate(chunked(candidates, config.candidate_batch_size)):
        batch_fasta = batch_inputs_dir / f"batch_{batch_index:03d}.fasta"
        write_candidate_fasta(batch_candidates, batch_fasta)
        batch_output = paths.binder_monomer / f"batch_{batch_index:03d}"
        batch_output.mkdir(parents=True, exist_ok=True)
        batch_specs.append((batch_index, batch_fasta, batch_output))

    results: list[dict[str, Any]] = []
    max_workers = max(1, config.max_concurrent_binder_batches)
    if max_workers == 1 or len(batch_specs) == 1:
        for batch_index, batch_fasta, batch_output in batch_specs:
            results.append(
                _run_binder_batch(
                    batch_index=batch_index,
                    batch_fasta=batch_fasta,
                    output_dir=batch_output,
                    config=config,
                    work_root=paths.root,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_binder_batch,
                    batch_index=batch_index,
                    batch_fasta=batch_fasta,
                    output_dir=batch_output,
                    config=config,
                    work_root=paths.root,
                ): batch_index
                for batch_index, batch_fasta, batch_output in batch_specs
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["batch_index"])

    write_state(paths, "binder_batches", results)
    return results


def find_candidate_artifacts(candidate_id: str, binder_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    score_json = recursive_matching_path(binder_root, f"{candidate_id}_scores_rank_001_*.json")
    binder_pdb = recursive_matching_path(binder_root, f"{candidate_id}_unrelaxed_rank_001_*.pdb")
    return binder_pdb, score_json, read_json(score_json)


def rank_candidates(
    candidates: list[CandidateInput],
    binder_output_root: Path,
    target_mean_plddt: float | None,
    gate: QualityGate,
) -> list[CandidateResult]:
    ranked: list[CandidateResult] = []
    for candidate in candidates:
        binder_pdb, score_json, score_payload = find_candidate_artifacts(candidate.candidate_id, binder_output_root)
        binder_mean_plddt = round(fmean(score_payload.get("plddt", [])), 3) if score_payload.get("plddt") else None
        binder_ptm = parse_float(score_payload.get("ptm"))
        source_coords = load_chain_ca_coords(Path(candidate.backbone_pdb), candidate.binder_chain)
        predicted_coords = load_first_chain_ca_coords(binder_pdb)
        binder_distance_rmse = pairwise_distance_rmse(source_coords, predicted_coords)
        plausibility = compute_monomer_plausibility_score(
            binder_mean_plddt=binder_mean_plddt,
            binder_distance_rmse=binder_distance_rmse,
            interface_residue_contacts=candidate.interface_residue_contacts,
            hotspot_fraction=candidate.hotspot_fraction,
        )
        base_result = CandidateResult(
            candidate_id=candidate.candidate_id,
            backbone_name=candidate.backbone_name,
            sample_index=candidate.sample_index,
            sequence=candidate.sequence,
            mpnn_score=candidate.mpnn_score,
            mpnn_global_score=candidate.mpnn_global_score,
            seq_recovery=candidate.seq_recovery,
            binder_length=candidate.binder_length,
            interface_residue_contacts=candidate.interface_residue_contacts,
            hotspot_contacts=candidate.hotspot_contacts,
            hotspot_fraction=candidate.hotspot_fraction,
            binder_mean_plddt=binder_mean_plddt,
            binder_ptm=binder_ptm,
            binder_distance_rmse=None if binder_distance_rmse is None else round(binder_distance_rmse, 3),
            monomer_plausibility_score=plausibility,
            passes_quality_gate=False,
            quality_gate_failures=[],
            binder_score_json=str(score_json),
            binder_pdb=str(binder_pdb),
        )
        failures = evaluate_quality_gate(candidate=base_result, target_mean_plddt=target_mean_plddt, gate=gate)
        base_result.quality_gate_failures = failures
        base_result.passes_quality_gate = not failures
        ranked.append(base_result)
    return sorted(
        ranked,
        key=lambda item: (
            item.passes_quality_gate,
            item.monomer_plausibility_score,
            item.hotspot_fraction,
            item.binder_mean_plddt or 0.0,
            -(item.binder_distance_rmse or 999.0),
        ),
        reverse=True,
    )


def summarize_stage(config: RunConfig, paths: RunPaths) -> dict[str, Any]:
    target_summary = read_state(paths, "target_summary")
    backbones = [BackboneMetrics(**payload) for payload in read_state(paths, "backbones")]
    candidates = [CandidateInput(**payload) for payload in read_state(paths, "candidates")]
    ranked_candidates = rank_candidates(
        candidates=candidates,
        binder_output_root=paths.binder_monomer,
        target_mean_plddt=parse_float(target_summary.get("target_mean_plddt")),
        gate=config.quality_gate,
    )
    best_candidate = asdict(ranked_candidates[0]) if ranked_candidates else None
    best_passing_candidate = next((asdict(candidate) for candidate in ranked_candidates if candidate.passes_quality_gate), None)
    payload = {
        "quality_gate": asdict(config.quality_gate),
        "target": {
            key: value
            for key, value in target_summary.items()
            if key not in {"target_score_payload"}
        },
        "backbones": [asdict(backbone) for backbone in backbones],
        "ranked_candidates": [asdict(candidate) for candidate in ranked_candidates],
        "best_candidate": best_candidate,
        "best_passing_candidate": best_passing_candidate,
        "num_candidates": len(ranked_candidates),
        "num_passing_candidates": sum(candidate.passes_quality_gate for candidate in ranked_candidates),
        "passing_candidate_ids": [candidate.candidate_id for candidate in ranked_candidates if candidate.passes_quality_gate],
    }
    write_json(paths.summary / "run_summary.json", payload)
    if ranked_candidates:
        with (paths.summary / "ranked_candidates.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(ranked_candidates[0]).keys()))
            writer.writeheader()
            for candidate in ranked_candidates:
                writer.writerow(asdict(candidate))
    return payload


def build_config_from_args(args: argparse.Namespace) -> RunConfig:
    helper_script = Path(__file__).with_name("run_rfdiffusion_blackwell.py")
    hotspots = [token.strip() for token in args.hotspots.split(",") if token.strip()]
    if not hotspots:
        raise ValueError("At least one hotspot residue must be provided.")
    if args.binder_length_min > args.binder_length_max:
        raise ValueError("--binder-length-min must be <= --binder-length-max.")
    return RunConfig(
        run_dir=args.run_dir.expanduser().resolve(),
        target_sequence_fasta=args.target_sequence_fasta.expanduser().resolve() if args.target_sequence_fasta else None,
        target_pdb=args.target_pdb.expanduser().resolve() if args.target_pdb else None,
        target_chain=args.target_chain,
        hotspots=hotspots,
        binder_length_min=args.binder_length_min,
        binder_length_max=args.binder_length_max,
        num_designs=args.num_designs,
        num_seqs_per_backbone=args.num_seqs_per_backbone,
        sampling_temp=args.sampling_temp,
        target_msa_mode=args.target_msa_mode,
        binder_msa_mode=args.binder_msa_mode,
        colabfold_image=args.colabfold_image,
        colabfold_cache=args.colabfold_cache.expanduser(),
        rfdiffusion_reproducible_wrapper=helper_script.resolve(),
        rfdiffusion_repo=args.rfdiffusion_repo.expanduser(),
        rfdiffusion_python=args.rfdiffusion_python.expanduser(),
        rfdiffusion_models=args.rfdiffusion_models.expanduser(),
        proteinmpnn_repo=args.proteinmpnn_repo.expanduser(),
        proteinmpnn_python=args.proteinmpnn_python.expanduser(),
        candidate_batch_size=args.candidate_batch_size,
        max_concurrent_binder_batches=args.max_concurrent_binder_batches,
        quality_gate=QualityGate(
            min_target_mean_plddt=args.min_target_mean_plddt,
            min_binder_mean_plddt=args.min_binder_mean_plddt,
            max_binder_distance_rmse=args.max_binder_distance_rmse,
            min_hotspot_fraction=args.min_hotspot_fraction,
            min_interface_residue_contacts=args.min_interface_residue_contacts,
            score_threshold=args.score_threshold,
        ),
    )


def add_full_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory where all artifacts will be written.")
    parser.add_argument("--target-sequence-fasta", type=Path, help="Target FASTA. Optional if --target-pdb is provided.")
    parser.add_argument("--target-pdb", type=Path, help="Input PDB used to derive the target sequence when FASTA is omitted.")
    parser.add_argument("--target-chain", default="A", help="Target chain ID in the input PDB and monomer outputs.")
    parser.add_argument("--hotspots", required=True, help="Comma-separated hotspot residues, e.g. A59,A83,A91")
    parser.add_argument("--binder-length-min", type=int, required=True, help="Minimum binder length for RFdiffusion.")
    parser.add_argument("--binder-length-max", type=int, required=True, help="Maximum binder length for RFdiffusion.")
    parser.add_argument("--num-designs", type=int, default=1, help="Number of RFdiffusion backbones to sample.")
    parser.add_argument("--num-seqs-per-backbone", type=int, default=4, help="ProteinMPNN samples per RFdiffusion backbone.")
    parser.add_argument("--sampling-temp", default="0.1", help="ProteinMPNN sampling temperature string.")
    parser.add_argument("--target-msa-mode", default="mmseqs2_uniref_env", help="ColabFold MSA mode for the target monomer step.")
    parser.add_argument("--binder-msa-mode", default="single_sequence", help="ColabFold MSA mode for binder monomer scoring.")
    parser.add_argument("--colabfold-image", default="ghcr.io/sokrypton/colabfold:1.6.0-cuda12")
    parser.add_argument("--colabfold-cache", type=Path, default=Path("~/protein-runtime/colabfold-cache").expanduser())
    parser.add_argument("--rfdiffusion-repo", type=Path, default=Path("~/protein-runtime/RFdiffusion-src").expanduser())
    parser.add_argument("--rfdiffusion-python", type=Path, default=Path("~/protein-runtime/rfdiffusion-bw-venv/bin/python").expanduser())
    parser.add_argument("--rfdiffusion-models", type=Path, default=Path("~/protein-runtime/rfdiffusion-models").expanduser())
    parser.add_argument("--proteinmpnn-repo", type=Path, default=Path("~/protein-runtime/ProteinMPNN-src").expanduser())
    parser.add_argument("--proteinmpnn-python", type=Path, default=Path("~/protein-runtime/proteinmpnn-test-venv/bin/python").expanduser())
    parser.add_argument("--candidate-batch-size", type=int, default=16, help="How many binder candidates to score per ColabFold monomer batch.")
    parser.add_argument("--max-concurrent-binder-batches", type=int, default=1, help="How many binder-scoring batches to run concurrently. Default is 1 for single-GPU safety.")
    parser.add_argument("--min-target-mean-plddt", type=float, default=80.0)
    parser.add_argument("--min-binder-mean-plddt", type=float, default=80.0)
    parser.add_argument("--max-binder-distance-rmse", type=float, default=1.5)
    parser.add_argument("--min-hotspot-fraction", type=float, default=0.33)
    parser.add_argument("--min-interface-residue-contacts", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.72)
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing run directory before starting.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real monomer-only protein binder pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pipeline_parser = subparsers.add_parser("pipeline", help="Initialize a run and execute all stages.")
    add_full_run_arguments(pipeline_parser)

    init_parser = subparsers.add_parser("init-run", help="Create the run directory, config, and target FASTA only.")
    add_full_run_arguments(init_parser)

    for name, help_text in [
        ("target-monomer", "Run the target ColabFold monomer stage."),
        ("rfdiffusion", "Run the RFdiffusion stage."),
        ("proteinmpnn", "Run the ProteinMPNN stage."),
        ("binder-monomer", "Run the binder ColabFold monomer batches."),
        ("summarize", "Rank candidates and write the summary artifacts."),
    ]:
        stage_parser = subparsers.add_parser(name, help=help_text)
        stage_parser.add_argument("--run-dir", type=Path, required=True)

    return parser


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def run_pipeline_command(config: RunConfig, overwrite: bool) -> dict[str, Any]:
    paths = initialize_run(config, overwrite=overwrite)
    target_summary = run_target_monomer_stage(config, paths)
    print_json(target_summary)
    backbones = run_rfdiffusion_stage(config, paths)
    print_json([asdict(backbone) for backbone in backbones])
    candidates = run_proteinmpnn_stage(config, paths)
    print_json({"num_candidates": len(candidates)})
    batches = run_binder_monomer_stage(config, paths)
    print_json({"num_batches": len(batches), "batches": batches})
    summary = summarize_stage(config, paths)
    return summary


def main() -> None:
    argv = normalize_argv(sys.argv)
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.command in {"pipeline", "init-run"}:
        config = build_config_from_args(args)
        if config.target_sequence_fasta is None and config.target_pdb is None:
            parser.error("Provide either --target-sequence-fasta or --target-pdb.")
        if args.command == "init-run":
            paths = initialize_run(config, overwrite=args.overwrite)
            print_json({
                "run_dir": str(paths.root),
                "run_config": str(paths.state / "run_config.json"),
                "target_info": read_state(paths, "target_info"),
            })
            return
        summary = run_pipeline_command(config, overwrite=args.overwrite)
        print("\nBest candidate")
        print_json(summary.get("best_candidate"))
        print("\nBest passing candidate")
        print_json(summary.get("best_passing_candidate"))
        print(f"\nSummary written to {config.run_dir / 'summary' / 'run_summary.json'}")
        return

    run_dir = args.run_dir.expanduser().resolve()
    config = load_run_config(run_dir)
    paths = RunPaths.from_root(run_dir)

    if args.command == "target-monomer":
        print_json(run_target_monomer_stage(config, paths))
    elif args.command == "rfdiffusion":
        print_json([asdict(backbone) for backbone in run_rfdiffusion_stage(config, paths)])
    elif args.command == "proteinmpnn":
        print_json({"num_candidates": len(run_proteinmpnn_stage(config, paths))})
    elif args.command == "binder-monomer":
        print_json({"batches": run_binder_monomer_stage(config, paths)})
    elif args.command == "summarize":
        summary = summarize_stage(config, paths)
        print_json({
            "best_candidate": summary.get("best_candidate"),
            "best_passing_candidate": summary.get("best_passing_candidate"),
            "num_passing_candidates": summary.get("num_passing_candidates"),
        })
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
