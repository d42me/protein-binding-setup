from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BACKBONE_MATCH = {
    "helix": {"helix": 1.0, "beta": 0.2, "balanced": 0.62},
    "beta": {"helix": 0.2, "beta": 1.0, "balanced": 0.62},
    "balanced": {"helix": 0.58, "beta": 0.58, "balanced": 1.0},
}

TEMPERATURE_MATCH = {
    "low": {"low": 1.0, "medium": 0.64, "high": 0.35},
    "medium": {"low": 0.66, "medium": 1.0, "high": 0.66},
    "high": {"low": 0.35, "medium": 0.64, "high": 1.0},
}

BACKBONE_TEMPLATES = {
    "helix": "AEKLMQRALEKLMQRALEKL",
    "beta": "VYIFTWVYIFTWVYIFTWVI",
    "balanced": "ASTKQNVLHGASTKQNVLHG",
}

SEQUENCE_TEMPLATES = {
    ("helix", "low"): "KLAEALKKLAEA",
    ("helix", "medium"): "QLAEAMRKLAEN",
    ("helix", "high"): "NLAQSYRKLTGT",
    ("beta", "low"): "VWYTRRVWYTRR",
    ("beta", "medium"): "VWYTSNVWYTSN",
    ("beta", "high"): "VWYTGAVWYTGQ",
    ("balanced", "low"): "STKQRRNSTKQR",
    ("balanced", "medium"): "STKQNVLHGAST",
    ("balanced", "high"): "GSTNQAVLGHSA",
}


def load_task(path: str) -> dict:
    return json.loads(Path(path).read_text())


def stable_unit(*parts: object) -> float:
    token = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(token.encode()).hexdigest()
    value = int(digest[:12], 16)
    return value / float(16**12 - 1)


def stable_range(low: float, high: float, *parts: object) -> float:
    return low + (high - low) * stable_unit(*parts)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def write_pdb(path: str, label: str, sequence: str) -> None:
    lines = [
        f"HEADER    SYNTHETIC {label.upper()}",
        f"REMARK    SEQUENCE {sequence}",
    ]
    for index, residue in enumerate(sequence[:40], start=1):
        lines.append(
            f"ATOM  {index:5d}  CA  {residue} A{index:4d}    {index % 7:8.3f}{index % 5:8.3f}{index % 3:8.3f}  1.00 20.00           C"
        )
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def write_fasta(path: str, label: str, sequence: str) -> None:
    Path(path).write_text(f">{label}\n{sequence}\n")


def run_alphafold2(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    sequence = Path(args.target_sequence).read_text().splitlines()[-1].strip()
    summary = {
        "node": "AlphaFold2",
        "input_sequence_length": len(sequence),
        "output_pdb": str(Path(args.output_pdb)),
        **task["structure_summary"],
    }
    write_pdb(args.output_pdb, "target_structure", sequence[:40])
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_rfdiffusion(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    target_pdb = Path(args.target_pdb)
    if not target_pdb.exists():
        raise SystemExit("Target structure PDB is missing. Run AlphaFold2 first.")

    design_mode = args.design_mode.lower().strip()
    if design_mode not in BACKBONE_MATCH:
        raise SystemExit(f"Unknown design_mode: {design_mode}")

    optimal_backbone = task["optimal_backbone"]
    confidence = clamp(
        0.48
        + 0.34 * BACKBONE_MATCH[design_mode][optimal_backbone]
        + stable_range(-0.03, 0.03, task["task_seed"], "rfdiffusion", design_mode),
    )
    backbone_sequence = BACKBONE_TEMPLATES[design_mode]
    write_pdb(args.output_pdb, f"binder_backbone_{design_mode}", backbone_sequence)

    summary = {
        "node": "RFdiffusion",
        "design_mode": design_mode,
        "output_pdb": str(Path(args.output_pdb)),
        "backbone_confidence": round(confidence, 3),
        "backbone_length": len(backbone_sequence),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_proteinmpnn(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    backbone_pdb = Path(args.backbone_pdb)
    if not backbone_pdb.exists():
        raise SystemExit("Binder backbone structure is missing. Run RFdiffusion first.")

    design_mode = args.design_mode.lower().strip()
    sampling_temperature = args.sampling_temperature.lower().strip()
    if design_mode not in BACKBONE_MATCH:
        raise SystemExit(f"Unknown design_mode: {design_mode}")
    if sampling_temperature not in TEMPERATURE_MATCH:
        raise SystemExit(f"Unknown sampling_temperature: {sampling_temperature}")

    base_sequence = SEQUENCE_TEMPLATES[(design_mode, sampling_temperature)]
    token = stable_unit(task["task_seed"], "mpnn", design_mode, sampling_temperature)
    shift = int(token * len(base_sequence))
    binder_sequence = base_sequence[shift:] + base_sequence[:shift]
    sequence_quality = clamp(
        0.45
        + 0.36 * TEMPERATURE_MATCH[sampling_temperature][task["optimal_sampling"]]
        + stable_range(-0.025, 0.025, task["task_seed"], "sequence_quality", binder_sequence),
    )

    write_fasta(args.output_fasta, "binder_sequence", binder_sequence)
    summary = {
        "node": "ProteinMPNN",
        "design_mode": design_mode,
        "sampling_temperature": sampling_temperature,
        "binder_sequence": binder_sequence,
        "output_fasta": str(Path(args.output_fasta)),
        "sequence_quality": round(sequence_quality, 3),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_alphafold_multimer(args: argparse.Namespace) -> None:
    task = load_task(args.task)
    target_pdb = Path(args.target_pdb)
    binder_fasta = Path(args.binder_fasta)
    if not target_pdb.exists():
        raise SystemExit("Target structure PDB is missing. Run AlphaFold2 first.")
    if not binder_fasta.exists():
        raise SystemExit("Binder sequence FASTA is missing. Run ProteinMPNN first.")

    design_mode = args.design_mode.lower().strip()
    sampling_temperature = args.sampling_temperature.lower().strip()
    if design_mode not in BACKBONE_MATCH:
        raise SystemExit(f"Unknown design_mode: {design_mode}")
    if sampling_temperature not in TEMPERATURE_MATCH:
        raise SystemExit(f"Unknown sampling_temperature: {sampling_temperature}")

    binder_sequence = binder_fasta.read_text().splitlines()[-1].strip()
    backbone_fit = BACKBONE_MATCH[design_mode][task["optimal_backbone"]]
    temperature_fit = TEMPERATURE_MATCH[sampling_temperature][task["optimal_sampling"]]
    score = clamp(
        0.42
        + 0.24 * backbone_fit
        + 0.16 * temperature_fit
        + task["headroom"]
        + stable_range(-0.02, 0.02, task["task_seed"], "multimer", binder_sequence),
    )
    iptm = clamp(score - 0.06 + stable_range(-0.02, 0.02, task["task_seed"], "iptm", binder_sequence))
    binder_plddt = clamp(0.58 + 0.3 * temperature_fit + stable_range(-0.02, 0.02, task["task_seed"], "plddt", binder_sequence))
    contacts = int(round(12 + 24 * backbone_fit + 8 * temperature_fit))
    threshold = float(task["threshold"])
    payload = {
        "node": "AlphaFold-Multimer",
        "design_mode": design_mode,
        "sampling_temperature": sampling_temperature,
        "binder_sequence": binder_sequence,
        "structural_plausibility": round(score, 3),
        "threshold": round(threshold, 3),
        "passes_threshold": bool(score > threshold),
        "iptm": round(iptm, 3),
        "binder_plddt": round(binder_plddt, 3),
        "interface_contacts": contacts,
        "output_json": str(Path(args.output_json)),
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", required=True)

    alphafold2 = subparsers.add_parser("alphafold2", parents=[common])
    alphafold2.add_argument("--target-sequence", required=True)
    alphafold2.add_argument("--output-pdb", required=True)
    alphafold2.add_argument("--output-json", required=True)
    alphafold2.set_defaults(func=run_alphafold2)

    rfdiffusion = subparsers.add_parser("rfdiffusion", parents=[common])
    rfdiffusion.add_argument("--target-pdb", required=True)
    rfdiffusion.add_argument("--design-mode", required=True)
    rfdiffusion.add_argument("--output-pdb", required=True)
    rfdiffusion.add_argument("--output-json", required=True)
    rfdiffusion.set_defaults(func=run_rfdiffusion)

    proteinmpnn = subparsers.add_parser("proteinmpnn", parents=[common])
    proteinmpnn.add_argument("--backbone-pdb", required=True)
    proteinmpnn.add_argument("--design-mode", required=True)
    proteinmpnn.add_argument("--sampling-temperature", required=True)
    proteinmpnn.add_argument("--output-fasta", required=True)
    proteinmpnn.add_argument("--output-json", required=True)
    proteinmpnn.set_defaults(func=run_proteinmpnn)

    multimer = subparsers.add_parser("alphafold-multimer", parents=[common])
    multimer.add_argument("--target-pdb", required=True)
    multimer.add_argument("--binder-fasta", required=True)
    multimer.add_argument("--design-mode", required=True)
    multimer.add_argument("--sampling-temperature", required=True)
    multimer.add_argument("--output-json", required=True)
    multimer.set_defaults(func=run_alphafold_multimer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
