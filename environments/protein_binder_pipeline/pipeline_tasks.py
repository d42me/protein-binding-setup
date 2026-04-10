from __future__ import annotations

import json
import random
from typing import Any

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
BACKBONE_MODES = ("helix", "beta", "balanced")
SAMPLING_TEMPERATURES = ("low", "medium", "high")
PIPELINE_STAGES = ["AlphaFold2", "RFdiffusion", "ProteinMPNN", "AlphaFold-Multimer"]

HYDROPHOBIC = set("AVILMFWY")
POLAR = set("STNQH")
CHARGED = set("KRDE")
GLY_PRO = set("GP")


def normalize_sequence(text: str | None) -> str:
    if not text:
        return ""
    return "".join(char for char in text.upper() if char.isalpha())


def _random_sequence(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(AMINO_ACIDS) for _ in range(length))


def _fraction(sequence: str, residues: set[str]) -> float:
    if not sequence:
        return 0.0
    return sum(1 for residue in sequence if residue in residues) / len(sequence)


def _net_charge(sequence: str) -> int:
    charge_map = {"K": 1, "R": 1, "D": -1, "E": -1}
    return sum(charge_map.get(residue, 0) for residue in sequence)


def _helix_fraction(sequence: str) -> float:
    helix_residues = set("AEHKLMQR")
    return _fraction(sequence, helix_residues)


def _beta_fraction(sequence: str) -> float:
    beta_residues = set("VIFYWT")
    return _fraction(sequence, beta_residues)


def _surface_motif(helix_fraction: float, beta_fraction: float) -> str:
    if helix_fraction - beta_fraction > 0.08:
        return "helical groove"
    if beta_fraction - helix_fraction > 0.08:
        return "beta edge"
    return "mixed pocket"


def _backbone_preference(helix_fraction: float, beta_fraction: float) -> str:
    if helix_fraction - beta_fraction > 0.08:
        return "helix"
    if beta_fraction - helix_fraction > 0.08:
        return "beta"
    return "balanced"


def _sampling_preference(net_charge: int, polar_fraction: float, flexibility: float) -> str:
    if abs(net_charge) >= 4 or polar_fraction >= 0.34:
        return "low"
    if abs(net_charge) >= 2 or flexibility >= 0.18:
        return "medium"
    return "high"


def _hotspot_profile(net_charge: int, hydrophobic_fraction: float, polar_fraction: float) -> str:
    if abs(net_charge) >= 4:
        return "charged rim"
    if hydrophobic_fraction >= 0.38:
        return "hydrophobic strip"
    if polar_fraction >= 0.3:
        return "polar patch"
    return "mixed patch"


def _flexibility_band(flexibility: float) -> str:
    if flexibility >= 0.18:
        return "high"
    if flexibility >= 0.12:
        return "medium"
    return "low"


def make_prompt(task: dict[str, Any]) -> str:
    return (
        "Run the protein binder pipeline in the exact left-to-right order below using the provided tools.\n"
        "Target protein sequence -> AlphaFold2 -> Target structure PDB -> RFdiffusion -> Binder backbone structure -> ProteinMPNN -> Binder sequence -> AlphaFold-Multimer.\n"
        "The environment gives Reward = 1 only when the final AlphaFold-Multimer structural_plausibility is strictly greater than the task threshold; otherwise Reward = 0.\n"
        "Choose exactly one RFdiffusion design_mode and one ProteinMPNN sampling_temperature, then submit the generated binder sequence.\n"
        "Formatting rule: after AlphaFold-Multimer, respond with only <sequence>GENERATED_SEQUENCE</sequence>. No commentary.\n"
        "Do not invent a sequence: submit the binder sequence that your ProteinMPNN stage actually produced.\n\n"
        f"Target ID: {task['target_id']}\n"
        f"Structural plausibility threshold: {task['threshold']:.3f}\n"
        "Allowed RFdiffusion design_mode values: helix, beta, balanced\n"
        "Allowed ProteinMPNN sampling_temperature values: low, medium, high\n"
        "Heuristic reminders: strong charge or polar surfaces usually prefer lower ProteinMPNN temperature; moderate charge/flexibility suggests medium; neutral rigid surfaces suggest high.\n\n"
        "Target protein sequence:\n"
        f"{task['target_sequence']}"
    )


def build_task_rows(num_examples: int, seed: int, split: str) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    for example_idx in range(num_examples):
        target_length = rng.randint(72, 120)
        target_sequence = _random_sequence(rng, target_length)
        helix_fraction = round(_helix_fraction(target_sequence), 3)
        beta_fraction = round(_beta_fraction(target_sequence), 3)
        hydrophobic_fraction = round(_fraction(target_sequence, HYDROPHOBIC), 3)
        polar_fraction = round(_fraction(target_sequence, POLAR), 3)
        flexibility_fraction = round(_fraction(target_sequence, GLY_PRO), 3)
        net_charge = _net_charge(target_sequence)

        optimal_backbone = _backbone_preference(helix_fraction, beta_fraction)
        optimal_sampling = _sampling_preference(net_charge, polar_fraction, flexibility_fraction)
        task_seed = seed * 10_000 + example_idx
        pass_possible = rng.random() < 0.72
        threshold = round(0.73 + rng.uniform(-0.03, 0.025), 3)
        headroom = round(rng.uniform(-0.015, 0.025) if pass_possible else rng.uniform(-0.11, -0.06), 3)

        task = {
            "task": "protein-binder-pipeline",
            "target_id": f"{split.upper()}-{example_idx:03d}",
            "target_sequence": target_sequence,
            "threshold": threshold,
            "task_seed": task_seed,
            "pass_possible": pass_possible,
            "headroom": headroom,
            "optimal_backbone": optimal_backbone,
            "optimal_sampling": optimal_sampling,
            "structure_summary": {
                "surface_motif": _surface_motif(helix_fraction, beta_fraction),
                "helix_fraction": helix_fraction,
                "beta_fraction": beta_fraction,
                "surface_charge": net_charge,
                "hydrophobic_fraction": hydrophobic_fraction,
                "polar_fraction": polar_fraction,
                "flexibility_band": _flexibility_band(flexibility_fraction),
                "hotspot_profile": _hotspot_profile(net_charge, hydrophobic_fraction, polar_fraction),
            },
        }

        rows.append(
            {
                "question": make_prompt(task),
                "answer": "pass" if pass_possible else "fail",
                "info": task,
                "task": "protein-binder-pipeline",
            }
        )

    return rows


def render_task_summary(task: dict[str, Any]) -> str:
    public_payload = {
        "target_id": task["target_id"],
        "threshold": task["threshold"],
        "structure_summary": task["structure_summary"],
    }
    return json.dumps(public_payload, indent=2, sort_keys=True)
