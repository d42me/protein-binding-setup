from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from datasets import Dataset

PROVEN_TASKS = [
    {
        "task": "protein-binder-monomer-real",
        "target_id": "rfdiffusion-insulin-chain-a",
        "target_sequence": "EVCPGMDIRNNLTRLHELENCSVIEGHLQILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYALVIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNHIVLNKDDNEEC",
        "target_chain": "A",
        "hotspots": ["A59", "A83", "A91"],
        "binder_length_min": 40,
        "binder_length_max": 55,
        "num_designs": 8,
        "num_seqs_per_backbone": 8,
        "candidate_batch_size": 16,
        "dataset_source": "RFdiffusion examples/design_ppi.sh",
        "quality_gate": {
            "min_target_mean_plddt": 80.0,
            "min_binder_mean_plddt": 80.0,
            "max_binder_distance_rmse": 1.5,
            "min_hotspot_fraction": 0.33,
            "min_interface_residue_contacts": 10,
            "score_threshold": 0.72,
        },
    },
    {
        "task": "protein-binder-monomer-real",
        "target_id": "rfdiffusion-gabarap-chain-a",
        "target_sequence": "MKFVYKEEHPFEKRRSEGEKIRKKYPDRVPVIVEKAPKARIGDLDKKKYLVPSDLTVGQFYFLIRKRIHLRAEDALFFFVNNVIPPTSATMGQLYQEHHEEDFFLYIAYSDESVY",
        "target_chain": "A",
        "hotspots": ["A46", "A48", "A60"],
        "binder_length_min": 40,
        "binder_length_max": 55,
        "num_designs": 8,
        "num_seqs_per_backbone": 8,
        "candidate_batch_size": 16,
        "dataset_source": "RFdiffusion examples/design_macrocyclic_binder.sh with sequential hotspot remap",
        "quality_gate": {
            "min_target_mean_plddt": 80.0,
            "min_binder_mean_plddt": 80.0,
            "max_binder_distance_rmse": 1.5,
            "min_hotspot_fraction": 0.33,
            "min_interface_residue_contacts": 10,
            "score_threshold": 0.72,
        },
    },
    {
        "task": "protein-binder-monomer-real",
        "target_id": "rfdiffusion-1ycr-chain-a",
        "target_sequence": "ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVV",
        "target_chain": "A",
        "hotspots": ["A27", "A47", "A70"],
        "binder_length_min": 40,
        "binder_length_max": 55,
        "num_designs": 8,
        "num_seqs_per_backbone": 8,
        "candidate_batch_size": 16,
        "dataset_source": "RFdiffusion examples/input_pdbs/1YCR.pdb with interface-derived sequential hotspots",
        "quality_gate": {
            "min_target_mean_plddt": 80.0,
            "min_binder_mean_plddt": 80.0,
            "max_binder_distance_rmse": 1.5,
            "min_hotspot_fraction": 0.33,
            "min_interface_residue_contacts": 10,
            "score_threshold": 0.72,
        },
    },
]

CURATED_RONIG_TASKS_PATH = Path(__file__).resolve().parent / "data" / "ronig_curated_tasks.json"
SUPPORTED_TASK_LIBRARIES = {"proven", "ronig", "all"}

SYSTEM_PROMPT = """You are running a real monomer-only protein binder design pipeline on a remote RTX 6000 GPU host.
Protocol:
- Use exactly one tool call per assistant turn while tools remain.
- Do not explain the plan, restate the task, or add prose before tool calls.
- Call the tools in this exact order and exactly once:
  1. run_target_monomer
  2. run_rfdiffusion
  3. run_proteinmpnn
  4. run_binder_monomer
  5. summarize_candidates
- After summarize_candidates, answer with only:
<candidate_id>CANDIDATE_ID</candidate_id>
- Your job is to choose the strongest candidate ID from the surfaced metrics.
- Do not output raw sequence text in the final answer.
- Reserve the final turn for the <candidate_id> answer only."""


def load_curated_ronig_tasks() -> list[dict]:
    return json.loads(CURATED_RONIG_TASKS_PATH.read_text())


def get_task_library(task_library: str = "all") -> list[dict]:
    if task_library not in SUPPORTED_TASK_LIBRARIES:
        supported = ", ".join(sorted(SUPPORTED_TASK_LIBRARIES))
        raise ValueError(f"Unsupported task_library={task_library!r}. Expected one of: {supported}")
    if task_library == "proven":
        return deepcopy(PROVEN_TASKS)
    if task_library == "ronig":
        return deepcopy(load_curated_ronig_tasks())
    return deepcopy(PROVEN_TASKS) + deepcopy(load_curated_ronig_tasks())


def make_prompt(task: dict) -> str:
    hotspots = ", ".join(task["hotspots"])
    gate = task["quality_gate"]

    return (
        "Use the staged tools to find and select the strongest binder candidate from the monomer-only search results.\n"
        "While tools remain, respond with a tool call only. Do not narrate.\n\n"
        f"Target ID: {task['target_id']}\n"
        f"Hotspots: {hotspots}\n"
        f"Binder length range: {task['binder_length_min']}-{task['binder_length_max']}\n"
        f"Search budget: {task['num_designs']} backbones, {task['num_seqs_per_backbone']} sequences per backbone\n"
        "Quality gate:\n"
        f"- target mean pLDDT >= {gate['min_target_mean_plddt']}\n"
        f"- binder mean pLDDT >= {gate['min_binder_mean_plddt']}\n"
        f"- binder distance RMSE <= {gate['max_binder_distance_rmse']}\n"
        f"- hotspot fraction >= {gate['min_hotspot_fraction']}\n"
        f"- interface residue contacts >= {gate['min_interface_residue_contacts']}\n"
        f"- monomer plausibility score >= {gate['score_threshold']}\n\n"
        "Target protein sequence:\n"
        f"{task['target_sequence']}"
    )


def build_task_rows(num_examples: int, split: str, task_library: str = "all") -> list[dict]:
    task_pool = get_task_library(task_library)
    rows: list[dict] = []
    for index in range(num_examples):
        task = deepcopy(task_pool[index % len(task_pool)])
        task["target_id"] = f"{split}-{task['target_id']}-{index:03d}"
        rows.append(
            {
                "question": make_prompt(task),
                "answer": "select_best_candidate_id",
                "info": task,
                "task": "protein-binder-monomer-real",
            }
        )
    return rows


def build_datasets(
    num_train_examples: int,
    num_eval_examples: int,
    task_library: str = "all",
) -> tuple[Dataset, Dataset]:
    return (
        Dataset.from_list(build_task_rows(num_train_examples, split="train", task_library=task_library)),
        Dataset.from_list(build_task_rows(num_eval_examples, split="eval", task_library=task_library)),
    )
