from __future__ import annotations

import json
import random
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

SYSTEM_PROMPT = """You are a scientific protein binder design analyst operating a monomer-only binder search workflow.
Principles:
- Each rollout starts with only the target specification. No backbones, candidates, or scores exist yet.
- You may call the tools in your own order and repeat stages when useful.
- The real dependency structure still matters: if you change an upstream stage, downstream evidence becomes stale and should be rebuilt before final selection.
- Use tools to explore, compare, and refine candidate evidence until you are confident in the strongest candidate.
- Keep reasoning short, explicit, and evidence-linked. Do not restate the task or speculate repeatedly.
- Before the final answer, every assistant turn must include a tool call.
- The main scientific objective is linear and visible in the returned metrics: prefer higher monomer plausibility, higher binder pLDDT, lower binder distance RMSE, higher hotspot fraction, and higher interface contacts.
- Tool use is free up to 20 total tool calls. After that, additional tool calls are increasingly penalized, so explore deliberately instead of repeating stages aimlessly.
- The final answer must be only:
<candidate_id>CANDIDATE_ID</candidate_id>
- Do not output raw sequence text in the final answer."""


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
        "Use the scientific tools to build evidence for promising binder candidates and then select the strongest candidate ID.\n"
        "Each rollout starts empty: no derived structures, candidates, or scores exist until you produce them with tools.\n"
        "You may call tools in your own order and repeat them if needed, but upstream changes make downstream evidence stale.\n"
        "Before the final answer, every assistant turn must include a tool call.\n"
        "Scientific selection objective: maximize candidate quality using the visible metrics. The main reward increases linearly with higher monomer plausibility, higher binder mean pLDDT, lower binder distance RMSE, higher hotspot fraction, and higher interface residue contacts.\n"
        "Exploration budget: the first 20 tool calls are free. After that, extra tool calls are increasingly penalized, so only repeat stages when the extra evidence is worth it.\n\n"
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


def build_task_rows(
    num_examples: int,
    split: str,
    task_library: str = "all",
    seed: int = 0,
) -> list[dict]:
    task_pool = get_task_library(task_library)
    ordering = list(range(len(task_pool)))
    random.Random(f"{task_library}:{split}:{seed}").shuffle(ordering)

    rows: list[dict] = []
    for index in range(num_examples):
        task = deepcopy(task_pool[ordering[index % len(ordering)]])
        task["target_id"] = f"{split}-{task['target_id']}-{index:03d}"
        task["task_library"] = task_library
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
    train_seed: int = 7,
    eval_seed: int = 17,
) -> tuple[Dataset, Dataset]:
    return (
        Dataset.from_list(build_task_rows(num_train_examples, split="train", task_library=task_library, seed=train_seed)),
        Dataset.from_list(build_task_rows(num_eval_examples, split="eval", task_library=task_library, seed=eval_seed)),
    )
