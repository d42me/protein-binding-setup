from __future__ import annotations

from copy import deepcopy
from datasets import Dataset

INSULIN_TARGET_SEQUENCE = (
    "EVCPGMDIRNNLTRLHELENCSVIEGHLQILLMFKTRPEDFRDLSFPKLIMITDYLLLFRVYGLESLKDLFPNLTVIRGSRLFFNYALVIFEMVHLKELGLYNLMNITRGSVRIEKNNELCYLATIDWSRILDSVEDNHIVLNKDDNEEC"
)

BASE_TASK = {
    "task": "protein-binder-monomer-real",
    "target_id": "insulin-chain-a",
    "target_sequence": INSULIN_TARGET_SEQUENCE,
    "remote_target_pdb": "/home/ubuntu/protein-runtime/work/input_pdbs/insulin_target.pdb",
    "target_chain": "A",
    "hotspots": ["A59", "A83", "A91"],
    "binder_length_min": 40,
    "binder_length_max": 55,
    "num_designs": 8,
    "num_seqs_per_backbone": 8,
    "candidate_batch_size": 16,
    "quality_gate": {
        "min_target_mean_plddt": 80.0,
        "min_binder_mean_plddt": 80.0,
        "max_binder_distance_rmse": 1.5,
        "min_hotspot_fraction": 0.33,
        "min_interface_residue_contacts": 10,
        "score_threshold": 0.72,
    },
}

SYSTEM_PROMPT = """You are running a real monomer-only protein binder design pipeline on a remote RTX 6000 GPU host.
Follow the tools in strict order.
Use all stages exactly once:
1. run_target_monomer
2. run_rfdiffusion
3. run_proteinmpnn
4. run_binder_monomer
5. summarize_candidates
Then answer with only the final passing binder sequence as:
<sequence>SEQUENCE</sequence>
Do not add prose.
If summarize_candidates reports at least one passing candidate, submit one of those passing sequences.
Reserve one turn for the final answer."""


def make_prompt(task: dict) -> str:
    hotspots = ", ".join(task["hotspots"])
    gate = task["quality_gate"]
    return (
        "Run the monomer-only protein binder pipeline using the provided tools.\n"
        "The remote rollout is preconfigured with a target structure source, hotspot set, and search budget.\n"
        "Your goal is to produce a binder sequence that passes the monomer-only quality gate.\n\n"
        f"Target ID: {task['target_id']}\n"
        f"Target chain: {task['target_chain']}\n"
        f"Hotspots: {hotspots}\n"
        f"Binder length range: {task['binder_length_min']}-{task['binder_length_max']}\n"
        f"RFdiffusion backbones to sample: {task['num_designs']}\n"
        f"ProteinMPNN sequences per backbone: {task['num_seqs_per_backbone']}\n\n"
        "Monomer-only quality gate:\n"
        f"- target mean pLDDT >= {gate['min_target_mean_plddt']}\n"
        f"- binder mean pLDDT >= {gate['min_binder_mean_plddt']}\n"
        f"- binder distance RMSE <= {gate['max_binder_distance_rmse']}\n"
        f"- hotspot fraction >= {gate['min_hotspot_fraction']}\n"
        f"- interface residue contacts >= {gate['min_interface_residue_contacts']}\n"
        f"- monomer plausibility score >= {gate['score_threshold']}\n\n"
        "Target protein sequence:\n"
        f"{task['target_sequence']}"
    )


def build_task_rows(num_examples: int, split: str) -> list[dict]:
    rows: list[dict] = []
    for index in range(num_examples):
        task = deepcopy(BASE_TASK)
        task["target_id"] = f"{split}-insulin-{index:03d}"
        rows.append(
            {
                "question": make_prompt(task),
                "answer": "pass",
                "info": task,
                "task": "protein-binder-monomer-real",
            }
        )
    return rows


def build_datasets(num_train_examples: int, num_eval_examples: int) -> tuple[Dataset, Dataset]:
    return (
        Dataset.from_list(build_task_rows(num_train_examples, split="train")),
        Dataset.from_list(build_task_rows(num_eval_examples, split="eval")),
    )
