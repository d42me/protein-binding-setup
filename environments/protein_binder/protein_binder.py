from __future__ import annotations

from datasets import Dataset

import verifiers as vf

from synthetic_dataset import (
    generate_rows,
    public_target_spec,
    render_sequence_profile,
    render_target_comparison,
    residue_reference,
    score_sequence,
    sequence_similarity_to_reference,
)

SYSTEM_PROMPT = """You are designing short peptide binders for synthetic protein-interface targets.
Use tools when helpful.
Return a single final candidate in <sequence> tags using only the 20 standard amino-acid letters.
Do not include extra commentary outside the XML tag in your final answer."""


async def amino_acid_reference(residue: str) -> str:
    """Look up amino-acid properties for a single residue.

    Args:
        residue: Single-letter amino-acid code such as K, W, or D.

    Returns:
        JSON describing residue classes, charge, and a simple chemistry hint.
    """

    return residue_reference(residue)


async def sequence_profile(sequence: str) -> str:
    """Analyze a peptide sequence.

    Args:
        sequence: Candidate peptide sequence using one-letter amino-acid codes.

    Returns:
        JSON with length, net charge, composition counts, repeats, and invalid residues.
    """

    return render_sequence_profile(sequence)


async def compare_to_target(sequence: str, target_spec_json: str) -> str:
    """Compare a candidate peptide against the visible target specification.

    Args:
        sequence: Candidate peptide sequence using one-letter amino-acid codes.
        target_spec_json: The target JSON block copied from the prompt.

    Returns:
        JSON with a sequence profile and an approximate target-fit breakdown.
    """

    return render_target_comparison(sequence, target_spec_json)


TOOLS = [amino_acid_reference, sequence_profile, compare_to_target]


def _sequence_from_completion(completion, parser: vf.XMLParser) -> str:
    parsed = parser.parse_answer(completion)
    return (parsed or "").strip().upper()


async def binding_proxy_reward(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["reward"]


async def constraint_reward(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    scored = score_sequence(sequence, public_target_spec(info))
    if scored["valid"] == 0.0:
        return 0.0
    return (scored["forbidden"] + scored["stability"]) / 2


async def reference_similarity_metric(completion, answer, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return sequence_similarity_to_reference(sequence, answer)


async def anchor_metric(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["anchor"]


async def position_metric(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["position"]


async def composition_metric(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["composition"]


async def charge_metric(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["charge"]


async def valid_sequence_metric(completion, info, parser: vf.XMLParser) -> float:
    sequence = _sequence_from_completion(completion, parser)
    return score_sequence(sequence, public_target_spec(info))["valid"]


def load_environment(
    num_train_examples: int = 96,
    num_eval_examples: int = 24,
    train_seed: int = 7,
    eval_seed: int = 17,
    max_turns: int = 4,
) -> vf.Environment:
    """Build a synthetic peptide-binder design environment.

    Args:
        num_train_examples: Number of synthetic training targets to generate.
        num_eval_examples: Number of synthetic evaluation targets to generate.
        train_seed: Random seed for the training split.
        eval_seed: Random seed for the eval split.
        max_turns: Maximum model turns, including tool-use turns.
    """

    train_dataset = Dataset.from_list(generate_rows(num_train_examples, train_seed, split="train"))
    eval_dataset = Dataset.from_list(generate_rows(num_eval_examples, eval_seed, split="eval"))

    parser = vf.XMLParser(["sequence"], answer_field="sequence")
    rubric = vf.Rubric(
        funcs=[binding_proxy_reward, constraint_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.25, 0.1],
        parser=parser,
    )
    rubric.add_metric(valid_sequence_metric)
    rubric.add_metric(position_metric)
    rubric.add_metric(composition_metric)
    rubric.add_metric(charge_metric)
    rubric.add_metric(anchor_metric)
    rubric.add_metric(reference_similarity_metric)

    return vf.ToolEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        tools=TOOLS,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
    )
