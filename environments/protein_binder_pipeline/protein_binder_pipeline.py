from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from typing import Any

from datasets import Dataset

import verifiers as vf

try:
    from .pipeline_tasks import (
        BACKBONE_MODES,
        PIPELINE_STAGES,
        SAMPLING_TEMPERATURES,
        build_task_rows,
        normalize_sequence,
    )
except ImportError:
    from pipeline_tasks import (
        BACKBONE_MODES,
        PIPELINE_STAGES,
        SAMPLING_TEMPERATURES,
        build_task_rows,
        normalize_sequence,
    )

SYSTEM_PROMPT = """You are running a synthetic protein-binder design pipeline inside a Prime sandbox.
Follow the stages in strict left-to-right order.
Choose one RFdiffusion design_mode, then one ProteinMPNN sampling_temperature, then run AlphaFold-Multimer.
Use the AlphaFold2 summary to pick the RFdiffusion mode:
- high helix_fraction or a helical groove -> helix
- high beta_fraction or a beta edge -> beta
- otherwise -> balanced
Use the AlphaFold2 summary to pick the ProteinMPNN sampling_temperature:
- strong charge or very polar surfaces -> low
- moderate charge or some flexibility -> medium
- neutral, rigid surfaces -> high
After the multimer stage, respond with the generated binder sequence only, exactly as:
<sequence>SEQUENCE</sequence>
Do not add prose, analysis, or extra tags. Do not invent a sequence and do not skip stages."""

RUNNER_SOURCE = Path(__file__).with_name("sandbox_runner.py").read_text()
EXPECTED_STAGE_SET = set(PIPELINE_STAGES)


def _parse_submitted_sequence(completion: vf.Messages, parser: vf.XMLParser) -> str:
    for msg in reversed(parser.get_assistant_messages(completion)):
        content = parser._content_to_text(
            msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
        )
        parsed = parser.parse(content, last=True)
        raw_sequence = getattr(parsed, "sequence", None) or ""
        normalized = normalize_sequence(raw_sequence)
        if normalized:
            return normalized
    return ""


async def structural_pass_reward(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    submitted_sequence = _parse_submitted_sequence(completion, parser)
    generated_sequence = state.get("generated_binder_sequence", "")
    if not submitted_sequence or submitted_sequence != generated_sequence:
        return 0.0
    if state.get("stage_history", []) != PIPELINE_STAGES:
        return 0.0
    return 1.0 if state.get("passes_threshold", False) else 0.0


async def submitted_sequence_matches_candidate(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    submitted_sequence = _parse_submitted_sequence(completion, parser)
    return 1.0 if submitted_sequence and submitted_sequence == state.get("generated_binder_sequence", "") else 0.0


async def pipeline_completed_metric(completion: vf.Messages, state: vf.State) -> float:
    return 1.0 if state.get("stage_history", []) == PIPELINE_STAGES else 0.0


async def structural_plausibility_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("multimer_report", {}).get("structural_plausibility", 0.0))


async def threshold_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("multimer_report", {}).get("threshold", state.get("info", {}).get("threshold", 0.0)))


async def pass_margin_metric(completion: vf.Messages, state: vf.State) -> float:
    report = state.get("multimer_report", {})
    return float(report.get("structural_plausibility", 0.0)) - float(report.get("threshold", state.get("info", {}).get("threshold", 0.0)))


async def passes_threshold_metric(completion: vf.Messages, state: vf.State) -> float:
    return 1.0 if state.get("passes_threshold", False) else 0.0


async def used_optimal_backbone_metric(completion: vf.Messages, state: vf.State) -> float:
    info = dict(state.get("info", {}) or {})
    return 1.0 if state.get("selected_design_mode") == info.get("optimal_backbone") else 0.0


async def used_optimal_sampling_metric(completion: vf.Messages, state: vf.State) -> float:
    info = dict(state.get("info", {}) or {})
    return 1.0 if state.get("selected_sampling_temperature") == info.get("optimal_sampling") else 0.0


async def stage_error_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("stage_error_count", 0))


class ProteinBinderPipelineEnv(vf.SandboxEnv):
    TOOL_NAMES = {
        "run_alphafold2",
        "run_rfdiffusion",
        "run_proteinmpnn",
        "run_alphafold_multimer",
    }

    def __init__(self, labels: list[str] | None = None, **kwargs):
        super().__init__(
            labels=labels or ["protein-binder-pipeline"],
            timeout_per_command_seconds=60,
            **kwargs,
        )
        self.remove_tool(self.bash)
        self.add_tool(
            self.run_alphafold2,
            args_to_skip=["sandbox_id", "sandbox_state", "working_dir", "state"],
        )
        self.add_tool(
            self.run_rfdiffusion,
            args_to_skip=["sandbox_id", "sandbox_state", "working_dir", "state"],
        )
        self.add_tool(
            self.run_proteinmpnn,
            args_to_skip=["sandbox_id", "sandbox_state", "working_dir", "state"],
        )
        self.add_tool(
            self.run_alphafold_multimer,
            args_to_skip=["sandbox_id", "sandbox_state", "working_dir", "state"],
        )

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["working_dir"] = "/"
        state["stage_history"] = []
        state["stage_error_count"] = 0
        state["selected_design_mode"] = ""
        state["selected_sampling_temperature"] = ""
        state["generated_binder_sequence"] = ""
        state["multimer_report"] = {}
        await self._bootstrap_workspace(
            sandbox_id=state["sandbox_id"],
            sandbox_state=state["sandbox_state"],
            working_dir="/workspace",
            task_info=dict(state.get("info", {}) or {}),
        )
        state["working_dir"] = "/workspace"
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name in self.TOOL_NAMES:
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
            updated_args["working_dir"] = state["working_dir"]
            updated_args["state"] = state
            return updated_args
        return super().update_tool_args(tool_name, tool_args, messages, state, **kwargs)

    async def _run_shell(
        self,
        command: str,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
    ) -> str:
        return await self.bash(
            command=command,
            sandbox_id=sandbox_id,
            sandbox_state=sandbox_state,
            working_dir=working_dir,
        )

    async def _bootstrap_workspace(
        self,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
        task_info: dict[str, Any],
    ) -> None:
        encoded_runner = base64.b64encode(RUNNER_SOURCE.encode()).decode()
        bootstrap_script = "\n".join(
            [
                "import base64",
                "import json",
                "import sys",
                "from pathlib import Path",
                "workspace = Path(sys.argv[1])",
                "workspace.mkdir(parents=True, exist_ok=True)",
                "runner_path = workspace / 'sandbox_runner.py'",
                "if not runner_path.exists():",
                "    runner_path.write_bytes(base64.b64decode(sys.argv[2]))",
                "task_path = workspace / 'task.json'",
                "task_path.write_text(sys.argv[3] + '\\n')",
                "target_path = workspace / 'target_sequence.fasta'",
                "target_path.write_text(f'>{sys.argv[5]}\\n{sys.argv[4]}\\n')",
            ]
        )
        command = " ".join(
            [
                "python",
                "-c",
                shlex.quote(bootstrap_script),
                shlex.quote(working_dir),
                shlex.quote(encoded_runner),
                shlex.quote(json.dumps(task_info, sort_keys=True)),
                shlex.quote(task_info["target_sequence"]),
                shlex.quote(task_info["target_id"]),
            ]
        )
        await self._run_shell(command, sandbox_id, sandbox_state, "/")

    def _validate_next_stage(self, state: vf.State, stage_name: str) -> str | None:
        history = list(state.get("stage_history", []))
        if stage_name in EXPECTED_STAGE_SET and stage_name in history:
            return json.dumps(
                {
                    "error": "stage_already_run",
                    "stage": stage_name,
                    "history": history,
                },
                indent=2,
            )
        expected_index = len(history)
        if expected_index >= len(PIPELINE_STAGES):
            return json.dumps(
                {
                    "error": "pipeline_already_completed",
                    "history": history,
                },
                indent=2,
            )
        expected_stage = PIPELINE_STAGES[expected_index]
        if expected_stage != stage_name:
            return json.dumps(
                {
                    "error": "wrong_stage_order",
                    "expected_next_stage": expected_stage,
                    "attempted_stage": stage_name,
                    "history": history,
                },
                indent=2,
            )
        return None

    async def _run_stage(
        self,
        stage_name: str,
        command: str,
        state: vf.State,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
    ) -> dict[str, Any]:
        order_error = self._validate_next_stage(state, stage_name)
        if order_error is not None:
            state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
            return json.loads(order_error)

        output = await self._run_shell(command, sandbox_id, sandbox_state, working_dir)
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
            return {
                "error": "stage_execution_failed",
                "stage": stage_name,
                "raw_output": output,
            }

        if payload.get("error"):
            state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
            return payload

        state["stage_history"] = [*state.get("stage_history", []), stage_name]
        payload["next_expected_stage"] = (
            PIPELINE_STAGES[len(state["stage_history"])]
            if len(state["stage_history"]) < len(PIPELINE_STAGES)
            else "submit_<sequence>"
        )
        return payload

    async def run_alphafold2(
        self,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
        state: vf.State,
    ) -> str:
        """Run AlphaFold2 on the target protein sequence to produce the target structure PDB.

        Returns:
            JSON with the generated target structure summary and the output PDB path.
        """
        command = " ".join(
            [
                "python",
                shlex.quote(f"{working_dir}/sandbox_runner.py"),
                "alphafold2",
                "--task",
                shlex.quote(f"{working_dir}/task.json"),
                "--target-sequence",
                shlex.quote(f"{working_dir}/target_sequence.fasta"),
                "--output-pdb",
                shlex.quote(f"{working_dir}/target_structure.pdb"),
                "--output-json",
                shlex.quote(f"{working_dir}/target_structure_summary.json"),
            ]
        )
        payload = await self._run_stage("AlphaFold2", command, state, sandbox_id, sandbox_state, working_dir)
        return json.dumps(payload, indent=2, sort_keys=True)

    async def run_rfdiffusion(
        self,
        design_mode: str,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
        state: vf.State,
    ) -> str:
        """Run RFdiffusion to generate a binder backbone structure.

        Args:
            design_mode: Backbone generation strategy. Choose one of helix, beta, or balanced.

        Returns:
            JSON with the generated binder backbone summary and output PDB path.
        """
        normalized_mode = (design_mode or "").strip().lower()
        if normalized_mode not in BACKBONE_MODES:
            return json.dumps(
                {
                    "error": "unknown_design_mode",
                    "allowed": list(BACKBONE_MODES),
                },
                indent=2,
            )
        command = " ".join(
            [
                "python",
                shlex.quote(f"{working_dir}/sandbox_runner.py"),
                "rfdiffusion",
                "--task",
                shlex.quote(f"{working_dir}/task.json"),
                "--target-pdb",
                shlex.quote(f"{working_dir}/target_structure.pdb"),
                "--design-mode",
                shlex.quote(normalized_mode),
                "--output-pdb",
                shlex.quote(f"{working_dir}/binder_backbone.pdb"),
                "--output-json",
                shlex.quote(f"{working_dir}/binder_backbone_summary.json"),
            ]
        )
        payload = await self._run_stage("RFdiffusion", command, state, sandbox_id, sandbox_state, working_dir)
        if not payload.get("error"):
            state["selected_design_mode"] = normalized_mode
        return json.dumps(payload, indent=2, sort_keys=True)

    async def run_proteinmpnn(
        self,
        sampling_temperature: str,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
        state: vf.State,
    ) -> str:
        """Run ProteinMPNN on the generated binder backbone to produce a binder sequence.

        Args:
            sampling_temperature: Sequence sampling setting. Choose one of low, medium, or high.

        Returns:
            JSON with the generated binder sequence and the output FASTA path.
        """
        normalized_temperature = (sampling_temperature or "").strip().lower()
        if normalized_temperature not in SAMPLING_TEMPERATURES:
            return json.dumps(
                {
                    "error": "unknown_sampling_temperature",
                    "allowed": list(SAMPLING_TEMPERATURES),
                },
                indent=2,
            )
        selected_design_mode = state.get("selected_design_mode", "")
        if not selected_design_mode:
            return json.dumps(
                {
                    "error": "missing_design_mode",
                    "message": "Run RFdiffusion successfully before ProteinMPNN.",
                },
                indent=2,
            )
        command = " ".join(
            [
                "python",
                shlex.quote(f"{working_dir}/sandbox_runner.py"),
                "proteinmpnn",
                "--task",
                shlex.quote(f"{working_dir}/task.json"),
                "--backbone-pdb",
                shlex.quote(f"{working_dir}/binder_backbone.pdb"),
                "--design-mode",
                shlex.quote(selected_design_mode),
                "--sampling-temperature",
                shlex.quote(normalized_temperature),
                "--output-fasta",
                shlex.quote(f"{working_dir}/binder_sequence.fasta"),
                "--output-json",
                shlex.quote(f"{working_dir}/binder_sequence_summary.json"),
            ]
        )
        payload = await self._run_stage("ProteinMPNN", command, state, sandbox_id, sandbox_state, working_dir)
        if not payload.get("error"):
            state["selected_sampling_temperature"] = normalized_temperature
            state["generated_binder_sequence"] = normalize_sequence(payload.get("binder_sequence", ""))
        return json.dumps(payload, indent=2, sort_keys=True)

    async def run_alphafold_multimer(
        self,
        sandbox_id: str,
        sandbox_state: dict[str, Any],
        working_dir: str,
        state: vf.State,
    ) -> str:
        """Run AlphaFold-Multimer on the target structure and generated binder sequence.

        Returns:
            JSON with structural plausibility, threshold, and pass/fail outcome.
        """
        selected_design_mode = state.get("selected_design_mode", "")
        selected_temperature = state.get("selected_sampling_temperature", "")
        if not selected_design_mode or not selected_temperature:
            return json.dumps(
                {
                    "error": "missing_pipeline_inputs",
                    "message": "Run RFdiffusion and ProteinMPNN successfully before AlphaFold-Multimer.",
                },
                indent=2,
            )
        command = " ".join(
            [
                "python",
                shlex.quote(f"{working_dir}/sandbox_runner.py"),
                "alphafold-multimer",
                "--task",
                shlex.quote(f"{working_dir}/task.json"),
                "--target-pdb",
                shlex.quote(f"{working_dir}/target_structure.pdb"),
                "--binder-fasta",
                shlex.quote(f"{working_dir}/binder_sequence.fasta"),
                "--design-mode",
                shlex.quote(selected_design_mode),
                "--sampling-temperature",
                shlex.quote(selected_temperature),
                "--output-json",
                shlex.quote(f"{working_dir}/multimer_report.json"),
            ]
        )
        payload = await self._run_stage("AlphaFold-Multimer", command, state, sandbox_id, sandbox_state, working_dir)
        if not payload.get("error"):
            state["multimer_report"] = payload
            state["passes_threshold"] = bool(payload.get("passes_threshold", False))
        return json.dumps(payload, indent=2, sort_keys=True)


def load_environment(
    num_train_examples: int = 96,
    num_eval_examples: int = 24,
    train_seed: int = 11,
    eval_seed: int = 23,
    max_turns: int = 6,
    docker_image: str = "python:3.11-slim",
) -> vf.Environment:
    train_dataset = Dataset.from_list(build_task_rows(num_train_examples, train_seed, split="train"))
    eval_dataset = Dataset.from_list(build_task_rows(num_eval_examples, eval_seed, split="eval"))

    parser = vf.XMLParser(["sequence"], answer_field="sequence")
    rubric = vf.Rubric(
        funcs=[structural_pass_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.1],
        parser=parser,
    )
    rubric.add_metric(submitted_sequence_matches_candidate)
    rubric.add_metric(pipeline_completed_metric)
    rubric.add_metric(structural_plausibility_metric)
    rubric.add_metric(threshold_metric)
    rubric.add_metric(pass_margin_metric)
    rubric.add_metric(passes_threshold_metric)
    rubric.add_metric(used_optimal_backbone_metric)
    rubric.add_metric(used_optimal_sampling_metric)
    rubric.add_metric(stage_error_metric)

    return ProteinBinderPipelineEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        docker_image=docker_image,
    )
