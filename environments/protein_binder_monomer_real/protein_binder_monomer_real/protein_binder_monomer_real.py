from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from pathlib import Path
from typing import Any

import verifiers as vf

from .tasks import SYSTEM_PROMPT, build_datasets


class CommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        printable = " ".join(shlex.quote(part) for part in command)
        super().__init__(f"Command failed ({returncode}): {printable}")


def _parse_submission(completion: vf.Messages, parser: vf.XMLParser) -> str:
    for msg in reversed(parser.get_assistant_messages(completion)):
        content = parser._content_to_text(
            msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
        )
        parsed = parser.parse(content, last=True)
        sequence = "".join(char for char in (getattr(parsed, "sequence", None) or "").upper() if char.isalpha())
        if sequence:
            return sequence
    return ""


async def submitted_sequence_matches_passing_candidate(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    submitted = _parse_submission(completion, parser)
    if not submitted:
        return 0.0
    return 1.0 if submitted in set(state.get("passing_candidate_sequences", [])) else 0.0


async def submitted_sequence_matches_best_passing_candidate(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    submitted = _parse_submission(completion, parser)
    best = str(state.get("best_passing_sequence", ""))
    return 1.0 if submitted and best and submitted == best else 0.0


async def monomer_success_reward(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    if state.get("stage_history", []) != [
        "target_monomer",
        "rfdiffusion",
        "proteinmpnn",
        "binder_monomer",
        "summarize_candidates",
    ]:
        return 0.0
    return await submitted_sequence_matches_passing_candidate(completion, state, parser)


async def pipeline_completed_metric(completion: vf.Messages, state: vf.State) -> float:
    expected = [
        "target_monomer",
        "rfdiffusion",
        "proteinmpnn",
        "binder_monomer",
        "summarize_candidates",
    ]
    return 1.0 if state.get("stage_history", []) == expected else 0.0


async def passing_candidate_available_metric(completion: vf.Messages, state: vf.State) -> float:
    return 1.0 if state.get("num_passing_candidates", 0) > 0 else 0.0


async def num_passing_candidates_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("num_passing_candidates", 0))


async def best_passing_score_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("best_passing_score", 0.0))


async def target_mean_plddt_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("target_mean_plddt", 0.0))


async def stage_error_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(state.get("stage_error_count", 0))


class ProteinBinderMonomerRealEnv(vf.StatefulToolEnv):
    PIPELINE_STAGES = [
        "target_monomer",
        "rfdiffusion",
        "proteinmpnn",
        "binder_monomer",
        "summarize_candidates",
    ]

    def __init__(
        self,
        *,
        remote_host: str,
        remote_support_dir: str,
        remote_run_root: str,
        local_support_dir: str,
        keep_remote_artifacts: bool = False,
        sync_support_on_start: bool = True,
        **kwargs,
    ):
        super().__init__(tools=[], **kwargs)
        self.remote_host = remote_host
        self.remote_support_dir = remote_support_dir
        self.remote_run_root = remote_run_root
        self.local_support_dir = Path(local_support_dir).resolve()
        self.keep_remote_artifacts = keep_remote_artifacts
        self.sync_support_on_start = sync_support_on_start
        self._support_sync_lock = asyncio.Lock()
        self._support_synced = False

        self.add_tool(self.run_target_monomer, args_to_skip=["state"])
        self.add_tool(self.run_rfdiffusion, args_to_skip=["state"])
        self.add_tool(self.run_proteinmpnn, args_to_skip=["state"])
        self.add_tool(self.run_binder_monomer, args_to_skip=["state"])
        self.add_tool(self.summarize_candidates, args_to_skip=["state"])

    async def _run_local_command(self, command: list[str]) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        out_text = stdout.decode()
        err_text = stderr.decode()
        if process.returncode != 0:
            raise CommandError(command, process.returncode, out_text, err_text)
        return out_text

    async def _run_remote_command(self, remote_command: str) -> str:
        return await self._run_local_command([
            "ssh",
            "-o",
            "BatchMode=yes",
            self.remote_host,
            f"bash -c {shlex.quote(remote_command)}",
        ])

    async def _read_remote_json(self, remote_path: str) -> Any:
        output = await self._run_remote_command(f"cat {shlex.quote(remote_path)}")
        return json.loads(output)

    async def _tail_remote_file(self, remote_path: str, lines: int = 80) -> str:
        try:
            return await self._run_remote_command(f"tail -n {lines} {shlex.quote(remote_path)}")
        except CommandError as exc:
            return exc.stdout or exc.stderr

    async def _ensure_remote_support_synced(self) -> None:
        if not self.sync_support_on_start or self._support_synced:
            return
        async with self._support_sync_lock:
            if self._support_synced:
                return
            await self._run_remote_command(f"mkdir -p {shlex.quote(self.remote_support_dir)}")
            await self._run_local_command(
                [
                    "rsync",
                    "-az",
                    "--delete",
                    f"{self.local_support_dir}/",
                    f"{self.remote_host}:{self.remote_support_dir}/",
                ]
            )
            self._support_synced = True

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        await self._ensure_remote_support_synced()
        rollout_id = uuid.uuid4().hex[:12]
        task_info = dict(state.get("info", {}) or {})
        remote_run_dir = f"{self.remote_run_root}/{rollout_id}"
        state["remote_run_dir"] = remote_run_dir
        state["remote_log_dir"] = f"{remote_run_dir}/state"
        state["stage_history"] = []
        state["stage_error_count"] = 0
        state["best_passing_sequence"] = ""
        state["passing_candidate_sequences"] = []
        state["num_passing_candidates"] = 0
        state["best_passing_score"] = 0.0
        state["target_mean_plddt"] = 0.0

        gate = task_info["quality_gate"]
        init_parts = [
            "python3",
            shlex.quote(f"{self.remote_support_dir}/run_monomer_pipeline.py"),
            "init-run",
            "--run-dir",
            shlex.quote(remote_run_dir),
            "--target-pdb",
            shlex.quote(task_info["remote_target_pdb"]),
            "--target-chain",
            shlex.quote(task_info["target_chain"]),
            "--hotspots",
            shlex.quote(",".join(task_info["hotspots"])),
            "--binder-length-min",
            str(task_info["binder_length_min"]),
            "--binder-length-max",
            str(task_info["binder_length_max"]),
            "--num-designs",
            str(task_info["num_designs"]),
            "--num-seqs-per-backbone",
            str(task_info["num_seqs_per_backbone"]),
            "--candidate-batch-size",
            str(task_info["candidate_batch_size"]),
            "--max-concurrent-binder-batches",
            "1",
            "--min-target-mean-plddt",
            str(gate["min_target_mean_plddt"]),
            "--min-binder-mean-plddt",
            str(gate["min_binder_mean_plddt"]),
            "--max-binder-distance-rmse",
            str(gate["max_binder_distance_rmse"]),
            "--min-hotspot-fraction",
            str(gate["min_hotspot_fraction"]),
            "--min-interface-residue-contacts",
            str(gate["min_interface_residue_contacts"]),
            "--score-threshold",
            str(gate["score_threshold"]),
            "--overwrite",
            ">",
            shlex.quote(f"{remote_run_dir}/state/init-run.log"),
            "2>&1",
        ]
        try:
            await self._run_remote_command(f"mkdir -p {shlex.quote(remote_run_dir)}/state && {' '.join(init_parts)}")
        except CommandError:
            init_log = await self._tail_remote_file(f"{remote_run_dir}/state/init-run.log")
            raise RuntimeError(f"Failed to initialize remote run.\n{init_log}")
        return await super().setup_state(state, **kwargs)

    @vf.cleanup
    async def cleanup_remote_run(self, state: vf.State):
        if self.keep_remote_artifacts:
            return
        remote_run_dir = state.get("remote_run_dir")
        if remote_run_dir:
            try:
                await self._run_remote_command(f"rm -rf {shlex.quote(remote_run_dir)}")
            except CommandError:
                pass

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict[str, Any]:
        tool_args["state"] = state
        return tool_args

    def _stage_order_error(self, state: vf.State, stage_name: str) -> str | None:
        history = list(state.get("stage_history", []))
        if stage_name in history:
            return json.dumps(
                {"error": "stage_already_run", "stage": stage_name, "history": history},
                indent=2,
            )
        expected_index = len(history)
        if expected_index >= len(self.PIPELINE_STAGES):
            return json.dumps({"error": "pipeline_already_completed", "history": history}, indent=2)
        expected_stage = self.PIPELINE_STAGES[expected_index]
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

    async def _execute_stage(
        self,
        *,
        state: vf.State,
        stage_name: str,
        subcommand: str,
        result_loader: str,
    ) -> dict[str, Any]:
        order_error = self._stage_order_error(state, stage_name)
        if order_error:
            state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
            return json.loads(order_error)

        remote_run_dir = str(state["remote_run_dir"])
        log_path = f"{remote_run_dir}/state/{subcommand}.log"
        command = " ".join(
            [
                "python3",
                shlex.quote(f"{self.remote_support_dir}/run_monomer_pipeline.py"),
                subcommand,
                "--run-dir",
                shlex.quote(remote_run_dir),
                ">",
                shlex.quote(log_path),
                "2>&1",
            ]
        )
        try:
            await self._run_remote_command(command)
        except CommandError:
            state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
            return {
                "error": "stage_execution_failed",
                "stage": stage_name,
                "log_tail": await self._tail_remote_file(log_path),
            }

        payload = await self._read_remote_json(f"{remote_run_dir}/{result_loader}")
        state["stage_history"] = [*state.get("stage_history", []), stage_name]
        if stage_name == "target_monomer":
            state["target_mean_plddt"] = float(payload["target_mean_plddt"] or 0.0)
        if stage_name == "summarize_candidates":
            best_passing = payload.get("best_passing_candidate") or {}
            state["best_passing_sequence"] = best_passing.get("sequence", "")
            state["best_passing_score"] = float(best_passing.get("monomer_plausibility_score", 0.0) or 0.0)
            state["num_passing_candidates"] = int(payload.get("num_passing_candidates", 0) or 0)
            state["passing_candidate_sequences"] = [
                candidate["sequence"]
                for candidate in payload.get("ranked_candidates", [])
                if candidate.get("passes_quality_gate")
            ]
        return payload

    async def run_target_monomer(self, state: vf.State) -> str:
        """Run the target ColabFold monomer stage on the remote GPU host.

        Returns:
            JSON with target monomer quality metrics and output paths.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="target_monomer",
            subcommand="target-monomer",
            result_loader="state/target_summary.json",
        )
        public_payload = {
            "target_pdb": payload.get("target_pdb"),
            "target_sequence_length": payload.get("target_sequence_length"),
            "target_mean_plddt": payload.get("target_mean_plddt"),
            "target_ptm": payload.get("target_ptm"),
        }
        return json.dumps(public_payload if "error" not in payload else payload, indent=2, sort_keys=True)

    async def run_rfdiffusion(self, state: vf.State) -> str:
        """Run RFdiffusion with the task's preconfigured search settings.

        Returns:
            JSON with the generated backbones and their interface proxy metrics.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="rfdiffusion",
            subcommand="rfdiffusion",
            result_loader="state/backbones.json",
        )
        if isinstance(payload, list):
            summary = {
                "num_backbones": len(payload),
                "backbones": payload,
            }
            return json.dumps(summary, indent=2, sort_keys=True)
        return json.dumps(payload, indent=2, sort_keys=True)

    async def run_proteinmpnn(self, state: vf.State) -> str:
        """Run ProteinMPNN on the generated backbones.

        Returns:
            JSON with the number of designed candidate sequences.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="proteinmpnn",
            subcommand="proteinmpnn",
            result_loader="state/candidates.json",
        )
        if isinstance(payload, list):
            return json.dumps({"num_candidates": len(payload)}, indent=2, sort_keys=True)
        return json.dumps(payload, indent=2, sort_keys=True)

    async def run_binder_monomer(self, state: vf.State) -> str:
        """Run monomer scoring for all binder candidates on the remote GPU host.

        Returns:
            JSON with the number of scoring batches that were executed.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="binder_monomer",
            subcommand="binder-monomer",
            result_loader="state/binder_batches.json",
        )
        if isinstance(payload, list):
            return json.dumps({"num_batches": len(payload), "batches": payload}, indent=2, sort_keys=True)
        return json.dumps(payload, indent=2, sort_keys=True)

    async def summarize_candidates(self, state: vf.State) -> str:
        """Summarize the remote run and surface the best passing candidates.

        Returns:
            JSON containing the best candidate, the best passing candidate, and the top ranked candidates.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="summarize_candidates",
            subcommand="summarize",
            result_loader="summary/run_summary.json",
        )
        if payload.get("error"):
            return json.dumps(payload, indent=2, sort_keys=True)
        summary = {
            "num_candidates": payload.get("num_candidates"),
            "num_passing_candidates": payload.get("num_passing_candidates"),
            "best_candidate": payload.get("best_candidate"),
            "best_passing_candidate": payload.get("best_passing_candidate"),
            "top_ranked_candidates": payload.get("ranked_candidates", [])[:8],
        }
        return json.dumps(summary, indent=2, sort_keys=True)


def load_environment(
    num_train_examples: int = 4,
    num_eval_examples: int = 1,
    max_turns: int = 8,
    remote_host: str = "ubuntu@154.54.100.216",
    remote_support_dir: str = "/home/ubuntu/pi-workspace/protein-binder/experiments/real_monomer_harness",
    remote_run_root: str = "/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real",
    keep_remote_artifacts: bool = False,
    sync_support_on_start: bool = True,
) -> vf.Environment:
    support_dir = Path(__file__).resolve().parent / "support"
    train_dataset, eval_dataset = build_datasets(num_train_examples, num_eval_examples)

    parser = vf.XMLParser(["sequence"], answer_field="sequence")
    rubric = vf.Rubric(
        funcs=[monomer_success_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.1],
        parser=parser,
    )
    rubric.add_metric(submitted_sequence_matches_passing_candidate)
    rubric.add_metric(submitted_sequence_matches_best_passing_candidate)
    rubric.add_metric(pipeline_completed_metric)
    rubric.add_metric(passing_candidate_available_metric)
    rubric.add_metric(num_passing_candidates_metric)
    rubric.add_metric(best_passing_score_metric)
    rubric.add_metric(target_mean_plddt_metric)
    rubric.add_metric(stage_error_metric)

    return ProteinBinderMonomerRealEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
        max_turns=max_turns,
        remote_host=remote_host,
        remote_support_dir=remote_support_dir,
        remote_run_root=remote_run_root,
        local_support_dir=str(support_dir),
        keep_remote_artifacts=keep_remote_artifacts,
        sync_support_on_start=sync_support_on_start,
    )
