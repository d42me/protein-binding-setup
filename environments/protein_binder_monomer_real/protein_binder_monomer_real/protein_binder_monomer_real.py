from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import random
import hashlib
import re
import shlex
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import paramiko
import verifiers as vf

from .tasks import SYSTEM_PROMPT, build_datasets


LOGGER = logging.getLogger("protein_binder_monomer_real")


def _remote_api_result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        summary: dict[str, Any] = {
            "type": "dict",
            "keys": sorted(result.keys())[:24],
        }
        if result.get("error"):
            summary["error"] = result.get("error")
        return summary
    if isinstance(result, list):
        return {"type": "list", "length": len(result)}
    if result is None:
        return {"type": "none"}
    return {"type": type(result).__name__}


class CommandError(RuntimeError):
    def __init__(self, command: list[str], returncode: int, stdout: str, stderr: str):
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        printable = " ".join(shlex.quote(part) for part in command)
        super().__init__(f"Command failed ({returncode}): {printable}")


PIPELINE_STAGES = [
    "target_monomer",
    "rfdiffusion",
    "proteinmpnn",
    "binder_monomer",
    "summarize_candidates",
]
FREE_TOOL_CALLS = 30
TOOL_OVERUSE_PENALTY_SCALE = 400.0


def _parse_submission_candidate_id(completion: vf.Messages, parser: vf.XMLParser) -> str:
    for msg in reversed(parser.get_assistant_messages(completion)):
        content = parser._content_to_text(
            msg.get("content", "") if isinstance(msg, dict) else (msg.content or "")
        )

        tag_matches = re.findall(
            r"<candidate_id>\s*([A-Za-z0-9_.:-]+)\s*</candidate_id>",
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if tag_matches:
            return tag_matches[-1].strip()

        parsed = parser.parse(content, last=True)
        candidate_id = str(getattr(parsed, "candidate_id", None) or "").strip()
        if candidate_id:
            return candidate_id
    return ""


def _candidate_lookup(state: vf.State) -> dict[str, dict[str, Any]]:
    return dict(state.get("candidate_lookup", {}) or {})


def _submitted_candidate(state: vf.State, completion: vf.Messages, parser: vf.XMLParser) -> tuple[str, dict[str, Any] | None]:
    candidate_id = _parse_submission_candidate_id(completion, parser)
    if not candidate_id:
        return "", None
    return candidate_id, _candidate_lookup(state).get(candidate_id)


def _pipeline_completed(state: vf.State) -> bool:
    successful_stage_counts = dict(state.get("successful_stage_counts", {}) or {})
    return all(int(successful_stage_counts.get(stage, 0) or 0) > 0 for stage in PIPELINE_STAGES)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _quality_gate(state: vf.State) -> dict[str, Any]:
    return dict((state.get("info", {}) or {}).get("quality_gate", {}) or {})


def _linear_score_above(value: float | None, floor: float, ceiling: float) -> float:
    if value is None:
        return 0.0
    if ceiling <= floor:
        return 1.0 if float(value) >= ceiling else 0.0
    return _clamp01((float(value) - floor) / (ceiling - floor))


def _linear_score_below(value: float | None, best: float, worst: float) -> float:
    if value is None:
        return 0.0
    if worst <= best:
        return 1.0 if float(value) <= best else 0.0
    return _clamp01((worst - float(value)) / (worst - best))


def _candidate_science_components(candidate: dict[str, Any], state: vf.State) -> dict[str, float]:
    gate = _quality_gate(state)
    max_rmse = float(gate.get("max_binder_distance_rmse", 1.5) or 1.5)
    min_binder_plddt = float(gate.get("min_binder_mean_plddt", 80.0) or 80.0)
    min_contacts = float(gate.get("min_interface_residue_contacts", 10) or 10)
    min_hotspot = float(gate.get("min_hotspot_fraction", 0.33) or 0.33)
    score_threshold = float(gate.get("score_threshold", 0.72) or 0.72)

    return {
        "plausibility": _linear_score_above(
            candidate.get("monomer_plausibility_score"),
            score_threshold,
            max(0.95, score_threshold + 0.23),
        ),
        "binder_confidence": _linear_score_above(
            candidate.get("binder_mean_plddt"),
            min_binder_plddt,
            97.0,
        ),
        "geometry": _linear_score_below(
            candidate.get("binder_distance_rmse"),
            0.2,
            max_rmse,
        ),
        "hotspot": _linear_score_above(
            candidate.get("hotspot_fraction"),
            min_hotspot,
            1.0,
        ),
        "interface": _linear_score_above(
            candidate.get("interface_residue_contacts"),
            min_contacts,
            max(30.0, min_contacts + 20.0),
        ),
    }



def _weighted_geometric_mean(weighted_components: list[tuple[float, float]]) -> float:
    total_weight = sum(weight for _, weight in weighted_components)
    if total_weight <= 0:
        return 0.0
    log_total = 0.0
    for value, weight in weighted_components:
        log_total += weight * math.log(max(1e-6, value))
    return math.exp(log_total / total_weight)



def _candidate_science_reward_value(candidate: dict[str, Any], state: vf.State) -> float:
    components = _candidate_science_components(candidate, state)
    strict_quality = _weighted_geometric_mean(
        [
            (components["plausibility"], 0.30),
            (components["geometry"], 0.25),
            (components["binder_confidence"], 0.20),
            (components["hotspot"], 0.15),
            (components["interface"], 0.10),
        ]
    )
    pass_factor = 1.0 if candidate.get("passes_quality_gate") else 0.6
    return round(pass_factor * strict_quality, 3)


def _sort_candidates_by_science_reward(candidates: list[dict[str, Any]], state: vf.State) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            _candidate_science_reward_value(candidate, state),
            float(candidate.get("monomer_plausibility_score", 0.0) or 0.0),
            -float(candidate.get("binder_distance_rmse", 999.0) or 999.0),
        ),
        reverse=True,
    )


def _total_stage_calls(state: vf.State) -> int:
    call_counts = dict(state.get("stage_call_counts", {}) or {})
    return sum(int(value or 0) for value in call_counts.values())


def _tool_overuse_penalty_value(state: vf.State, free_calls: int = FREE_TOOL_CALLS) -> float:
    total_calls = _total_stage_calls(state)
    if total_calls <= free_calls:
        return 0.0
    overflow = total_calls - free_calls
    return round(min(1.0, (overflow * overflow) / TOOL_OVERUSE_PENALTY_SCALE), 3)


async def candidate_selection_reward(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    if not _pipeline_completed(state):
        return 0.0

    _, candidate = _submitted_candidate(state, completion, parser)
    if not candidate:
        return 0.0

    return _candidate_science_reward_value(candidate, state)


async def tool_overuse_penalty_reward(completion: vf.Messages, state: vf.State) -> float:
    return -_tool_overuse_penalty_value(state)


async def submitted_candidate_known_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return 1.0 if candidate else 0.0


async def submitted_candidate_passes_quality_gate(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return 1.0 if candidate and candidate.get("passes_quality_gate") else 0.0


async def submitted_candidate_is_best_candidate(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    candidate_id, candidate = _submitted_candidate(state, completion, parser)
    best_candidate_id = str(state.get("best_candidate_id", ""))
    return 1.0 if candidate and candidate_id and candidate_id == best_candidate_id else 0.0


async def submitted_candidate_rank_percentile_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return float(candidate.get("rank_percentile", 0.0) or 0.0) if candidate else 0.0


async def submitted_candidate_quality_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return float(candidate.get("monomer_plausibility_score", 0.0) or 0.0) if candidate else 0.0


async def submitted_candidate_science_reward_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_reward_value(candidate, state) if candidate else 0.0


async def submitted_candidate_plausibility_component_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_components(candidate, state)["plausibility"] if candidate else 0.0


async def submitted_candidate_geometry_component_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_components(candidate, state)["geometry"] if candidate else 0.0


async def submitted_candidate_binder_confidence_component_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_components(candidate, state)["binder_confidence"] if candidate else 0.0


async def submitted_candidate_hotspot_component_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_components(candidate, state)["hotspot"] if candidate else 0.0


async def submitted_candidate_interface_component_metric(
    completion: vf.Messages,
    state: vf.State,
    parser: vf.XMLParser,
) -> float:
    _, candidate = _submitted_candidate(state, completion, parser)
    return _candidate_science_components(candidate, state)["interface"] if candidate else 0.0


async def pipeline_completed_metric(completion: vf.Messages, state: vf.State) -> float:
    return 1.0 if _pipeline_completed(state) else 0.0


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


async def total_stage_calls_metric(completion: vf.Messages, state: vf.State) -> float:
    return float(_total_stage_calls(state))


async def tool_overuse_penalty_metric(completion: vf.Messages, state: vf.State) -> float:
    return _tool_overuse_penalty_value(state)


def _rank_percentile(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return round(1.0 - (index / (total - 1)), 3)


def _candidate_state_record(candidate: dict[str, Any], state: vf.State, *, rank_index: int, total: int) -> dict[str, Any]:
    science_components = _candidate_science_components(candidate, state)
    return {
        "candidate_id": candidate.get("candidate_id"),
        "sequence": candidate.get("sequence", ""),
        "passes_quality_gate": bool(candidate.get("passes_quality_gate")),
        "monomer_plausibility_score": float(candidate.get("monomer_plausibility_score", 0.0) or 0.0),
        "binder_mean_plddt": float(candidate.get("binder_mean_plddt", 0.0) or 0.0),
        "binder_distance_rmse": candidate.get("binder_distance_rmse"),
        "hotspot_fraction": float(candidate.get("hotspot_fraction", 0.0) or 0.0),
        "interface_residue_contacts": int(candidate.get("interface_residue_contacts", 0) or 0),
        "science_reward": _candidate_science_reward_value(candidate, state),
        "science_components": science_components,
        "rank": rank_index + 1,
        "rank_percentile": _rank_percentile(rank_index, total),
    }


def _public_candidate_view(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "binder_length": candidate.get("binder_length"),
        "binder_mean_plddt": candidate.get("binder_mean_plddt"),
        "binder_ptm": candidate.get("binder_ptm"),
        "binder_distance_rmse": candidate.get("binder_distance_rmse"),
        "hotspot_fraction": candidate.get("hotspot_fraction"),
        "interface_residue_contacts": candidate.get("interface_residue_contacts"),
        "passes_quality_gate": candidate.get("passes_quality_gate"),
        "quality_gate_failures": candidate.get("quality_gate_failures", []),
    }


def _stage_state_key(stage_name: str) -> str:
    return {
        "target_monomer": "target_summary",
        "rfdiffusion": "backbones",
        "proteinmpnn": "candidates",
        "binder_monomer": "binder_candidate_results",
        "summarize_candidates": "run_summary",
    }[stage_name]


def _downstream_stages(stage_name: str) -> list[str]:
    stage_index = PIPELINE_STAGES.index(stage_name)
    return PIPELINE_STAGES[stage_index + 1 :]


def _clear_stage_state(state: vf.State, stage_name: str) -> None:
    state[_stage_state_key(stage_name)] = {} if stage_name in {"target_monomer", "summarize_candidates"} else []
    successful_stage_counts = dict(state.get("successful_stage_counts", {}) or {})
    successful_stage_counts[stage_name] = 0
    state["successful_stage_counts"] = successful_stage_counts
    if stage_name == "summarize_candidates":
        state["candidate_lookup"] = {}
        state["best_candidate_id"] = ""
        state["best_passing_candidate_id"] = ""
        state["num_passing_candidates"] = 0
        state["best_passing_score"] = 0.0


def _invalidate_downstream_stage_state(state: vf.State, stage_name: str) -> None:
    for downstream_stage in _downstream_stages(stage_name):
        _clear_stage_state(state, downstream_stage)


def _stage_tool_input(state: vf.State, stage_name: str) -> dict[str, Any]:
    task_info = dict(state.get("info", {}) or {})
    common = {
        "remote_run_dir": state.get("remote_run_dir"),
        "transport": (state.get("environment_resolution", {}) or {}).get("transport"),
    }
    if stage_name == "target_monomer":
        return {
            **common,
            "target": {
                "target_id": task_info.get("target_id"),
                "target_sequence": task_info.get("target_sequence"),
                "target_chain": task_info.get("target_chain"),
                "hotspots": task_info.get("hotspots"),
            },
            "search_budget": {
                "binder_length_min": task_info.get("binder_length_min"),
                "binder_length_max": task_info.get("binder_length_max"),
                "num_designs": task_info.get("num_designs"),
                "num_seqs_per_backbone": task_info.get("num_seqs_per_backbone"),
                "candidate_batch_size": task_info.get("candidate_batch_size"),
            },
            "quality_gate": task_info.get("quality_gate"),
        }
    if stage_name == "rfdiffusion":
        return {
            **common,
            "target_summary": _budgeted_stage_output("target_monomer", state.get("target_summary")),
            "design_request": {
                "binder_length_min": task_info.get("binder_length_min"),
                "binder_length_max": task_info.get("binder_length_max"),
                "num_designs": task_info.get("num_designs"),
                "hotspots": task_info.get("hotspots"),
            },
        }
    if stage_name == "proteinmpnn":
        return {
            **common,
            "backbones": _budgeted_stage_output("rfdiffusion", state.get("backbones", [])),
            "sequencing_request": {
                "num_seqs_per_backbone": task_info.get("num_seqs_per_backbone"),
            },
        }
    if stage_name == "binder_monomer":
        return {
            **common,
            "candidates": _budgeted_stage_output("proteinmpnn", state.get("candidates", [])),
            "batching": {
                "candidate_batch_size": task_info.get("candidate_batch_size"),
            },
        }
    return {
        **common,
        "target_summary": _budgeted_stage_output("target_monomer", state.get("target_summary")),
        "backbones": _budgeted_stage_output("rfdiffusion", state.get("backbones", [])),
        "candidates": _budgeted_stage_output("proteinmpnn", state.get("candidates", [])),
        "binder_candidate_results": _budgeted_stage_output("binder_monomer", state.get("binder_candidate_results", [])),
        "quality_gate": task_info.get("quality_gate"),
    }


def _budgeted_candidate_view(candidate: dict[str, Any]) -> dict[str, Any]:
    budgeted = {
        "candidate_id": candidate.get("candidate_id"),
        "backbone_name": candidate.get("backbone_name"),
        "sample_index": candidate.get("sample_index"),
        "binder_length": candidate.get("binder_length"),
        "binder_mean_plddt": candidate.get("binder_mean_plddt"),
        "binder_ptm": candidate.get("binder_ptm"),
        "binder_distance_rmse": candidate.get("binder_distance_rmse"),
        "monomer_plausibility_score": candidate.get("monomer_plausibility_score"),
        "mpnn_score": candidate.get("mpnn_score"),
        "mpnn_global_score": candidate.get("mpnn_global_score"),
        "seq_recovery": candidate.get("seq_recovery"),
        "hotspot_fraction": candidate.get("hotspot_fraction"),
        "interface_residue_contacts": candidate.get("interface_residue_contacts"),
        "hotspot_contacts": candidate.get("hotspot_contacts"),
        "passes_quality_gate": candidate.get("passes_quality_gate"),
        "quality_gate_failures": candidate.get("quality_gate_failures"),
    }
    sequence = candidate.get("sequence")
    if isinstance(sequence, str) and sequence:
        budgeted["sequence_length"] = len(sequence)
    return {key: value for key, value in budgeted.items() if value is not None}


def _budgeted_stage_output(stage_name: str, payload: Any) -> Any:
    if isinstance(payload, dict):
        if payload.get("error"):
            return payload
        if stage_name == "target_monomer":
            return {
                "target_pdb": payload.get("target_pdb"),
                "target_sequence_length": payload.get("target_sequence_length"),
                "target_mean_plddt": payload.get("target_mean_plddt"),
                "target_ptm": payload.get("target_ptm"),
            }
        if stage_name == "summarize_candidates":
            ranked_candidates = [_budgeted_candidate_view(candidate) for candidate in payload.get("ranked_candidates", [])]
            return {
                "quality_gate": payload.get("quality_gate"),
                "target": {
                    "target_job_name": (payload.get("target") or {}).get("target_job_name"),
                    "target_mean_plddt": (payload.get("target") or {}).get("target_mean_plddt"),
                    "target_ptm": (payload.get("target") or {}).get("target_ptm"),
                    "target_sequence_length": (payload.get("target") or {}).get("target_sequence_length"),
                },
                "backbones": _budgeted_stage_output("rfdiffusion", payload.get("backbones", [])),
                "ranked_candidates_top_k": ranked_candidates[:16],
                "ranked_candidate_ids": [candidate.get("candidate_id") for candidate in ranked_candidates],
                "best_candidate": _budgeted_candidate_view(payload.get("best_candidate") or {}),
                "best_passing_candidate": _budgeted_candidate_view(payload.get("best_passing_candidate") or {}),
                "num_candidates": payload.get("num_candidates"),
                "num_passing_candidates": payload.get("num_passing_candidates"),
                "passing_candidate_ids": payload.get("passing_candidate_ids"),
            }
        return payload
    if isinstance(payload, list):
        if stage_name == "proteinmpnn":
            candidate_views = [_budgeted_candidate_view(candidate) for candidate in payload]
            top_candidates = sorted(
                candidate_views,
                key=lambda candidate: (
                    float(candidate.get("mpnn_score") or -1e9),
                    float(candidate.get("seq_recovery") or -1e9),
                ),
                reverse=True,
            )
            return {
                "num_candidates": len(candidate_views),
                "candidate_ids": [candidate.get("candidate_id") for candidate in candidate_views],
                "top_candidates_by_mpnn_score": top_candidates[:16],
            }
        if stage_name == "binder_monomer":
            candidate_views = [_budgeted_candidate_view(candidate) for candidate in payload]
            top_candidates = sorted(
                candidate_views,
                key=lambda candidate: (
                    bool(candidate.get("passes_quality_gate")),
                    float(candidate.get("monomer_plausibility_score") or -1e9),
                    float(candidate.get("binder_mean_plddt") or -1e9),
                ),
                reverse=True,
            )
            return {
                "num_scored_candidates": len(candidate_views),
                "candidate_ids": [candidate.get("candidate_id") for candidate in candidate_views],
                "num_quality_gate_passes": sum(1 for candidate in candidate_views if candidate.get("passes_quality_gate")),
                "top_candidates_by_plausibility": top_candidates[:16],
            }
        if stage_name == "rfdiffusion":
            return [
                {
                    "backbone_name": item.get("backbone_name"),
                    "binder_chain": item.get("binder_chain"),
                    "binder_length": item.get("binder_length"),
                    "interface_residue_contacts": item.get("interface_residue_contacts"),
                    "hotspot_contacts": item.get("hotspot_contacts"),
                    "hotspot_fraction": item.get("hotspot_fraction"),
                }
                for item in payload
            ]
    return payload


def _stage_response_payload(
    state: vf.State,
    *,
    stage_name: str,
    payload: Any,
) -> str:
    total_calls = _total_stage_calls(state)
    free_calls = FREE_TOOL_CALLS
    response = {
        "stage": stage_name,
        "stage_complete": stage_name if not (isinstance(payload, dict) and payload.get("error")) else None,
        "stage_history": state.get("stage_history", []),
        "stage_error_count": state.get("stage_error_count", 0),
        "tool_budget": {
            "total_tool_calls": total_calls,
            "free_tool_calls_before_penalty": free_calls,
            "remaining_free_tool_calls": max(0, free_calls - total_calls),
            "current_tool_overuse_penalty": _tool_overuse_penalty_value(state, free_calls=free_calls),
        },
        "tool_input": _stage_tool_input(state, stage_name),
        "tool_output": _budgeted_stage_output(stage_name, payload),
    }
    return json.dumps(response, indent=2, sort_keys=True)


class ProteinBinderMonomerRealEnv(vf.StatefulToolEnv):
    PIPELINE_STAGES = PIPELINE_STAGES

    def __init__(
        self,
        *,
        remote_host: str,
        remote_support_dir: str,
        remote_run_root: str,
        local_support_dir: str,
        keep_remote_artifacts: bool = False,
        sync_support_on_start: bool = True,
        remote_api_base_url: str | None = None,
        remote_api_token_env_var: str = "PROTEIN_BINDER_API_TOKEN",
        remote_api_timeout_seconds: int = 43200,
        **kwargs,
    ):
        super().__init__(tools=[], **kwargs)
        self.remote_host = remote_host
        self.remote_support_dir = remote_support_dir
        self.remote_run_root = remote_run_root
        self.local_support_dir = Path(local_support_dir).resolve()
        self.keep_remote_artifacts = keep_remote_artifacts
        self.sync_support_on_start = sync_support_on_start
        self.remote_api_base_url = (remote_api_base_url or "").rstrip("/") or None
        self.remote_api_token_env_var = remote_api_token_env_var
        self.remote_api_timeout_seconds = remote_api_timeout_seconds
        self._support_sync_lock = asyncio.Lock()
        self._support_synced = False
        self._remote_command_lock = asyncio.Lock()
        self._remote_lock_dir = "/tmp/protein-binder-monomer-real-locks"
        self._remote_host_lock_path = f"{self._remote_lock_dir}/host.lock"
        self._ssh_key_dir: Path | None = None
        self._ssh_key_path: Path | None = None
        self._paramiko_client: paramiko.SSHClient | None = None

        self.add_tool(self.run_target_monomer, args_to_skip=["state"])
        self.add_tool(self.run_rfdiffusion, args_to_skip=["state"])
        self.add_tool(self.run_proteinmpnn, args_to_skip=["state"])
        self.add_tool(self.run_binder_monomer, args_to_skip=["state"])
        self.add_tool(self.summarize_candidates, args_to_skip=["state"])

    def _api_enabled(self) -> bool:
        return self.remote_api_base_url is not None

    def _api_headers(self) -> dict[str, str]:
        token = os.environ.get(self.remote_api_token_env_var, "").strip()
        if not token:
            raise RuntimeError(
                f"Remote API mode requires environment variable {self.remote_api_token_env_var}"
            )
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def _remote_api_request_sync(self, method: str, path: str, payload: Any | None = None) -> Any:
        if not self.remote_api_base_url:
            raise RuntimeError("Remote API base URL is not configured")
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            url=f"{self.remote_api_base_url}{path}",
            data=data,
            headers=self._api_headers(),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.remote_api_timeout_seconds) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as exc:
            body = exc.read().decode()
            raise RuntimeError(f"Remote API request failed ({exc.code}) for {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote API request failed for {path}: {exc}") from exc
        return json.loads(body) if body else None

    async def _remote_api_request(self, method: str, path: str, payload: Any | None = None) -> Any:
        return await asyncio.to_thread(self._remote_api_request_sync, method, path, payload)

    async def _remote_api_request_with_retry(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
        *,
        attempts: int = 4,
        base_sleep_seconds: float = 1.0,
    ) -> Any:
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                return await self._remote_api_request(method, path, payload)
            except Exception as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(base_sleep_seconds * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    async def _remote_api_run_job(
        self,
        path: str,
        payload: Any,
        *,
        poll_seconds: float = 5.0,
        log_context: dict[str, Any] | None = None,
    ) -> Any:
        started = time.monotonic()
        context = dict(log_context or {})
        if isinstance(payload, dict) and payload.get("run_dir"):
            context.setdefault("run_dir", payload.get("run_dir"))
        context["path"] = path

        job = await self._remote_api_request_with_retry("POST", path, payload)
        job_id = str(job["job_id"])
        LOGGER.info(
            "Submitted protein binder remote API job: %s",
            json.dumps({**context, "job_id": job_id, "initial_status": job.get("status")}, sort_keys=True),
        )

        last_status: str | None = None
        last_slurm_state: str | None = None
        while True:
            status_payload = await self._remote_api_request_with_retry("GET", f"/v1/jobs/{job_id}")
            status = str(status_payload.get("status") or "unknown")
            slurm_state = status_payload.get("slurm_state")
            if status != last_status or slurm_state != last_slurm_state:
                LOGGER.info(
                    "Protein binder remote API job status: %s",
                    json.dumps(
                        {
                            **context,
                            "job_id": job_id,
                            "status": status,
                            "slurm_job_id": status_payload.get("slurm_job_id"),
                            "slurm_state": slurm_state,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                        },
                        sort_keys=True,
                    ),
                )
                last_status = status
                last_slurm_state = slurm_state
            if status == "completed":
                result = status_payload.get("result")
                LOGGER.info(
                    "Completed protein binder remote API job: %s",
                    json.dumps(
                        {
                            **context,
                            "job_id": job_id,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "result_summary": _remote_api_result_summary(result),
                        },
                        sort_keys=True,
                    ),
                )
                return result
            if status == "failed":
                result = status_payload.get("result")
                LOGGER.warning(
                    "Protein binder remote API job failed: %s",
                    json.dumps(
                        {
                            **context,
                            "job_id": job_id,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            "result_summary": _remote_api_result_summary(result),
                        },
                        sort_keys=True,
                    ),
                )
                return result
            if time.monotonic() - started > self.remote_api_timeout_seconds:
                raise RuntimeError(f"Remote API job timed out after {self.remote_api_timeout_seconds}s: {path}")
            await asyncio.sleep(poll_seconds)

    def _ensure_ssh_key_path(self) -> Path | None:
        if self._ssh_key_path is not None:
            return self._ssh_key_path

        key_text = os.environ.get("PROTEIN_BINDER_SSH_PRIVATE_KEY")
        key_b64 = os.environ.get("PROTEIN_BINDER_SSH_PRIVATE_KEY_B64")
        if key_b64:
            key_text = base64.b64decode(key_b64).decode()
        if not key_text:
            return None

        key_dir = Path(tempfile.mkdtemp(prefix="protein-binder-ssh-"))
        key_path = key_dir / "id_ed25519"
        key_path.write_text(key_text)
        key_path.chmod(0o600)
        self._ssh_key_dir = key_dir
        self._ssh_key_path = key_path
        return key_path

    def _ssh_command_prefix(self) -> list[str]:
        command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=30",
            "-o",
            "ConnectionAttempts=3",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]
        key_path = self._ensure_ssh_key_path()
        if key_path is not None:
            command.extend(["-i", str(key_path)])
        command.append(self.remote_host)
        return command

    def _parse_remote_host(self) -> tuple[str, str, int]:
        if "@" in self.remote_host:
            username, host_part = self.remote_host.split("@", 1)
        else:
            username, host_part = "ubuntu", self.remote_host
        if ":" in host_part:
            host, port_text = host_part.rsplit(":", 1)
            if port_text.isdigit():
                return username, host, int(port_text)
        return username, host_part, 22

    def _connect_paramiko_client(self, *, force_reconnect: bool = False) -> paramiko.SSHClient:
        if force_reconnect:
            self._close_paramiko_client()
        elif self._paramiko_client is not None:
            transport = self._paramiko_client.get_transport()
            if transport is not None and transport.is_active():
                return self._paramiko_client
            self._close_paramiko_client()

        key_path = self._ensure_ssh_key_path()
        if key_path is None:
            raise RuntimeError("Paramiko fallback requires PROTEIN_BINDER_SSH_PRIVATE_KEY or PROTEIN_BINDER_SSH_PRIVATE_KEY_B64")

        username, hostname, port = self._parse_remote_host()
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            hostname=hostname,
            port=port,
            username=username,
            key_filename=str(key_path),
            look_for_keys=False,
            allow_agent=False,
            timeout=30,
            banner_timeout=30,
            auth_timeout=30,
        )
        self._paramiko_client = client
        return client

    def _close_paramiko_client(self) -> None:
        if self._paramiko_client is None:
            return
        try:
            self._paramiko_client.close()
        except Exception:
            pass
        self._paramiko_client = None

    def _run_paramiko_command_sync(self, remote_command: str) -> str:
        client = self._connect_paramiko_client()
        command = ["paramiko", self.remote_host, remote_command]
        try:
            _, stdout, stderr = client.exec_command(f"bash -c {shlex.quote(remote_command)}")
            out_text = stdout.read().decode()
            err_text = stderr.read().decode()
            returncode = stdout.channel.recv_exit_status()
            if returncode != 0:
                raise CommandError(command, returncode, out_text, err_text)
            return out_text
        except Exception:
            self._close_paramiko_client()
            raise

    async def _run_local_command(self, command: list[str], *, stdin: bytes | None = None) -> str:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate(stdin)
        out_text = stdout.decode()
        err_text = stderr.decode()
        if process.returncode != 0:
            raise CommandError(command, process.returncode, out_text, err_text)
        return out_text

    def _is_retryable_ssh_exception(self, exc: Exception) -> bool:
        retryable_markers = [
            "kex_exchange_identification",
            "connection reset by",
            "connection closed by",
            "broken pipe",
            "error reading ssh protocol banner",
            "unable to connect",
            "operation timed out",
            "connection timed out",
            "connection refused",
        ]
        haystacks = [str(exc).lower()]
        if isinstance(exc, CommandError):
            haystacks.extend([exc.stdout.lower(), exc.stderr.lower()])
        return any(marker in haystack for marker in retryable_markers for haystack in haystacks)

    async def _run_remote_command_once(self, remote_command: str) -> str:
        if self._ensure_ssh_key_path() is not None:
            return await asyncio.to_thread(self._run_paramiko_command_sync, remote_command)
        if shutil.which("ssh"):
            return await self._run_local_command([
                *self._ssh_command_prefix(),
                f"bash -c {shlex.quote(remote_command)}",
            ])
        return await asyncio.to_thread(self._run_paramiko_command_sync, remote_command)

    def _wrap_remote_with_host_lock(self, remote_command: str, *, wait_seconds: int = 7200) -> str:
        return (
            f"mkdir -p {shlex.quote(self._remote_lock_dir)} && "
            f"flock -w {wait_seconds} {shlex.quote(self._remote_host_lock_path)} "
            f"bash -lc {shlex.quote(remote_command)}"
        )

    async def _run_remote_command(self, remote_command: str) -> str:
        async with self._remote_command_lock:
            last_exc: Exception | None = None
            for attempt in range(4):
                try:
                    return await self._run_remote_command_once(remote_command)
                except Exception as exc:
                    last_exc = exc
                    if not self._is_retryable_ssh_exception(exc) or attempt == 3:
                        raise
                    self._close_paramiko_client()
                    backoff_seconds = min(8.0, 1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
                    await asyncio.sleep(backoff_seconds)
            assert last_exc is not None
            raise last_exc

    async def _read_remote_json(self, remote_path: str) -> Any:
        output = await self._run_remote_command(f"cat {shlex.quote(remote_path)}")
        return json.loads(output)

    async def _tail_remote_file(self, remote_path: str, lines: int = 80) -> str:
        try:
            return await self._run_remote_command(f"tail -n {lines} {shlex.quote(remote_path)}")
        except CommandError as exc:
            return exc.stdout or exc.stderr

    def _sync_support_via_paramiko_sync(self, archive_bytes: bytes, sync_command: str) -> None:
        client = self._connect_paramiko_client()
        remote_archive_path = f"/tmp/protein-binder-support-{uuid.uuid4().hex}.tar.gz"
        try:
            with client.open_sftp() as sftp:
                with sftp.file(remote_archive_path, "wb") as remote_file:
                    remote_file.write(archive_bytes)
            sync_with_archive = sync_command.replace("tar -xzf -", f"tar -xzf {shlex.quote(remote_archive_path)}")
            _, stdout, stderr = client.exec_command("bash -c " + shlex.quote(sync_with_archive))
            out_text = stdout.read().decode()
            err_text = stderr.read().decode()
            returncode = stdout.channel.recv_exit_status()
            if returncode != 0:
                raise CommandError(["paramiko-sync", self.remote_host], returncode, out_text, err_text)
        except Exception:
            self._close_paramiko_client()
            raise
        finally:
            try:
                client.exec_command(f"rm -f {shlex.quote(remote_archive_path)}")
            except Exception:
                pass

    async def _ensure_remote_support_synced(self) -> None:
        if self._api_enabled() or not self.sync_support_on_start or self._support_synced:
            return
        async with self._support_sync_lock:
            if self._support_synced:
                return
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                for child in sorted(self.local_support_dir.iterdir()):
                    archive.add(child, arcname=child.name)
            archive_bytes = buffer.getvalue()
            support_hash = hashlib.sha256(archive_bytes).hexdigest()
            remote_hash_path = f"{self.remote_support_dir}/.support_hash"
            tmp_support_dir = f"{self.remote_support_dir}.incoming-{uuid.uuid4().hex}"
            previous_support_dir = f"{self.remote_support_dir}.bak"
            sync_command = self._wrap_remote_with_host_lock(
                " && ".join(
                    [
                        (
                            f"if [ -f {shlex.quote(remote_hash_path)} ] && "
                            f"[ \"$(cat {shlex.quote(remote_hash_path)})\" = {shlex.quote(support_hash)} ]; "
                            "then exit 0; fi"
                        ),
                        f"rm -rf {shlex.quote(tmp_support_dir)} {shlex.quote(previous_support_dir)}",
                        f"mkdir -p {shlex.quote(tmp_support_dir)}",
                        f"tar -xzf - -C {shlex.quote(tmp_support_dir)}",
                        f"printf %s {shlex.quote(support_hash)} > {shlex.quote(tmp_support_dir + '/.support_hash')}",
                        (
                            f"if [ -d {shlex.quote(self.remote_support_dir)} ]; "
                            f"then mv {shlex.quote(self.remote_support_dir)} {shlex.quote(previous_support_dir)}; fi"
                        ),
                        f"mv {shlex.quote(tmp_support_dir)} {shlex.quote(self.remote_support_dir)}",
                        f"rm -rf {shlex.quote(previous_support_dir)}",
                    ]
                ),
                wait_seconds=600,
            )

            async with self._remote_command_lock:
                last_exc: Exception | None = None
                for attempt in range(4):
                    try:
                        if self._ensure_ssh_key_path() is not None or not shutil.which("ssh"):
                            await asyncio.to_thread(self._sync_support_via_paramiko_sync, archive_bytes, sync_command)
                        else:
                            await self._run_local_command(
                                [*self._ssh_command_prefix(), f"bash -c {shlex.quote(sync_command)}"],
                                stdin=archive_bytes,
                            )
                        self._support_synced = True
                        return
                    except Exception as exc:
                        last_exc = exc
                        if not self._is_retryable_ssh_exception(exc) or attempt == 3:
                            raise
                        self._close_paramiko_client()
                        backoff_seconds = min(8.0, 1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
                        await asyncio.sleep(backoff_seconds)
                assert last_exc is not None
                raise last_exc

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        await self._ensure_remote_support_synced()
        rollout_id = uuid.uuid4().hex[:12]
        task_info = dict(state.get("info", {}) or {})
        remote_run_dir = f"{self.remote_run_root}/{rollout_id}"
        state["remote_run_dir"] = remote_run_dir
        state["remote_log_dir"] = f"{remote_run_dir}/state"
        state["stage_history"] = []
        state["stage_call_counts"] = {stage: 0 for stage in self.PIPELINE_STAGES}
        state["successful_stage_counts"] = {stage: 0 for stage in self.PIPELINE_STAGES}
        state["stage_error_count"] = 0
        state["candidate_lookup"] = {}
        state["best_candidate_id"] = ""
        state["best_passing_candidate_id"] = ""
        state["num_passing_candidates"] = 0
        state["best_passing_score"] = 0.0
        state["target_mean_plddt"] = 0.0
        state["target_summary"] = {}
        state["backbones"] = []
        state["candidates"] = []
        state["binder_candidate_results"] = []
        state["run_summary"] = {}
        state["environment_resolution"] = {
            "transport": "http" if self._api_enabled() else "ssh",
            "sync_support_on_start": self.sync_support_on_start,
            "keep_remote_artifacts": self.keep_remote_artifacts,
            "remote_host": self.remote_host,
            "remote_api_base_url": self.remote_api_base_url,
            "remote_run_root": self.remote_run_root,
        }

        LOGGER.info(
            "Setting up protein binder rollout: %s",
            json.dumps(
                {
                    "rollout_id": rollout_id,
                    "remote_run_dir": remote_run_dir,
                    "target_id": task_info.get("target_id"),
                    "task_library": task_info.get("task_library"),
                    "binder_length_min": task_info.get("binder_length_min"),
                    "binder_length_max": task_info.get("binder_length_max"),
                    "num_designs": task_info.get("num_designs"),
                    "num_seqs_per_backbone": task_info.get("num_seqs_per_backbone"),
                    "candidate_batch_size": task_info.get("candidate_batch_size"),
                    "transport": state["environment_resolution"]["transport"],
                },
                sort_keys=True,
            ),
        )

        gate = task_info["quality_gate"]
        remote_target_fasta = f"{remote_run_dir}/inputs/target_sequence.fasta"
        fasta_content = f">{task_info['target_id']}\n{task_info['target_sequence']}\n"
        write_fasta_code = (
            "from pathlib import Path; "
            f"Path({json.dumps(remote_target_fasta)}).parent.mkdir(parents=True, exist_ok=True); "
            f"Path({json.dumps(remote_target_fasta)}).write_text({json.dumps(fasta_content)})"
        )
        init_log_path = f"{remote_run_dir}/state/init-run.log"
        init_parts = [
            "python3",
            shlex.quote(f"{self.remote_support_dir}/run_monomer_pipeline.py"),
            "init-run",
            "--run-dir",
            shlex.quote(remote_run_dir),
            "--target-sequence-fasta",
            shlex.quote(remote_target_fasta),
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
            ">",
            shlex.quote(init_log_path),
            "2>&1",
        ]
        init_command = self._wrap_remote_with_host_lock(
            f"mkdir -p {shlex.quote(remote_run_dir)}/state && "
            f"python3 -c {shlex.quote(write_fasta_code)} && "
            f"{' '.join(init_parts)}"
        )
        if self._api_enabled():
            payload = await self._remote_api_run_job(
                "/v1/jobs/init-run",
                {
                    "run_dir": remote_run_dir,
                    "target_id": task_info["target_id"],
                    "target_sequence": task_info["target_sequence"],
                    "target_chain": task_info["target_chain"],
                    "hotspots": task_info["hotspots"],
                    "binder_length_min": task_info["binder_length_min"],
                    "binder_length_max": task_info["binder_length_max"],
                    "num_designs": task_info["num_designs"],
                    "num_seqs_per_backbone": task_info["num_seqs_per_backbone"],
                    "candidate_batch_size": task_info["candidate_batch_size"],
                    "quality_gate": gate,
                },
                log_context={"operation": "init-run", "target_id": task_info["target_id"]},
            )
            if payload.get("error"):
                raise RuntimeError(f"Failed to initialize remote run.\n{payload.get('log_tail') or payload}")
        else:
            try:
                await self._run_remote_command(init_command)
            except CommandError as exc:
                init_log = (await self._tail_remote_file(init_log_path)).strip()
                fallback_output = "\n".join(
                    part for part in [exc.stdout.strip(), exc.stderr.strip()] if part
                )
                error_output = init_log
                if fallback_output:
                    error_output = (
                        f"{init_log}\n{fallback_output}" if init_log and fallback_output not in init_log else fallback_output
                    )
                raise RuntimeError(f"Failed to initialize remote run.\n{error_output}")
        return await super().setup_state(state, **kwargs)

    @vf.cleanup
    async def cleanup_remote_run(self, state: vf.State):
        if self.keep_remote_artifacts:
            return
        remote_run_dir = state.get("remote_run_dir")
        if remote_run_dir:
            try:
                if self._api_enabled():
                    await self._remote_api_run_job(
                        "/v1/jobs/delete-run",
                        {"run_dir": remote_run_dir},
                        log_context={"operation": "delete-run"},
                    )
                else:
                    await self._run_remote_command(f"rm -rf {shlex.quote(remote_run_dir)}")
            except Exception:
                pass

    @vf.teardown
    async def cleanup_ssh_key(self):
        self._close_paramiko_client()
        if self._ssh_key_dir is not None:
            shutil.rmtree(self._ssh_key_dir, ignore_errors=True)
        self._ssh_key_dir = None
        self._ssh_key_path = None

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
        return None

    async def _execute_stage(
        self,
        *,
        state: vf.State,
        stage_name: str,
        subcommand: str,
        result_loader: str,
    ) -> dict[str, Any]:
        stage_call_counts = dict(state.get("stage_call_counts", {}) or {})
        stage_call_counts[stage_name] = int(stage_call_counts.get(stage_name, 0) or 0) + 1
        state["stage_call_counts"] = stage_call_counts

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
        start_time = time.monotonic()
        LOGGER.info(
            "Starting protein binder stage: %s",
            json.dumps(
                {
                    "remote_run_dir": remote_run_dir,
                    "stage_name": stage_name,
                    "subcommand": subcommand,
                    "transport": "http" if self._api_enabled() else "ssh",
                    "stage_history": state.get("stage_history", []),
                },
                sort_keys=True,
            ),
        )
        if self._api_enabled():
            payload = await self._remote_api_run_job(
                f"/v1/jobs/stages/{subcommand}",
                {"run_dir": remote_run_dir},
                log_context={"operation": "stage", "stage_name": stage_name, "subcommand": subcommand},
            )
            if isinstance(payload, dict) and payload.get("error"):
                state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
                LOGGER.warning(
                    "Protein binder stage failed: %s",
                    json.dumps(
                        {
                            "remote_run_dir": remote_run_dir,
                            "stage_name": stage_name,
                            "elapsed_seconds": round(time.monotonic() - start_time, 3),
                            "summary": _stage_payload_summary(stage_name, payload),
                        },
                        sort_keys=True,
                    ),
                )
                return payload
        else:
            try:
                await self._run_remote_command(self._wrap_remote_with_host_lock(command))
            except CommandError:
                state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
                payload = {
                    "error": "stage_execution_failed",
                    "stage": stage_name,
                    "log_tail": await self._tail_remote_file(log_path),
                }
                LOGGER.warning(
                    "Protein binder stage failed: %s",
                    json.dumps(
                        {
                            "remote_run_dir": remote_run_dir,
                            "stage_name": stage_name,
                            "elapsed_seconds": round(time.monotonic() - start_time, 3),
                            "summary": _stage_payload_summary(stage_name, payload),
                        },
                        sort_keys=True,
                    ),
                )
                return payload

            payload = await self._read_remote_json(f"{remote_run_dir}/{result_loader}")
        if stage_name != "summarize_candidates":
            _invalidate_downstream_stage_state(state, stage_name)
        state["stage_history"] = [*state.get("stage_history", []), stage_name]
        state[_stage_state_key(stage_name)] = payload
        successful_stage_counts = dict(state.get("successful_stage_counts", {}) or {})
        successful_stage_counts[stage_name] = int(successful_stage_counts.get(stage_name, 0) or 0) + 1
        state["successful_stage_counts"] = successful_stage_counts
        if stage_name == "target_monomer":
            state["target_mean_plddt"] = float(payload["target_mean_plddt"] or 0.0)
        if stage_name == "summarize_candidates":
            ranked_candidates = _sort_candidates_by_science_reward(list(payload.get("ranked_candidates", []) or []), state)
            payload["ranked_candidates"] = ranked_candidates
            payload["best_candidate"] = ranked_candidates[0] if ranked_candidates else None
            payload["best_passing_candidate"] = next(
                (candidate for candidate in ranked_candidates if candidate.get("passes_quality_gate")),
                None,
            )
            payload["passing_candidate_ids"] = [
                candidate.get("candidate_id") for candidate in ranked_candidates if candidate.get("passes_quality_gate")
            ]
            payload["num_passing_candidates"] = len(payload["passing_candidate_ids"])
            total_candidates = len(ranked_candidates)
            state["candidate_lookup"] = {
                str(candidate.get("candidate_id")): _candidate_state_record(
                    candidate,
                    state,
                    rank_index=index,
                    total=total_candidates,
                )
                for index, candidate in enumerate(ranked_candidates)
                if candidate.get("candidate_id")
            }
            best_candidate = payload.get("best_candidate") or {}
            best_passing = payload.get("best_passing_candidate") or {}
            state["best_candidate_id"] = str(best_candidate.get("candidate_id", ""))
            state["best_passing_candidate_id"] = str(best_passing.get("candidate_id", ""))
            state["best_passing_score"] = float(best_passing.get("monomer_plausibility_score", 0.0) or 0.0)
            state["num_passing_candidates"] = int(payload.get("num_passing_candidates", 0) or 0)
        LOGGER.info(
            "Completed protein binder stage: %s",
            json.dumps(
                {
                    "remote_run_dir": remote_run_dir,
                    "stage_name": stage_name,
                    "elapsed_seconds": round(time.monotonic() - start_time, 3),
                    "stage_history": state.get("stage_history", []),
                    "summary": _stage_payload_summary(stage_name, payload),
                },
                sort_keys=True,
            ),
        )
        return payload

    async def run_target_monomer(self, state: vf.State) -> str:
        """Run the target ColabFold monomer stage.

        Returns:
            JSON with the exact tool input and the full target-stage output payload.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="target_monomer",
            subcommand="target-monomer",
            result_loader="state/target_summary.json",
        )
        return _stage_response_payload(state, stage_name="target_monomer", payload=payload)

    async def run_rfdiffusion(self, state: vf.State) -> str:
        """Run RFdiffusion with the task's configured design budget.

        Returns:
            JSON with the exact tool input and the full generated-backbone payload.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="rfdiffusion",
            subcommand="rfdiffusion",
            result_loader="state/backbones.json",
        )
        return _stage_response_payload(state, stage_name="rfdiffusion", payload=payload)

    async def run_proteinmpnn(self, state: vf.State) -> str:
        """Run ProteinMPNN on the generated backbones.

        Returns:
            JSON with the exact tool input and the full candidate-sequence payload.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="proteinmpnn",
            subcommand="proteinmpnn",
            result_loader="state/candidates.json",
        )
        return _stage_response_payload(state, stage_name="proteinmpnn", payload=payload)

    async def run_binder_monomer(self, state: vf.State) -> str:
        """Run monomer scoring for all binder candidates.

        Returns:
            JSON with the exact tool input and the full scored-candidate payload.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="binder_monomer",
            subcommand="binder-monomer",
            result_loader="state/binder_candidate_results.json",
        )
        return _stage_response_payload(state, stage_name="binder_monomer", payload=payload)

    async def summarize_candidates(self, state: vf.State) -> str:
        """Summarize the remote run and surface the full comparison payload.

        Returns:
            JSON with the exact tool input and the full run summary payload.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="summarize_candidates",
            subcommand="summarize",
            result_loader="summary/run_summary.json",
        )
        return _stage_response_payload(state, stage_name="summarize_candidates", payload=payload)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _value_source(*, env_name: str, argument_value: Any, default_value: Any) -> str:
    if os.environ.get(env_name) is not None:
        return f"env:{env_name}"
    if argument_value != default_value:
        return "argument"
    return "default"


def _stage_payload_summary(stage_name: str, payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        if payload.get("error"):
            return {
                "error": payload.get("error"),
                "stage": payload.get("stage", stage_name),
            }
        if stage_name == "target_monomer":
            return {
                "target_mean_plddt": payload.get("target_mean_plddt"),
                "target_ptm": payload.get("target_ptm"),
            }
        if stage_name == "summarize_candidates":
            return {
                "num_candidates": payload.get("num_candidates"),
                "num_passing_candidates": payload.get("num_passing_candidates"),
                "best_candidate_id": (payload.get("best_candidate") or {}).get("candidate_id"),
                "best_passing_candidate_id": (payload.get("best_passing_candidate") or {}).get("candidate_id"),
            }
        return {"keys": sorted(payload.keys())[:8]}
    if isinstance(payload, list):
        return {"num_items": len(payload)}
    return {"payload_type": type(payload).__name__}


def load_environment(
    num_train_examples: int = 96,
    num_eval_examples: int = 24,
    max_turns: int = 30,
    remote_host: str = "ubuntu@154.54.100.216",
    remote_support_dir: str = "/home/ubuntu/pi-workspace/protein-binder/experiments/real_monomer_harness",
    remote_run_root: str = "/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real",
    keep_remote_artifacts: bool = False,
    sync_support_on_start: bool = True,
    remote_api_base_url: str | None = None,
    remote_api_token_env_var: str = "PROTEIN_BINDER_API_TOKEN",
    remote_api_timeout_seconds: int = 43200,
    task_library: str = "ronig",
    train_seed: int = 7,
    eval_seed: int = 17,
) -> vf.Environment:
    support_dir = Path(__file__).resolve().parent / "support"
    resolved_remote_api_base_url = remote_api_base_url or os.environ.get("PROTEIN_BINDER_REMOTE_API_BASE_URL")
    resolved_num_train_examples = _env_int("PROTEIN_BINDER_NUM_TRAIN_EXAMPLES", num_train_examples)
    resolved_num_eval_examples = _env_int("PROTEIN_BINDER_NUM_EVAL_EXAMPLES", num_eval_examples)
    resolved_max_turns = _env_int("PROTEIN_BINDER_MAX_TURNS", max_turns)
    resolved_train_seed = _env_int("PROTEIN_BINDER_TRAIN_SEED", train_seed)
    resolved_eval_seed = _env_int("PROTEIN_BINDER_EVAL_SEED", eval_seed)
    resolved_task_library = os.environ.get("PROTEIN_BINDER_TASK_LIBRARY", task_library)
    resolved_keep_remote_artifacts = _env_flag("PROTEIN_BINDER_KEEP_REMOTE_ARTIFACTS", keep_remote_artifacts)
    resolved_sync_support_on_start = _env_flag(
        "PROTEIN_BINDER_SYNC_SUPPORT_ON_START",
        False if resolved_remote_api_base_url else sync_support_on_start,
    )

    train_dataset, eval_dataset = build_datasets(
        resolved_num_train_examples,
        resolved_num_eval_examples,
        task_library=resolved_task_library,
        train_seed=resolved_train_seed,
        eval_seed=resolved_eval_seed,
    )

    LOGGER.info(
        "Resolved protein binder environment config: %s",
        json.dumps(
            {
                "num_train_examples": {
                    "value": resolved_num_train_examples,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_NUM_TRAIN_EXAMPLES",
                        argument_value=num_train_examples,
                        default_value=96,
                    ),
                },
                "num_eval_examples": {
                    "value": resolved_num_eval_examples,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_NUM_EVAL_EXAMPLES",
                        argument_value=num_eval_examples,
                        default_value=24,
                    ),
                },
                "max_turns": {
                    "value": resolved_max_turns,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_MAX_TURNS",
                        argument_value=max_turns,
                        default_value=30,
                    ),
                },
                "train_seed": {
                    "value": resolved_train_seed,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_TRAIN_SEED",
                        argument_value=train_seed,
                        default_value=7,
                    ),
                },
                "eval_seed": {
                    "value": resolved_eval_seed,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_EVAL_SEED",
                        argument_value=eval_seed,
                        default_value=17,
                    ),
                },
                "task_library": {
                    "value": resolved_task_library,
                    "source": _value_source(
                        env_name="PROTEIN_BINDER_TASK_LIBRARY",
                        argument_value=task_library,
                        default_value="ronig",
                    ),
                },
                "keep_remote_artifacts": resolved_keep_remote_artifacts,
                "sync_support_on_start": resolved_sync_support_on_start,
                "transport": "http" if resolved_remote_api_base_url else "ssh",
                "remote_api_base_url": resolved_remote_api_base_url,
                "remote_host": remote_host,
                "dataset_sizes": {
                    "train": len(train_dataset),
                    "eval": len(eval_dataset),
                },
            },
            sort_keys=True,
        ),
    )

    parser = vf.XMLParser(["candidate_id"], answer_field="candidate_id")
    rubric = vf.Rubric(
        funcs=[candidate_selection_reward, tool_overuse_penalty_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.25, 0.01],
        parser=parser,
    )
    rubric.add_metric(submitted_candidate_known_metric)
    rubric.add_metric(submitted_candidate_passes_quality_gate)
    rubric.add_metric(submitted_candidate_is_best_candidate)
    rubric.add_metric(submitted_candidate_rank_percentile_metric)
    rubric.add_metric(submitted_candidate_quality_metric)
    rubric.add_metric(submitted_candidate_science_reward_metric)
    rubric.add_metric(submitted_candidate_plausibility_component_metric)
    rubric.add_metric(submitted_candidate_geometry_component_metric)
    rubric.add_metric(submitted_candidate_binder_confidence_component_metric)
    rubric.add_metric(submitted_candidate_hotspot_component_metric)
    rubric.add_metric(submitted_candidate_interface_component_metric)
    rubric.add_metric(pipeline_completed_metric)
    rubric.add_metric(passing_candidate_available_metric)
    rubric.add_metric(num_passing_candidates_metric)
    rubric.add_metric(best_passing_score_metric)
    rubric.add_metric(target_mean_plddt_metric)
    rubric.add_metric(stage_error_metric)
    rubric.add_metric(total_stage_calls_metric)
    rubric.add_metric(tool_overuse_penalty_metric)

    return ProteinBinderMonomerRealEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=SYSTEM_PROMPT,
        max_turns=resolved_max_turns,
        remote_host=remote_host,
        remote_support_dir=remote_support_dir,
        remote_run_root=remote_run_root,
        local_support_dir=str(support_dir),
        keep_remote_artifacts=resolved_keep_remote_artifacts,
        sync_support_on_start=resolved_sync_support_on_start,
        remote_api_base_url=resolved_remote_api_base_url,
        remote_api_token_env_var=remote_api_token_env_var,
        remote_api_timeout_seconds=remote_api_timeout_seconds,
    )
