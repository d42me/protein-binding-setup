from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
TOOL_METRIC_KEYS = [
    "run_target_monomer_calls",
    "run_rfdiffusion_calls",
    "run_proteinmpnn_calls",
    "run_binder_monomer_calls",
    "summarize_candidates_calls",
]


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, text=True, capture_output=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(args)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed.stdout


def parse_eval_id(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Evaluation ID:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"Could not parse evaluation id from output:\n{output}")


def fetch_eval(eval_id: str) -> dict[str, Any]:
    output = run_command(["prime", "eval", "get", eval_id, "-o", "json", "--plain"])
    return json.loads(output, strict=False)


def tool_calling_ok(metrics: dict[str, Any]) -> bool:
    if metrics.get("pipeline_completed_metric", 0.0) < 0.999:
        return False
    if metrics.get("stage_error_metric", 1.0) > 0.001:
        return False
    if metrics.get("total_tool_calls", 0.0) < 4.999:
        return False
    if metrics.get("total_stage_calls_metric", metrics.get("total_tool_calls", 0.0)) > 20.001:
        return False
    return all(metrics.get(key, 0.0) >= 0.999 for key in TOOL_METRIC_KEYS)


def summarize_run(run_spec: dict[str, Any], eval_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    metrics_block = payload.get("metrics", {}) if isinstance(payload.get("metrics", {}), dict) else {}
    metrics = metrics_block.get("metrics", {}) if isinstance(metrics_block.get("metrics", {}), dict) else {}
    return {
        "model": run_spec["model"],
        "env_args": run_spec.get("env_args", {}),
        "evaluation_id": eval_id,
        "status": payload.get("status"),
        "created_at": payload.get("created_at"),
        "completed_at": payload.get("completed_at"),
        "avg_score": payload.get("avg_score"),
        "error_rate": metrics_block.get("error"),
        "tool_calling_ok": tool_calling_ok(metrics) if payload.get("status") == "COMPLETED" else False,
        "pipeline_completed_metric": metrics.get("pipeline_completed_metric"),
        "stage_error_metric": metrics.get("stage_error_metric"),
        "total_tool_calls": metrics.get("total_tool_calls"),
        "num_turns": metrics.get("num_turns"),
        "submitted_candidate_known_metric": metrics.get("submitted_candidate_known_metric"),
        "submitted_candidate_passes_quality_gate": metrics.get("submitted_candidate_passes_quality_gate"),
        "submitted_candidate_is_best_candidate": metrics.get("submitted_candidate_is_best_candidate"),
        "submitted_candidate_science_reward_metric": metrics.get("submitted_candidate_science_reward_metric"),
        "passing_candidate_available_metric": metrics.get("passing_candidate_available_metric"),
        "num_passing_candidates_metric": metrics.get("num_passing_candidates_metric"),
        "best_passing_score_metric": metrics.get("best_passing_score_metric"),
        "target_mean_plddt_metric": metrics.get("target_mean_plddt_metric"),
        "total_stage_calls_metric": metrics.get("total_stage_calls_metric"),
        "tool_overuse_penalty_metric": metrics.get("tool_overuse_penalty_metric"),
        **{key: metrics.get(key) for key in TOOL_METRIC_KEYS},
    }


def launch_run(env_id: str, run_spec: dict[str, Any], num_examples: int, timeout_minutes: int) -> str:
    eval_name = f"monomer-real-toolcheck-{run_spec['env_args']['task_library']}-{run_spec['model'].replace('/', '-')}"
    command = [
        "prime",
        "eval",
        "run",
        env_id,
        "-m",
        run_spec["model"],
        "-n",
        str(num_examples),
        "-a",
        json.dumps(run_spec.get("env_args", {}), separators=(",", ":")),
        "--hosted",
        "--timeout-minutes",
        str(timeout_minutes),
        "--eval-name",
        eval_name,
        "-A",
        "--plain",
    ]
    output = run_command(command)
    return parse_eval_id(output)


def wait_for_terminal(eval_id: str, poll_interval_seconds: int) -> dict[str, Any]:
    while True:
        payload = fetch_eval(eval_id)
        if payload.get("status") in TERMINAL_STATUSES:
            return payload
        time.sleep(poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="logs/monomer-real-fleet")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary.jsonl"
    status_path = output_dir / "status.json"

    summaries: list[dict[str, Any]] = []
    for index, run_spec in enumerate(config["runs"], start=1):
        started_at = datetime.now(timezone.utc).isoformat()
        eval_id = launch_run(
            env_id=config["env_id"],
            run_spec=run_spec,
            num_examples=config["num_examples"],
            timeout_minutes=config["timeout_minutes"],
        )
        payload = wait_for_terminal(eval_id, poll_interval_seconds=config["poll_interval_seconds"])
        summary = summarize_run(run_spec, eval_id, payload)
        summary["queue_index"] = index
        summary["launched_at"] = started_at
        summaries.append(summary)
        with summary_path.open("a") as handle:
            handle.write(json.dumps(summary) + "\n")
        status_path.write_text(json.dumps({"completed_runs": len(summaries), "total_runs": len(config["runs"]), "last_summary": summary}, indent=2))
        print(f"[{index}/{len(config['runs'])}] {run_spec['model']} {run_spec.get('env_args', {})} -> {summary['status']} tool_ok={summary['tool_calling_ok']} score={summary['avg_score']}")

    final_report = {
        "env_id": config["env_id"],
        "num_runs": len(summaries),
        "completed": sum(1 for row in summaries if row["status"] == "COMPLETED"),
        "tool_calling_ok": sum(1 for row in summaries if row["tool_calling_ok"]),
        "results": summaries,
    }
    (output_dir / "final-report.json").write_text(json.dumps(final_report, indent=2))
    print(json.dumps(final_report, indent=2))


if __name__ == "__main__":
    main()
