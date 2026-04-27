from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prime_cli.api.rl import RLClient, RLRun
from prime_cli.core import APIClient, APIError

from submit_hosted_rl_with_top_level import build_payload

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "STOPPED", "CANCELLED"}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def now_iso() -> str:
    return utc_now().isoformat()


def append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload) + "\n")


def call_with_retries(
    func: callable,
    *,
    log_path: Path | None,
    event: str,
    max_attempts: int = 5,
    retry_delay_seconds: int = 10,
):
    last_error: APIError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except APIError as exc:
            last_error = exc
            append_jsonl(
                log_path,
                {
                    "ts": now_iso(),
                    "event": event,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "error": str(exc),
                },
            )
            if attempt == max_attempts:
                break
            time.sleep(retry_delay_seconds)
    assert last_error is not None
    raise last_error


def submit_run(
    api_client: APIClient,
    config_path: str,
    top_level: dict[str, Any],
    cli_env: list[str] | None,
    cli_env_files: list[str] | None,
    log_path: Path | None,
) -> RLRun:
    payload = build_payload(config_path, cli_env, cli_env_files)
    payload.update(top_level)
    response = call_with_retries(
        lambda: api_client.post("/rft/runs", json=payload),
        log_path=log_path,
        event="submit_retryable_error",
    )
    return RLRun.model_validate(response.get("run"))


def monitor_run(
    rl_client: RLClient,
    run_id: str,
    poll_seconds: int,
    no_sample_timeout_minutes: int | None,
    log_path: Path | None,
) -> dict[str, Any]:
    while True:
        run = call_with_retries(
            lambda: rl_client.get_run(run_id),
            log_path=log_path,
            event="get_run_retryable_error",
        )
        progress = call_with_retries(
            lambda: rl_client.get_progress(run_id),
            log_path=log_path,
            event="get_progress_retryable_error",
        )
        metrics = call_with_retries(
            lambda: rl_client.get_metrics(run_id, limit=1),
            log_path=log_path,
            event="get_metrics_retryable_error",
        )
        latest_metrics = metrics[-1] if metrics else None
        snapshot = {
            "ts": now_iso(),
            "event": "poll",
            "run_id": run.id,
            "status": run.status,
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "latest_step": progress.get("latest_step"),
            "steps_with_samples": progress.get("steps_with_samples", []),
            "steps_with_distributions": progress.get("steps_with_distributions", []),
            "latest_metric_step": latest_metrics.get("step") if latest_metrics else None,
        }
        append_jsonl(log_path, snapshot)

        if run.status in TERMINAL_STATUSES:
            return snapshot

        started_at = normalize_utc(run.started_at)
        if (
            no_sample_timeout_minutes is not None
            and run.status == "RUNNING"
            and started_at is not None
            and not progress.get("steps_with_samples")
        ):
            elapsed_minutes = (utc_now() - started_at).total_seconds() / 60.0
            if elapsed_minutes >= no_sample_timeout_minutes:
                stopped = call_with_retries(
                    lambda: rl_client.stop_run(run.id),
                    log_path=log_path,
                    event="stop_run_retryable_error",
                )
                snapshot = {
                    "ts": now_iso(),
                    "event": "stopped_no_samples_timeout",
                    "run_id": stopped.id,
                    "status": stopped.status,
                    "elapsed_minutes": elapsed_minutes,
                }
                append_jsonl(log_path, snapshot)
                return snapshot

        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--wait-for-run", default=None)
    parser.add_argument("--top-level-json", default="{}")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--no-sample-timeout-minutes", type=int, default=90)
    parser.add_argument("--log-path", default=None)
    parser.add_argument("-e", "--env-var", action="append", default=None)
    parser.add_argument("--env-file", action="append", default=None)
    args = parser.parse_args()

    top_level = json.loads(args.top_level_json)
    if not isinstance(top_level, dict):
        raise SystemExit("--top-level-json must decode to a JSON object")

    log_path = Path(args.log_path) if args.log_path else None
    api_client = APIClient()
    rl_client = RLClient(api_client)

    if args.wait_for_run:
        append_jsonl(log_path, {"ts": now_iso(), "event": "wait_for_existing_run", "run_id": args.wait_for_run})
        result = monitor_run(
            rl_client,
            args.wait_for_run,
            poll_seconds=args.poll_seconds,
            no_sample_timeout_minutes=args.no_sample_timeout_minutes,
            log_path=log_path,
        )
        print(json.dumps(result, indent=2))

    for config_path in args.configs:
        run = submit_run(api_client, config_path, top_level, args.env_var, args.env_file, log_path)
        submitted = {
            "ts": now_iso(),
            "event": "submitted",
            "config_path": config_path,
            "run_id": run.id,
            "status": run.status,
            "model": run.base_model,
            "name": run.name,
        }
        append_jsonl(log_path, submitted)
        print(json.dumps(submitted, indent=2))
        result = monitor_run(
            rl_client,
            run.id,
            poll_seconds=args.poll_seconds,
            no_sample_timeout_minutes=args.no_sample_timeout_minutes,
            log_path=log_path,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
