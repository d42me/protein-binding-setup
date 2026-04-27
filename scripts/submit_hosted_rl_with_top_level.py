from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prime_cli.api.rl import RLRun
from prime_cli.commands.rl import load_config
from prime_cli.core import APIClient, APIError, Config
from prime_cli.utils.env_vars import EnvParseError, collect_env_vars


def build_payload(config_path: str, cli_env: list[str] | None, cli_env_files: list[str] | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    app_config = Config()

    config_dir = Path(config_path).parent
    config_env_files = cfg.env_file + cfg.env_files
    resolved_config_env_files = [str(config_dir / p) for p in config_env_files]
    all_env_files = resolved_config_env_files + (cli_env_files or [])

    try:
        secrets = collect_env_vars(env_args=cli_env, env_files=all_env_files or None)
    except EnvParseError as exc:
        raise SystemExit(f"Failed to collect env vars: {exc}") from exc

    payload: dict[str, Any] = {
        "model": {"name": cfg.model},
        "environments": [e.to_api_dict() for e in cfg.env],
        "rollouts_per_example": cfg.rollouts_per_example,
        "max_steps": cfg.max_steps,
        "batch_size": cfg.batch_size,
        "secrets": [{"key": k, "value": v} for k, v in (secrets or {}).items()],
    }

    if cfg.name:
        payload["name"] = cfg.name
    if app_config.team_id:
        payload["team_id"] = app_config.team_id
    if cfg.sampling.max_tokens is not None:
        payload["max_tokens"] = cfg.sampling.max_tokens
    if cfg.sampling.temperature is not None:
        payload["temperature"] = cfg.sampling.temperature
    if cfg.sampling.repetition_penalty is not None:
        payload["repetition_penalty"] = cfg.sampling.repetition_penalty
    if cfg.sampling.min_tokens is not None:
        payload["min_tokens"] = cfg.sampling.min_tokens
    if cfg.sampling.seed is not None:
        payload["seed"] = cfg.sampling.seed
    if cfg.sampling.temp_scheduler is not None:
        payload["temp_scheduler"] = cfg.sampling.temp_scheduler.model_dump(exclude_none=True)
    if cfg.sampling.extra_body is not None:
        payload["extra_body"] = cfg.sampling.extra_body
    if cfg.eval.to_api_dict() is not None:
        payload["eval"] = cfg.eval.to_api_dict()
    if cfg.val.to_api_dict() is not None:
        payload["val"] = cfg.val.to_api_dict()
    if cfg.buffer.to_api_dict() is not None:
        payload["buffer"] = cfg.buffer.to_api_dict()
    if cfg.learning_rate is not None:
        payload["learning_rate"] = cfg.learning_rate
    if cfg.lora_alpha is not None:
        payload["lora_alpha"] = cfg.lora_alpha
    if cfg.oversampling_factor is not None:
        payload["oversampling_factor"] = cfg.oversampling_factor
    if cfg.max_async_level is not None:
        payload["max_async_level"] = cfg.max_async_level
    if cfg.checkpoints.to_api_dict() is not None:
        payload["checkpoints"] = cfg.checkpoints.to_api_dict()
    if cfg.adapters.to_api_dict() is not None:
        payload["adapters"] = cfg.adapters.to_api_dict()
    if cfg.checkpoint_id is not None:
        payload["checkpoint_id"] = cfg.checkpoint_id
    if cfg.cluster_name is not None:
        payload["cluster_name"] = cfg.cluster_name
    if cfg.infrastructure.to_api_dict() is not None:
        infrastructure = cfg.infrastructure.to_api_dict()
        if infrastructure and "compute_size" in infrastructure:
            payload["compute_size"] = infrastructure["compute_size"]
    if cfg.wandb.entity or cfg.wandb.project or cfg.wandb.name:
        wandb: dict[str, Any] = {}
        if cfg.wandb.entity:
            wandb["entity"] = cfg.wandb.entity
        if cfg.wandb.project:
            wandb["project"] = cfg.wandb.project
        if cfg.wandb.name:
            wandb["name"] = cfg.wandb.name
        payload["monitoring"] = {"wandb": wandb}

    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument(
        "--top-level-json",
        default="{}",
        help='Extra top-level fields to merge into the /rft/runs payload, e.g. {"max_concurrent":1}',
    )
    parser.add_argument("-e", "--env-var", action="append", default=None)
    parser.add_argument("--env-file", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    payload = build_payload(args.config_path, args.env_var, args.env_file)
    extra_top_level = json.loads(args.top_level_json)
    if not isinstance(extra_top_level, dict):
        raise SystemExit("--top-level-json must decode to a JSON object")
    payload.update(extra_top_level)

    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    client = APIClient()
    try:
        response = client.post("/rft/runs", json=payload)
    except APIError as exc:
        raise SystemExit(f"Failed to create RL run: {exc}") from exc

    run = RLRun.model_validate(response.get("run"))
    frontend_url = Config().frontend_url
    print(json.dumps({
        "id": run.id,
        "status": run.status,
        "model": run.base_model,
        "name": run.name,
        "dashboard_url": f"{frontend_url}/dashboard/training/{run.id}",
        "run_config": run.run_config,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
