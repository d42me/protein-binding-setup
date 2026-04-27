#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

APP = FastAPI(title="protein-binder-monomer-real-api")
LOGGER = logging.getLogger("protein_binder_monomer_real.api")
SUPPORT_DIR = Path(__file__).resolve().parent
API_SERVER_PATH = Path(__file__).resolve()
PIPELINE_SCRIPT = SUPPORT_DIR / "run_monomer_pipeline.py"
LOCK_DIR = Path("/tmp/protein-binder-monomer-real-locks")
HOST_LOCK_PATH = LOCK_DIR / "host.lock"
DEFAULT_RUN_ROOT = Path("/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real").resolve()
API_TOKEN = os.environ.get("PROTEIN_BINDER_API_TOKEN", "").strip()
ALLOWED_RUN_ROOT = Path(
    os.environ.get("PROTEIN_BINDER_API_ALLOWED_RUN_ROOT", str(DEFAULT_RUN_ROOT))
).expanduser().resolve()
EXECUTOR = os.environ.get("PROTEIN_BINDER_API_EXECUTOR", "local").strip().lower()
JOB_STATE_DIR = Path(
    os.environ.get("PROTEIN_BINDER_API_JOB_STATE_DIR", "/tmp/protein-binder-monomer-real-api-jobs")
).expanduser().resolve()
SLURM_GPU_PARTITION = os.environ.get("PROTEIN_BINDER_API_SLURM_GPU_PARTITION", "").strip()
SLURM_CPU_PARTITION = os.environ.get("PROTEIN_BINDER_API_SLURM_CPU_PARTITION", "").strip()
SLURM_ACCOUNT = os.environ.get("PROTEIN_BINDER_API_SLURM_ACCOUNT", "").strip()
SLURM_QOS = os.environ.get("PROTEIN_BINDER_API_SLURM_QOS", "").strip()
SLURM_EXTRA_ARGS = shlex.split(os.environ.get("PROTEIN_BINDER_API_SLURM_EXTRA_ARGS", ""))
STAGE_LOCK_MODE = os.environ.get("PROTEIN_BINDER_API_STAGE_LOCK_MODE", "host").strip().lower()

STAGE_RESULT_LOADERS: dict[str, str] = {
    "target-monomer": "state/target_summary.json",
    "rfdiffusion": "state/backbones.json",
    "proteinmpnn": "state/candidates.json",
    "binder-monomer": "state/binder_batches.json",
    "summarize": "summary/run_summary.json",
}

GPU_STAGES = {"target-monomer", "rfdiffusion", "proteinmpnn", "binder-monomer"}
CPU_STAGES = {"summarize"}
JOB_WRITE_LOCK = threading.Lock()


class QualityGateModel(BaseModel):
    min_target_mean_plddt: float
    min_binder_mean_plddt: float
    max_binder_distance_rmse: float
    min_hotspot_fraction: float
    min_interface_residue_contacts: int
    score_threshold: float


class InitRunRequest(BaseModel):
    run_dir: str
    target_id: str
    target_sequence: str
    target_chain: str
    hotspots: list[str]
    binder_length_min: int
    binder_length_max: int
    num_designs: int
    num_seqs_per_backbone: int
    candidate_batch_size: int
    quality_gate: QualityGateModel


class RunDirRequest(BaseModel):
    run_dir: str


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _payload_run_dir(payload: dict[str, Any]) -> str | None:
    run_dir = payload.get("run_dir")
    return str(run_dir) if run_dir is not None else None


def _result_summary(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        summary: dict[str, Any] = {"type": "dict", "keys": sorted(result.keys())[:24]}
        if result.get("error"):
            summary["error"] = result.get("error")
        return summary
    if isinstance(result, list):
        return {"type": "list", "length": len(result)}
    if result is None:
        return {"type": "none"}
    return {"type": type(result).__name__}


def require_auth(authorization: str | None) -> None:
    if not API_TOKEN:
        raise HTTPException(status_code=500, detail="PROTEIN_BINDER_API_TOKEN is not configured on server")
    expected = f"Bearer {API_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="invalid bearer token")


def resolve_run_dir(run_dir: str) -> Path:
    candidate = Path(run_dir).expanduser().resolve()
    if ALLOWED_RUN_ROOT not in (candidate, *candidate.parents):
        raise HTTPException(status_code=400, detail=f"run_dir must stay under {ALLOWED_RUN_ROOT}")
    return candidate


def maybe_wrap_stage_command(command: str, *, wait_seconds: int = 7200) -> str:
    if STAGE_LOCK_MODE == "none":
        return command
    if STAGE_LOCK_MODE != "host":
        raise RuntimeError(f"Unsupported PROTEIN_BINDER_API_STAGE_LOCK_MODE={STAGE_LOCK_MODE!r}")
    return (
        f"mkdir -p {shlex.quote(str(LOCK_DIR))} && "
        f"flock -w {wait_seconds} {shlex.quote(str(HOST_LOCK_PATH))} "
        f"bash -lc {shlex.quote(command)}"
    )


def stage_command(subcommand: str, run_dir: Path, log_path: Path) -> str:
    command = " ".join(
        [
            "python3",
            shlex.quote(str(PIPELINE_SCRIPT)),
            subcommand,
            "--run-dir",
            shlex.quote(str(run_dir)),
            ">",
            shlex.quote(str(log_path)),
            "2>&1",
        ]
    )
    if subcommand == "proteinmpnn":
        return f"unset CUDA_VISIBLE_DEVICES && {command}"
    return command


def run_shell(command: str) -> tuple[int, str, str]:
    completed = subprocess.run(["bash", "-lc", command], text=True, capture_output=True)
    return completed.returncode, completed.stdout, completed.stderr


def run_subprocess(command: list[str]) -> tuple[int, str, str]:
    completed = subprocess.run(command, text=True, capture_output=True)
    return completed.returncode, completed.stdout, completed.stderr


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(path)


def tail_file(path: Path, lines: int = 120) -> str:
    if not path.exists():
        return ""
    return "\n".join(path.read_text().splitlines()[-lines:])


def job_dir(job_id: str) -> Path:
    return JOB_STATE_DIR / job_id


def job_record_path(job_id: str) -> Path:
    return job_dir(job_id) / "record.json"


def load_job_record(job_id: str) -> dict[str, Any]:
    path = job_record_path(job_id)
    if not path.exists():
        raise FileNotFoundError(f"Unknown job_id: {job_id}")
    return read_json(path)


def save_job_record(record: dict[str, Any]) -> None:
    with JOB_WRITE_LOCK:
        write_json_atomic(job_record_path(record["job_id"]), record)


def update_job_record(job_id: str, **updates: Any) -> dict[str, Any]:
    record = load_job_record(job_id)
    record.update(updates)
    save_job_record(record)
    return record


def slurm_logs_for(job_id: str) -> tuple[Path, Path]:
    base = job_dir(job_id)
    return base / "slurm-%j.out", base / "slurm-%j.err"


def slurm_resource_spec(kind: str, stage: str | None) -> dict[str, Any]:
    if kind == "init-run":
        return {"partition": SLURM_CPU_PARTITION, "gpus": 0, "cpus": 2, "memory_mb": 4096, "time": "00:10:00"}
    if kind == "delete-run":
        return {"partition": SLURM_CPU_PARTITION, "gpus": 0, "cpus": 1, "memory_mb": 2048, "time": "00:10:00"}
    if stage in GPU_STAGES:
        return {"partition": SLURM_GPU_PARTITION, "gpus": 1, "cpus": 8, "memory_mb": 32768, "time": "08:00:00"}
    if stage in CPU_STAGES:
        return {"partition": SLURM_CPU_PARTITION, "gpus": 0, "cpus": 8, "memory_mb": 16384, "time": "02:00:00"}
    raise ValueError(f"No SLURM resource spec for kind={kind!r}, stage={stage!r}")


def lookup_slurm_state(slurm_job_id: str) -> str | None:
    returncode, stdout, _ = run_subprocess(["squeue", "-h", "-j", slurm_job_id, "-o", "%T"])
    if returncode == 0 and stdout.strip():
        return stdout.strip().splitlines()[0]
    returncode, stdout, _ = run_subprocess(["sacct", "-n", "-X", "-j", slurm_job_id, "--format=State"])
    if returncode == 0 and stdout.strip():
        return stdout.strip().splitlines()[0].strip()
    return None


def build_slurm_submission(job_id: str, kind: str, stage: str | None) -> dict[str, Any]:
    spec = slurm_resource_spec(kind, stage)
    stdout_path, stderr_path = slurm_logs_for(job_id)
    worker_command = [sys.executable, str(API_SERVER_PATH), "__run-job", job_id]
    args = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(SUPPORT_DIR),
        "--job-name",
        f"pb-{(stage or kind).replace('-', '_')}-{job_id[:8]}",
        "--output",
        str(stdout_path),
        "--error",
        str(stderr_path),
        "--open-mode=append",
        "--cpus-per-task",
        str(spec["cpus"]),
        "--mem",
        str(spec["memory_mb"]),
        "--time",
        spec["time"],
    ]
    partition = spec["partition"]
    if partition:
        args.extend(["--partition", partition])
    if spec["gpus"]:
        args.extend(["--gres", f"gpu:{spec['gpus']}"])
    if SLURM_ACCOUNT:
        args.extend(["--account", SLURM_ACCOUNT])
    if SLURM_QOS:
        args.extend(["--qos", SLURM_QOS])
    args.extend(SLURM_EXTRA_ARGS)
    args.extend(["--wrap", " ".join(shlex.quote(part) for part in worker_command)])
    return {
        "resource_spec": spec,
        "worker_command": worker_command,
        "sbatch_args": args,
    }



def submit_slurm_job(job_id: str, kind: str, stage: str | None) -> str:
    submission = build_slurm_submission(job_id, kind, stage)
    args = list(submission["sbatch_args"])
    returncode, stdout, stderr = run_subprocess(args)
    if returncode != 0:
        raise RuntimeError(f"sbatch failed: {stderr.strip() or stdout.strip()}")
    slurm_job_id = stdout.strip().split(";", 1)[0].strip()
    if not slurm_job_id:
        raise RuntimeError("sbatch did not return a job id")
    return slurm_job_id



def _init_command_preview(payload: dict[str, Any]) -> str:
    run_dir = resolve_run_dir(str(payload["run_dir"]))
    remote_target_fasta = run_dir / "inputs" / "target_sequence.fasta"
    init_log_path = run_dir / "state" / "init-run.log"
    fasta_content = f">{payload['target_id']}\n{payload['target_sequence']}\n"
    write_fasta_code = (
        "from pathlib import Path; "
        f"Path({json.dumps(str(remote_target_fasta))}).parent.mkdir(parents=True, exist_ok=True); "
        f"Path({json.dumps(str(remote_target_fasta))}).write_text({json.dumps(fasta_content)})"
    )
    gate = QualityGateModel.model_validate(payload["quality_gate"])
    init_parts = [
        "python3",
        shlex.quote(str(PIPELINE_SCRIPT)),
        "init-run",
        "--run-dir",
        shlex.quote(str(run_dir)),
        "--target-sequence-fasta",
        shlex.quote(str(remote_target_fasta)),
        "--target-chain",
        shlex.quote(str(payload["target_chain"])),
        "--hotspots",
        shlex.quote(",".join(payload["hotspots"])),
        "--binder-length-min",
        str(payload["binder_length_min"]),
        "--binder-length-max",
        str(payload["binder_length_max"]),
        "--num-designs",
        str(payload["num_designs"]),
        "--num-seqs-per-backbone",
        str(payload["num_seqs_per_backbone"]),
        "--candidate-batch-size",
        str(payload["candidate_batch_size"]),
        "--max-concurrent-binder-batches",
        "1",
        "--min-target-mean-plddt",
        str(gate.min_target_mean_plddt),
        "--min-binder-mean-plddt",
        str(gate.min_binder_mean_plddt),
        "--max-binder-distance-rmse",
        str(gate.max_binder_distance_rmse),
        "--min-hotspot-fraction",
        str(gate.min_hotspot_fraction),
        "--min-interface-residue-contacts",
        str(gate.min_interface_residue_contacts),
        "--score-threshold",
        str(gate.score_threshold),
        ">",
        shlex.quote(str(init_log_path)),
        "2>&1",
    ]
    return maybe_wrap_stage_command(
        f"mkdir -p {shlex.quote(str(run_dir / 'state'))} && python3 -c {shlex.quote(write_fasta_code)} && {' '.join(init_parts)}"
    )



def _job_command_preview(kind: str, stage: str | None, payload: dict[str, Any]) -> str:
    if kind == "init-run":
        return _init_command_preview(payload)
    if kind == "delete-run":
        run_dir = resolve_run_dir(str(payload["run_dir"]))
        return f"rm -rf {shlex.quote(str(run_dir))}"
    if kind == "stage":
        run_dir = resolve_run_dir(str(payload["run_dir"]))
        log_path = run_dir / "state" / f"{stage}.log"
        return maybe_wrap_stage_command(stage_command(str(stage), run_dir, log_path))
    raise ValueError(f"Unknown job kind: {kind!r}")



def _job_debug_payload(record: dict[str, Any]) -> dict[str, Any]:
    debug_payload: dict[str, Any] = {
        "job_id": record["job_id"],
        "kind": record["kind"],
        "stage": record.get("stage"),
        "status": record.get("status"),
        "executor": record.get("executor"),
        "slurm_job_id": record.get("slurm_job_id"),
        "payload": record.get("payload"),
        "stdout_log": record.get("stdout_log"),
        "stderr_log": record.get("stderr_log"),
        "command_preview": _job_command_preview(record["kind"], record.get("stage"), record["payload"]),
        "stdout_tail": tail_file(Path(record["stdout_log"]), lines=80) if record.get("stdout_log") else "",
        "stderr_tail": tail_file(Path(record["stderr_log"]), lines=80) if record.get("stderr_log") else "",
    }
    if record.get("slurm_job_id"):
        submission = build_slurm_submission(record["job_id"], record["kind"], record.get("stage"))
        debug_payload["slurm_state"] = lookup_slurm_state(record["slurm_job_id"])
        debug_payload["slurm_resource_spec"] = submission["resource_spec"]
        debug_payload["slurm_worker_command"] = submission["worker_command"]
        debug_payload["slurm_sbatch_command"] = submission["sbatch_args"]
    return debug_payload


def run_init(payload: InitRunRequest) -> dict[str, Any]:
    run_dir = resolve_run_dir(payload.run_dir)
    remote_target_fasta = run_dir / "inputs" / "target_sequence.fasta"
    remote_target_fasta.parent.mkdir(parents=True, exist_ok=True)
    remote_target_fasta.write_text(f">{payload.target_id}\n{payload.target_sequence}\n")

    init_log_path = run_dir / "state" / "init-run.log"
    init_log_path.parent.mkdir(parents=True, exist_ok=True)

    gate = payload.quality_gate
    init_parts = [
        "python3",
        shlex.quote(str(PIPELINE_SCRIPT)),
        "init-run",
        "--run-dir",
        shlex.quote(str(run_dir)),
        "--target-sequence-fasta",
        shlex.quote(str(remote_target_fasta)),
        "--target-chain",
        shlex.quote(payload.target_chain),
        "--hotspots",
        shlex.quote(",".join(payload.hotspots)),
        "--binder-length-min",
        str(payload.binder_length_min),
        "--binder-length-max",
        str(payload.binder_length_max),
        "--num-designs",
        str(payload.num_designs),
        "--num-seqs-per-backbone",
        str(payload.num_seqs_per_backbone),
        "--candidate-batch-size",
        str(payload.candidate_batch_size),
        "--max-concurrent-binder-batches",
        "1",
        "--min-target-mean-plddt",
        str(gate.min_target_mean_plddt),
        "--min-binder-mean-plddt",
        str(gate.min_binder_mean_plddt),
        "--max-binder-distance-rmse",
        str(gate.max_binder_distance_rmse),
        "--min-hotspot-fraction",
        str(gate.min_hotspot_fraction),
        "--min-interface-residue-contacts",
        str(gate.min_interface_residue_contacts),
        "--score-threshold",
        str(gate.score_threshold),
        ">",
        shlex.quote(str(init_log_path)),
        "2>&1",
    ]
    command = maybe_wrap_stage_command(
        f"mkdir -p {shlex.quote(str(run_dir / 'state'))} && {' '.join(init_parts)}"
    )
    returncode, stdout, stderr = run_shell(command)
    if returncode != 0:
        return {
            "error": "init_run_failed",
            "run_dir": str(run_dir),
            "stdout": stdout,
            "stderr": stderr,
            "log_tail": tail_file(init_log_path),
        }
    return {"ok": True, "run_dir": str(run_dir)}


def run_stage(subcommand: str, payload: RunDirRequest) -> Any:
    run_dir = resolve_run_dir(payload.run_dir)
    log_path = run_dir / "state" / f"{subcommand}.log"
    command = maybe_wrap_stage_command(stage_command(subcommand, run_dir, log_path))
    returncode, stdout, stderr = run_shell(command)
    if returncode != 0:
        return {
            "error": "stage_execution_failed",
            "stage": subcommand,
            "stdout": stdout,
            "stderr": stderr,
            "log_tail": tail_file(log_path),
        }

    result_path = run_dir / STAGE_RESULT_LOADERS[subcommand]
    if not result_path.exists():
        return {
            "error": "missing_result_file",
            "stage": subcommand,
            "result_path": str(result_path),
            "log_tail": tail_file(log_path),
        }
    return read_json(result_path)


def delete_run(payload: RunDirRequest) -> dict[str, Any]:
    run_dir = resolve_run_dir(payload.run_dir)
    command = f"rm -rf {shlex.quote(str(run_dir))}"
    returncode, stdout, stderr = run_shell(command)
    return {
        "ok": returncode == 0,
        "run_dir": str(run_dir),
        "stdout": stdout,
        "stderr": stderr,
    }


def execute_job(job_id: str) -> None:
    record = update_job_record(job_id, status="running", started_at=utc_now_iso())
    kind = record["kind"]
    stage = record.get("stage")
    payload = record["payload"]
    started = time.monotonic()
    LOGGER.info(
        "Starting protein binder API job: %s",
        json.dumps(
            {
                "job_id": job_id,
                "kind": kind,
                "stage": stage,
                "run_dir": _payload_run_dir(payload),
                "slurm_job_id": record.get("slurm_job_id"),
            },
            sort_keys=True,
        ),
    )
    try:
        if kind == "init-run":
            result = run_init(InitRunRequest.model_validate(payload))
        elif kind == "delete-run":
            result = delete_run(RunDirRequest.model_validate(payload))
        elif kind == "stage":
            result = run_stage(stage, RunDirRequest.model_validate(payload))
        else:
            raise ValueError(f"Unknown job kind: {kind}")
        status = "failed" if isinstance(result, dict) and result.get("error") else "completed"
        update_job_record(job_id, status=status, result=result, completed_at=utc_now_iso())
        log_fn = LOGGER.warning if status == "failed" else LOGGER.info
        log_fn(
            "Finished protein binder API job: %s",
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "stage": stage,
                    "run_dir": _payload_run_dir(payload),
                    "slurm_job_id": record.get("slurm_job_id"),
                    "status": status,
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "result_summary": _result_summary(result),
                },
                sort_keys=True,
            ),
        )
    except Exception as exc:
        update_job_record(
            job_id,
            status="failed",
            result={"error": "job_exception", "detail": str(exc)},
            completed_at=utc_now_iso(),
        )
        LOGGER.exception(
            "Protein binder API job raised exception: %s",
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "stage": stage,
                    "run_dir": _payload_run_dir(payload),
                    "slurm_job_id": record.get("slurm_job_id"),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                },
                sort_keys=True,
            ),
        )
        raise


def create_job(kind: Literal["init-run", "stage", "delete-run"], payload: BaseModel, *, stage: str | None = None) -> str:
    if EXECUTOR not in {"local", "slurm"}:
        raise RuntimeError(f"Unsupported executor {EXECUTOR!r}; expected 'local' or 'slurm'")

    job_id = uuid.uuid4().hex
    job_root = job_dir(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    payload_dict = payload.model_dump(mode="json")
    record = {
        "job_id": job_id,
        "kind": kind,
        "stage": stage,
        "status": "queued",
        "executor": EXECUTOR,
        "payload": payload_dict,
        "result": None,
        "slurm_job_id": None,
        "created_at": utc_now_iso(),
        "started_at": None,
        "completed_at": None,
        "stdout_log": str(job_root / "slurm-%j.out"),
        "stderr_log": str(job_root / "slurm-%j.err"),
    }
    save_job_record(record)
    LOGGER.info(
        "Queued protein binder API job: %s",
        json.dumps(
            {
                "job_id": job_id,
                "kind": kind,
                "stage": stage,
                "executor": EXECUTOR,
                "run_dir": _payload_run_dir(payload_dict),
            },
            sort_keys=True,
        ),
    )

    if EXECUTOR == "local":
        thread = threading.Thread(target=execute_job, args=(job_id,), daemon=True)
        thread.start()
        return job_id

    try:
        slurm_job_id = submit_slurm_job(job_id, kind=kind, stage=stage)
    except Exception as exc:
        update_job_record(
            job_id,
            status="failed",
            result={"error": "slurm_submit_failed", "detail": str(exc)},
            completed_at=utc_now_iso(),
        )
        LOGGER.warning(
            "Protein binder API job failed during SLURM submission: %s",
            json.dumps(
                {
                    "job_id": job_id,
                    "kind": kind,
                    "stage": stage,
                    "run_dir": _payload_run_dir(payload_dict),
                    "detail": str(exc),
                },
                sort_keys=True,
            ),
        )
        return job_id

    update_job_record(job_id, slurm_job_id=slurm_job_id)
    LOGGER.info(
        "Submitted protein binder API job to SLURM: %s",
        json.dumps(
            {
                "job_id": job_id,
                "kind": kind,
                "stage": stage,
                "run_dir": _payload_run_dir(payload_dict),
                "slurm_job_id": slurm_job_id,
            },
            sort_keys=True,
        ),
    )
    return job_id


@APP.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "pipeline_script": str(PIPELINE_SCRIPT),
        "allowed_run_root": str(ALLOWED_RUN_ROOT),
        "executor": EXECUTOR,
        "job_state_dir": str(JOB_STATE_DIR),
        "stage_lock_mode": STAGE_LOCK_MODE,
    }


@APP.post("/v1/jobs/init-run")
def start_init_job(payload: InitRunRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    job_id = create_job("init-run", payload)
    record = load_job_record(job_id)
    return {"job_id": job_id, "status": record["status"]}


@APP.post("/v1/jobs/stages/{subcommand}")
def start_stage_job(
    subcommand: Literal["target-monomer", "rfdiffusion", "proteinmpnn", "binder-monomer", "summarize"],
    payload: RunDirRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_auth(authorization)
    job_id = create_job("stage", payload, stage=subcommand)
    record = load_job_record(job_id)
    return {"job_id": job_id, "status": record["status"]}


@APP.post("/v1/jobs/delete-run")
def start_delete_job(payload: RunDirRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    job_id = create_job("delete-run", payload)
    record = load_job_record(job_id)
    return {"job_id": job_id, "status": record["status"]}


@APP.get("/v1/jobs/{job_id}")
def get_job(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    try:
        job = load_job_record(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="unknown job_id") from None
    if job.get("slurm_job_id"):
        job["slurm_state"] = lookup_slurm_state(job["slurm_job_id"])
    return job


@APP.get("/v1/jobs/{job_id}/debug")
def get_job_debug(job_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    try:
        job = load_job_record(job_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="unknown job_id") from None
    return _job_debug_payload(job)


@APP.get("/v1/debug/slurm")
def get_slurm_debug(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    require_auth(authorization)
    specs = {
        "init-run": slurm_resource_spec("init-run", None),
        "delete-run": slurm_resource_spec("delete-run", None),
        **{stage: slurm_resource_spec("stage", stage) for stage in sorted(GPU_STAGES | CPU_STAGES)},
    }
    return {
        "executor": EXECUTOR,
        "support_dir": str(SUPPORT_DIR),
        "pipeline_script": str(PIPELINE_SCRIPT),
        "stage_lock_mode": STAGE_LOCK_MODE,
        "slurm_gpu_partition": SLURM_GPU_PARTITION,
        "slurm_cpu_partition": SLURM_CPU_PARTITION,
        "slurm_account_configured": bool(SLURM_ACCOUNT),
        "slurm_qos_configured": bool(SLURM_QOS),
        "slurm_extra_args": SLURM_EXTRA_ARGS,
        "resource_specs": specs,
        "job_command_examples": {
            kind: build_slurm_submission(f"example-{kind}", kind if kind in {"init-run", "delete-run"} else "stage", None if kind in {"init-run", "delete-run"} else kind)["sbatch_args"]
            for kind in ["init-run", "delete-run"]
        },
        "stage_job_command_examples": {
            stage: build_slurm_submission(f"example-{stage}", "stage", stage)["sbatch_args"]
            for stage in sorted(GPU_STAGES | CPU_STAGES)
        },
    }


def run_job_cli(job_id: str) -> int:
    try:
        execute_job(job_id)
    except Exception as exc:
        print(f"job {job_id} failed: {exc}", file=sys.stderr)
        return 1
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="protein binder monomer real API server helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_job_parser = subparsers.add_parser("__run-job", help="Internal helper used by SLURM to execute a queued API job")
    run_job_parser.add_argument("job_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli_parser()
    args = parser.parse_args(argv)
    if args.command == "__run-job":
        return run_job_cli(args.job_id)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
