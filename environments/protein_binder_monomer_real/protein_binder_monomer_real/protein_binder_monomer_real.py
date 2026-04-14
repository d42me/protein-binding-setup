from __future__ import annotations

import asyncio
import base64
import io
import json
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
    return state.get("stage_history", []) == PIPELINE_STAGES


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

    quality = float(candidate.get("monomer_plausibility_score", 0.0) or 0.0)
    rank_percentile = float(candidate.get("rank_percentile", 0.0) or 0.0)
    pass_bonus = 1.0 if candidate.get("passes_quality_gate") else 0.0
    return round((0.6 * quality) + (0.25 * rank_percentile) + (0.15 * pass_bonus), 3)


async def pipeline_progress_reward(completion: vf.Messages, state: vf.State) -> float:
    return round(len(state.get("stage_history", [])) / len(PIPELINE_STAGES), 3)


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


def _rank_percentile(index: int, total: int) -> float:
    if total <= 1:
        return 1.0
    return round(1.0 - (index / (total - 1)), 3)


def _candidate_state_record(candidate: dict[str, Any], *, rank_index: int, total: int) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "sequence": candidate.get("sequence", ""),
        "passes_quality_gate": bool(candidate.get("passes_quality_gate")),
        "monomer_plausibility_score": float(candidate.get("monomer_plausibility_score", 0.0) or 0.0),
        "binder_mean_plddt": float(candidate.get("binder_mean_plddt", 0.0) or 0.0),
        "binder_distance_rmse": candidate.get("binder_distance_rmse"),
        "hotspot_fraction": float(candidate.get("hotspot_fraction", 0.0) or 0.0),
        "interface_residue_contacts": int(candidate.get("interface_residue_contacts", 0) or 0),
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

    async def _remote_api_run_job(self, path: str, payload: Any, *, poll_seconds: float = 5.0) -> Any:
        started = time.monotonic()
        job = await self._remote_api_request("POST", path, payload)
        job_id = str(job["job_id"])
        while True:
            status_payload = await self._remote_api_request("GET", f"/v1/jobs/{job_id}")
            status = status_payload.get("status")
            if status == "completed":
                return status_payload.get("result")
            if status == "failed":
                return status_payload.get("result")
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
        state["stage_error_count"] = 0
        state["candidate_lookup"] = {}
        state["best_candidate_id"] = ""
        state["best_passing_candidate_id"] = ""
        state["num_passing_candidates"] = 0
        state["best_passing_score"] = 0.0
        state["target_mean_plddt"] = 0.0

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
                    await self._remote_api_run_job("/v1/jobs/delete-run", {"run_dir": remote_run_dir})
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
        if self._api_enabled():
            payload = await self._remote_api_run_job(
                f"/v1/jobs/stages/{subcommand}",
                {"run_dir": remote_run_dir},
            )
            if isinstance(payload, dict) and payload.get("error"):
                state["stage_error_count"] = int(state.get("stage_error_count", 0)) + 1
                return payload
        else:
            try:
                await self._run_remote_command(self._wrap_remote_with_host_lock(command))
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
            ranked_candidates = list(payload.get("ranked_candidates", []) or [])
            total_candidates = len(ranked_candidates)
            state["candidate_lookup"] = {
                str(candidate.get("candidate_id")): _candidate_state_record(
                    candidate,
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
            "stage_complete": "target_monomer",
            "next_required_tool": "run_rfdiffusion",
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
                "stage_complete": "rfdiffusion",
                "next_required_tool": "run_proteinmpnn",
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
            return json.dumps(
                {
                    "stage_complete": "proteinmpnn",
                    "next_required_tool": "run_binder_monomer",
                    "num_candidates": len(payload),
                },
                indent=2,
                sort_keys=True,
            )
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
            return json.dumps(
                {
                    "stage_complete": "binder_monomer",
                    "next_required_tool": "summarize_candidates",
                    "num_batches": len(payload),
                    "batches": payload,
                },
                indent=2,
                sort_keys=True,
            )
        return json.dumps(payload, indent=2, sort_keys=True)

    async def summarize_candidates(self, state: vf.State) -> str:
        """Summarize the remote run and surface a candidate comparison table.

        Returns:
            JSON containing stripped candidate metrics for candidate-ID selection.
        """
        payload = await self._execute_stage(
            state=state,
            stage_name="summarize_candidates",
            subcommand="summarize",
            result_loader="summary/run_summary.json",
        )
        if payload.get("error"):
            return json.dumps(payload, indent=2, sort_keys=True)
        public_candidates = sorted(
            [_public_candidate_view(candidate) for candidate in payload.get("ranked_candidates", [])],
            key=lambda candidate: str(candidate.get("candidate_id", "")),
        )
        summary = {
            "stage_complete": "summarize_candidates",
            "final_response_format": "<candidate_id>CANDIDATE_ID</candidate_id>",
            "selection_objective": "Choose the single strongest candidate ID from the surfaced metrics. Do not output raw sequence text.",
            "num_candidates": payload.get("num_candidates"),
            "num_passing_candidates": payload.get("num_passing_candidates"),
            "candidates": public_candidates,
        }
        return json.dumps(summary, indent=2, sort_keys=True)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_environment(
    num_train_examples: int = 4,
    num_eval_examples: int = 1,
    max_turns: int = 8,
    remote_host: str = "ubuntu@154.54.100.216",
    remote_support_dir: str = "/home/ubuntu/pi-workspace/protein-binder/experiments/real_monomer_harness",
    remote_run_root: str = "/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real",
    keep_remote_artifacts: bool = False,
    sync_support_on_start: bool = True,
    remote_api_base_url: str | None = None,
    remote_api_token_env_var: str = "PROTEIN_BINDER_API_TOKEN",
    remote_api_timeout_seconds: int = 43200,
    task_library: str = "all",
) -> vf.Environment:
    support_dir = Path(__file__).resolve().parent / "support"
    resolved_remote_api_base_url = remote_api_base_url or os.environ.get("PROTEIN_BINDER_REMOTE_API_BASE_URL")
    resolved_task_library = os.environ.get("PROTEIN_BINDER_TASK_LIBRARY", task_library)
    resolved_keep_remote_artifacts = _env_flag("PROTEIN_BINDER_KEEP_REMOTE_ARTIFACTS", keep_remote_artifacts)
    resolved_sync_support_on_start = _env_flag(
        "PROTEIN_BINDER_SYNC_SUPPORT_ON_START",
        False if resolved_remote_api_base_url else sync_support_on_start,
    )

    train_dataset, eval_dataset = build_datasets(
        num_train_examples,
        num_eval_examples,
        task_library=resolved_task_library,
    )

    parser = vf.XMLParser(["candidate_id"], answer_field="candidate_id")
    rubric = vf.Rubric(
        funcs=[candidate_selection_reward, pipeline_progress_reward, parser.get_format_reward_func()],
        weights=[1.0, 0.15, 0.05],
        parser=parser,
    )
    rubric.add_metric(submitted_candidate_known_metric)
    rubric.add_metric(submitted_candidate_passes_quality_gate)
    rubric.add_metric(submitted_candidate_is_best_candidate)
    rubric.add_metric(submitted_candidate_rank_percentile_metric)
    rubric.add_metric(submitted_candidate_quality_metric)
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
        keep_remote_artifacts=resolved_keep_remote_artifacts,
        sync_support_on_start=resolved_sync_support_on_start,
        remote_api_base_url=resolved_remote_api_base_url,
        remote_api_token_env_var=remote_api_token_env_var,
        remote_api_timeout_seconds=remote_api_timeout_seconds,
    )
