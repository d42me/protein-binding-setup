# protein-binder-monomer-real

SLURM-backed **monomer-only** protein binder environment.

## Overview
- **Environment ID**: `protein-binder-monomer-real`
- **Type**: `StatefulToolEnv`
- **Execution model**: tool calls run the real monomer harness on `ubuntu@154.54.100.216`
- **Current default dataset**: ronig-backed 100-task monomer library with deterministic train/eval shuffles

## Why this environment exists
This environment turns the real monomer harness into rollout tools without requiring AlphaFold-Multimer.

It is intended as a stepping stone toward GPU sandbox support:
- today, rollouts can call the real pipeline either over SSH or through an authenticated FastAPI wrapper on the provided RTX 6000 host
- a **WIP GPU sandbox Dockerfile** is included in this directory so the dev team can try containerizing the same stack later

## Tool contract
Each rollout gets a fresh remote run directory and five scientific tools:
1. `run_target_monomer()`
2. `run_rfdiffusion()`
3. `run_proteinmpnn()`
4. `run_binder_monomer()`
5. `summarize_candidates()`

The tools no longer emit `next_required_tool` hints. The model must keep track of the workflow itself.
Each tool now returns its full structured `tool_input` and `tool_output`, so the model can reason from the actual scientific evidence instead of following hidden scaffolding hints.

After the model has enough evidence, it must answer with only:

```xml
<candidate_id>CANDIDATE_ID</candidate_id>
```

The environment rewards **scientific candidate selection quality**, not raw sequence copying.

The model has more freedom now:
- it can call tools in its own order
- it can repeat stages when exploration is scientifically justified
- the first `30` tool calls are free
- after `30`, additional tool calls incur an increasing penalty

## Monomer-only quality gate
The task currently uses the tuned insulin gate:
- target mean pLDDT >= 80
- binder mean pLDDT >= 80
- binder distance RMSE <= 1.5
- hotspot fraction >= 0.33
- interface residue contacts >= 10
- monomer plausibility score >= 0.72

## Current dataset
The environment now supports three task libraries via `task_library`:
- `proven` — the original 3 hand-validated RFdiffusion examples
- `ronig` — a bundled curated subset of **100** structure-validated tasks derived from `ronig/protein_binding_sequences`
- `all` — the mix of `proven + ronig`

### Proven library
These rows come from a curated **real target set** built from RFdiffusion example structures:
- insulin receptor target
- GABARAP target
- 1YCR target

### ronig curated library
The bundled ronig subset is produced offline by:
- `experiments/real_monomer_harness/scout_ronig_dataset.py`
- `experiments/real_monomer_harness/curate_ronig_dataset.py`

Current curation heuristics are intentionally conservative but broader than before:
- peptide length `30-50`
- receptor length `50-400`
- peptide shorter than receptor
- remove self-like receptor/peptide pairs
- dedupe unordered PDB chain pairs
- require exact PDB ATOM-derived sequence agreement for both chains
- remove highly hydrophobic / transmembrane-like chains with relaxed but still conservative thresholds
- require at least `5` receptor interface residues at `6A`
- derive the top `3` receptor hotspots from residue-level contact counts
- allow up to `2` tasks per exact receptor sequence and up to `2` tasks per exact peptide sequence so the environment can reach useful scale without collapsing diversity
- shuffle the filtered ronig pool deterministically before structural validation so the final library is not just the first alphabetical slice of PDBs

Each rollout now materializes the target from the bundled `target_sequence` directly, so hosted workers do not depend on a pre-existing remote PDB path for dataset tasks.

This is still an early real-tool benchmark, but it now goes materially beyond the previous 3-target loop and is suitable for broader hosted eval scouting.

## Reward design (v0.4.1)
The reward is now deliberately harder: it gives the model more room to explore, but it no longer saturates just because a candidate clears the quality gate.

### Main scientific reward
The dominant term is a **strict nonlinear aggregate** of the same visible candidate metrics:
- monomer plausibility
- geometry quality from low binder distance RMSE
- binder confidence from high binder mean pLDDT
- hotspot coverage
- interface residue contacts

Each component is normalized against a **stricter ceiling** than the basic quality gate, and the final score uses a weighted geometric mean rather than a simple linear average. This means:
- one weak scientific dimension now hurts more
- barely-passing candidates no longer look near-perfect
- high reward requires a frontier-leading candidate with strong headroom across several metrics at once

At summary time, candidates are re-ranked by this same strict scientific objective, so the surfaced top candidate order still matches the main reward.

The main scientific reward is only paid when the rollout has a fresh successful end-to-end summary and the submitted candidate ID is known.

### Tool-call penalty
The first `30` tool calls are free.
After that, the environment applies an **increasing overuse penalty**, so the model has more room to explore and refine candidate evidence before extra search becomes expensive.

### Format reward
A tiny XML-format reward remains for valid final output formatting:

```xml
<candidate_id>CANDIDATE_ID</candidate_id>
```

In practice this makes the main objective much more explicit: explore enough to find strong candidates, then stop once extra tool calls are no longer worth the marginal scientific improvement.

## Quickstart
Install locally:

```bash
prime env install protein-binder-monomer-real
```

Smoke eval:

```bash
prime eval run protein-binder-monomer-real -m gpt-4.1-mini -n 1
```

Keep remote artifacts for inspection:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  -x '{"keep_remote_artifacts": true}'
```

## How to replicate results

This section is the shortest path to reproducing the environment behavior seen in local evals, hosted evals, and the bundled dataset/task curation.

### 1. Reproduce a local smoke run
From the repo root:

```bash
prime env install protein-binder-monomer-real
prime eval run configs/eval/protein-binder-monomer-real-smoke.toml -A
```

This uses the checked-in smoke config:
- env: `protein-binder-monomer-real`
- model: `gpt-4.1-mini`
- examples: `1`
- rollouts per example: `1`

If you want to inspect the exact remote rollout artifacts afterward:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  -x '{"keep_remote_artifacts": true, "task_library": "proven"}'
```

### 2. Reproduce the current default environment behavior
The current checked-in defaults are approximately:
- task library: `ronig`
- train rows: `96`
- eval rows: `24`
- max turns: `30`
- reward design: strict nonlinear scientific candidate selection with post-30-call overuse penalty (`v0.4.1`)

To run that behavior explicitly instead of relying on defaults:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  -a '{"num_train_examples":96,"num_eval_examples":24,"max_turns":30,"task_library":"ronig","train_seed":7,"eval_seed":17}'
```

To lock the task set to the original hand-curated examples:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  -a '{"task_library":"proven"}'
```

To reproduce the broader curated dataset setting:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  -a '{"task_library":"ronig"}'
```

### 3. Reproduce hosted runs
Hosted runs need the same environment installed locally plus the required hosted secrets.

#### SSH transport
Provide:
- `PROTEIN_BINDER_SSH_PRIVATE_KEY_B64`

Then launch:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  --hosted \
  -a '{"task_library":"proven","sync_support_on_start":true}' \
  -A
```

#### HTTP API transport
If the remote host is serving `support/api_server.py`, provide:
- `PROTEIN_BINDER_API_TOKEN`
- `PROTEIN_BINDER_REMOTE_API_BASE_URL`

Then launch:

```bash
prime eval run protein-binder-monomer-real \
  -m gpt-4.1-mini \
  -n 1 \
  --hosted \
  -a '{"task_library":"proven","remote_api_base_url":"https://YOUR-HOST"}' \
  -A
```

In practice, most hosted runs rely on the environment variable / secret `PROTEIN_BINDER_REMOTE_API_BASE_URL`, so you can usually omit the explicit arg and just run:

```bash
prime eval run protein-binder-monomer-real -m gpt-4.1-mini -n 1 --hosted -A
```

### Hosted training / eval observability
Recent versions add more rollout-level observability for the HTTP backend:
- the environment logs remote API job submission, status transitions, completion/failure, and elapsed time with the rollout run dir
- the API server logs queued jobs, SLURM submission, job start, and job completion/failure with `job_id`, `stage`, `run_dir`, and `slurm_job_id`

When debugging hosted training, inspect both:
- platform-side `prime rl logs <run_id>`
- host-side `journalctl -u protein-binder-api` on `31.22.104.83`

### 4. Reproduce training configs used in this repo
The repo keeps multiple historical training configs under `configs/rl/`.

Example:

```bash
prime rl run configs/rl/protein-binder-monomer-real-qwen4b-instruct-proven-bqxlike-v018.toml
```

That config currently pins:
- model: `Qwen/Qwen3-4B-Instruct-2507`
- env id: `d42me/protein-binder-monomer-real@0.1.8`
- task library: `proven`
- rollouts per example: `2`
- batch size: `2`

If you want exact reproducibility, record all of:
- repo commit SHA
- environment version in the config (`@...`)
- task library (`proven`, `ronig`, or `all`)
- transport mode (SSH vs HTTP API)
- remote host / remote API deployment version
- whether `sync_support_on_start` was enabled

### 5. Reproduce the curated ronig task library
The bundled `ronig` library is not magic; it is produced offline from the scouting + curation scripts in `experiments/real_monomer_harness/`.

To regenerate the conservative task library used by the environment:

```bash
python experiments/real_monomer_harness/curate_ronig_dataset.py \
  --max-tasks 100 \
  --selection-seed 7 \
  --output ./environments/protein_binder_monomer_real/protein_binder_monomer_real/data/ronig_curated_tasks.json
```

To scout the upstream dataset before curation:

```bash
python experiments/real_monomer_harness/scout_ronig_dataset.py \
  --min-peptide-length 30 \
  --max-peptide-length 50 \
  --structure-sample-size 20 \
  --output-jsonl ./artifacts/dataset_scout/ronig_30_50_candidates.jsonl
```

### 6. Reproduce the remote API deployment
From the support directory on the remote host, run the FastAPI wrapper with a bearer token configured:

```bash
cd environments/protein_binder_monomer_real/protein_binder_monomer_real/support
export PROTEIN_BINDER_API_TOKEN=...
export PROTEIN_BINDER_API_EXECUTOR=local
uvicorn api_server:APP --host 0.0.0.0 --port 8000
```

For SLURM-backed execution, switch the executor and set the server-side partition variables before starting the API process:

```bash
cd environments/protein_binder_monomer_real/protein_binder_monomer_real/support
export PROTEIN_BINDER_API_TOKEN=...
export PROTEIN_BINDER_API_EXECUTOR=slurm
export PROTEIN_BINDER_API_SLURM_GPU_PARTITION=...
export PROTEIN_BINDER_API_SLURM_CPU_PARTITION=...
uvicorn api_server:APP --host 0.0.0.0 --port 8000
```

### 7. What to log when claiming a result is reproduced
At minimum, save:
- Prime evaluation ID
- repo commit SHA
- environment version / env ID
- model name
- task library
- transport mode
- whether remote artifacts were kept
- remote API executor mode if HTTP API was used

Without those, two runs can look similar while still differing in important ways.

Hosted evals can use either transport:

### SSH mode
Provide an environment secret or custom secret named:
- `PROTEIN_BINDER_SSH_PRIVATE_KEY_B64`

This should contain a base64-encoded private key that is authorized on the target RTX 6000 pod.
The environment will materialize that key at runtime and use it for both SSH and support-file sync.

### HTTP API mode
If the remote host is running the packaged `support/api_server.py` FastAPI wrapper, set either:
- `remote_api_base_url` as an environment arg, or
- `PROTEIN_BINDER_REMOTE_API_BASE_URL` as an environment variable/hosted secret

Also provide:
- secret env var `PROTEIN_BINDER_API_TOKEN`

Optional env-var overrides for hosted runs:
- `PROTEIN_BINDER_NUM_TRAIN_EXAMPLES`
- `PROTEIN_BINDER_NUM_EVAL_EXAMPLES`
- `PROTEIN_BINDER_MAX_TURNS`
- `PROTEIN_BINDER_TRAIN_SEED`
- `PROTEIN_BINDER_EVAL_SEED`
- `PROTEIN_BINDER_TASK_LIBRARY`
- `PROTEIN_BINDER_KEEP_REMOTE_ARTIFACTS`
- `PROTEIN_BINDER_SYNC_SUPPORT_ON_START`

Best practice:
- prefer normal `load_environment(...)` args for local development and reproducibility
- use the `PROTEIN_BINDER_*` numeric env vars mainly on hosted runs, where constructor arg propagation can be flaky
- when using these overrides, log the requested values next to the eval ID so run metadata stays interpretable later

In HTTP mode, the environment skips SSH transport entirely and calls the remote harness through the bearer-token API.

The packaged API server now supports two execution backends selected by server-side env var:
- `PROTEIN_BINDER_API_EXECUTOR=local` — existing in-process background-thread execution
- `PROTEIN_BINDER_API_EXECUTOR=slurm` — submit each API job as a SLURM job and persist job state on disk

Useful SLURM server env vars:
- `PROTEIN_BINDER_API_JOB_STATE_DIR`
- `PROTEIN_BINDER_API_SLURM_GPU_PARTITION`
- `PROTEIN_BINDER_API_SLURM_CPU_PARTITION`
- `PROTEIN_BINDER_API_SLURM_ACCOUNT`
- `PROTEIN_BINDER_API_SLURM_QOS`
- `PROTEIN_BINDER_API_SLURM_EXTRA_ARGS`

Both modes preserve the same HTTP contract (`/v1/jobs/init-run`, `/v1/jobs/stages/...`, `/v1/jobs/delete-run`, `/v1/jobs/{job_id}`), so the environment client does not need to change when the remote host switches from a single serialized worker to a SLURM-backed executor.

## Environment arguments
| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `num_train_examples` | int | `96` | Number of train rows sampled from the task library |
| `num_eval_examples` | int | `24` | Number of eval rows sampled from the task library |
| `max_turns` | int | `30` | Maximum assistant turns |
| `remote_host` | str | `ubuntu@154.54.100.216` | Remote RTX 6000 host |
| `remote_support_dir` | str | `/home/ubuntu/pi-workspace/protein-binder/experiments/real_monomer_harness` | Remote path where the harness scripts are synced |
| `remote_run_root` | str | `/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real` | Root directory for per-rollout remote artifacts |
| `keep_remote_artifacts` | bool | `false` | Keep remote run directories after rollout cleanup |
| `sync_support_on_start` | bool | `true` | Atomically sync the packaged harness support files to the remote host during setup |
| `remote_api_base_url` | str \| null | `null` | Optional bearer-token FastAPI endpoint for the remote harness |
| `remote_api_token_env_var` | str | `"PROTEIN_BINDER_API_TOKEN"` | Environment variable name containing the bearer token for HTTP mode |
| `remote_api_timeout_seconds` | int | `43200` | Socket timeout for long-running HTTP stage requests |
| `task_library` | str | `"ronig"` | Which bundled task pool to use: `proven`, `ronig`, or `all` |
| `train_seed` | int | `7` | Deterministic shuffle seed for selecting train rows from the task library |
| `eval_seed` | int | `17` | Deterministic shuffle seed for selecting eval rows from the task library |

## Metrics
| Metric | Meaning |
| --- | --- |
| `reward` | Main scalar reward: strict nonlinear scientific candidate score minus any post-30-call overuse penalty |
| `submitted_candidate_known_metric` | Final answer references a known candidate ID |
| `submitted_candidate_passes_quality_gate` | Final answer selects a candidate that passes the quality gate |
| `submitted_candidate_is_best_candidate` | Final answer selects the internally top-ranked candidate |
| `submitted_candidate_rank_percentile_metric` | Percentile rank of the selected candidate among all candidates |
| `submitted_candidate_quality_metric` | Raw monomer plausibility score of the selected candidate |
| `submitted_candidate_science_reward_metric` | Strict nonlinear scientific reward of the selected candidate |
| `submitted_candidate_plausibility_component_metric` | Normalized plausibility component used by the main reward |
| `submitted_candidate_geometry_component_metric` | Normalized geometry / RMSE component used by the main reward |
| `submitted_candidate_binder_confidence_component_metric` | Normalized binder pLDDT component used by the main reward |
| `submitted_candidate_hotspot_component_metric` | Normalized hotspot coverage component used by the main reward |
| `submitted_candidate_interface_component_metric` | Normalized interface-contact component used by the main reward |
| `pipeline_completed_metric` | Every required stage has succeeded at least once on the current fresh pipeline state |
| `passing_candidate_available_metric` | At least one passing candidate existed |
| `num_passing_candidates_metric` | Number of passing candidates from `summarize_candidates()` |
| `best_passing_score_metric` | Monomer plausibility score of the best passing candidate |
| `target_mean_plddt_metric` | Target monomer quality |
| `total_stage_calls_metric` | Total number of stage-tool calls in the rollout |
| `tool_overuse_penalty_metric` | Current overuse penalty after the free 20-call budget |
| `stage_error_metric` | Count of failed or invalid stage executions |

## GPU sandbox images
This directory includes two image tracks:

1. root `Dockerfile`
   - Blackwell-oriented WIP image validated on the remote RTX 6000 host
2. `experimental_h200_sandbox/`
   - experimental H200-targeted bundle for future GPU sandbox work
   - uses a stable CUDA 12.4 / PyTorch 2.4 stack instead of the Blackwell nightly stack
   - includes entrypoint and smoke-test scripts plus a README describing the remaining sandbox requirements

The current environment does **not** require either image yet. Today it still executes over SSH on the remote RTX 6000 host.
