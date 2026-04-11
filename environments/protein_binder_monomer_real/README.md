# protein-binder-monomer-real

Remote RTX 6000-backed **monomer-only** protein binder environment.

## Overview
- **Environment ID**: `protein-binder-monomer-real`
- **Type**: `StatefulToolEnv`
- **Execution model**: tool calls run the real monomer harness on `ubuntu@154.54.100.216`
- **Current target**: insulin chain A example with tuned monomer-only search settings

## Why this environment exists
This environment turns the real monomer harness into rollout tools without requiring AlphaFold-Multimer.

It is intended as a stepping stone toward GPU sandbox support:
- today, rollouts call the real pipeline over SSH on the provided RTX 6000 host
- a **WIP GPU sandbox Dockerfile** is included in this directory so the dev team can try containerizing the same stack later

## Tool contract
Each rollout gets a fresh remote run directory and should use the tools in this exact order:
1. `run_target_monomer()`
2. `run_rfdiffusion()`
3. `run_proteinmpnn()`
4. `run_binder_monomer()`
5. `summarize_candidates()`

After `summarize_candidates()`, the model must answer with only:

```xml
<sequence>SEQUENCE</sequence>
```

The environment rewards the model when the submitted sequence matches **any passing candidate** from the monomer-only quality gate.

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
- `ronig` — a bundled curated subset derived from `ronig/protein_binding_sequences`
- `all` — the default mix of `proven + ronig`

### Proven library
These rows come from a curated **real target set** built from RFdiffusion example structures:
- insulin receptor target
- GABARAP target
- 1YCR target

### ronig curated library
The bundled ronig subset is produced offline by:
- `experiments/real_monomer_harness/scout_ronig_dataset.py`
- `experiments/real_monomer_harness/curate_ronig_dataset.py`

Current curation heuristics are intentionally conservative:
- peptide length `30-50`
- receptor length `50-400`
- peptide shorter than receptor
- remove self-like receptor/peptide pairs
- dedupe unordered PDB chain pairs
- require exact PDB ATOM-derived sequence agreement for both chains
- remove highly hydrophobic / transmembrane-like chains
- require at least `5` receptor interface residues at `6A`
- derive the top `3` receptor hotspots from residue-level contact counts
- keep one task per exact receptor sequence and one per exact peptide sequence for diversity

Each rollout now materializes the target from the bundled `target_sequence` directly, so hosted workers do not depend on a pre-existing remote PDB path for dataset tasks.

This is still an early real-tool benchmark, but it now goes materially beyond the previous 3-target loop and is suitable for broader hosted eval scouting.

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

Hosted evals can use the same SSH-backed pod path.
For hosted runs, provide an environment secret or custom secret named:
- `PROTEIN_BINDER_SSH_PRIVATE_KEY_B64`

This should contain a base64-encoded private key that is authorized on the target RTX 6000 pod.
The environment will materialize that key at runtime and use it for both SSH and support-file sync.

## Environment arguments
| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `num_train_examples` | int | `4` | Number of train rows |
| `num_eval_examples` | int | `1` | Number of eval rows |
| `max_turns` | int | `8` | Maximum assistant turns |
| `remote_host` | str | `ubuntu@154.54.100.216` | Remote RTX 6000 host |
| `remote_support_dir` | str | `/home/ubuntu/pi-workspace/protein-binder/experiments/real_monomer_harness` | Remote path where the harness scripts are synced |
| `remote_run_root` | str | `/home/ubuntu/protein-runtime/rollouts/protein-binder-monomer-real` | Root directory for per-rollout remote artifacts |
| `keep_remote_artifacts` | bool | `false` | Keep remote run directories after rollout cleanup |
| `sync_support_on_start` | bool | `true` | Rsync the packaged harness support files to the remote host during setup |
| `task_library` | str | `"all"` | Which bundled task pool to use: `proven`, `ronig`, or `all` |

## Metrics
| Metric | Meaning |
| --- | --- |
| `reward` | Main scalar reward: submitted sequence matches any passing candidate |
| `submitted_sequence_matches_passing_candidate` | Final answer matches one of the passing candidate sequences |
| `submitted_sequence_matches_best_passing_candidate` | Final answer matches the best passing candidate exactly |
| `pipeline_completed_metric` | All five stages ran in order |
| `passing_candidate_available_metric` | At least one passing candidate existed |
| `num_passing_candidates_metric` | Number of passing candidates from `summarize_candidates()` |
| `best_passing_score_metric` | Monomer plausibility score of the best passing candidate |
| `target_mean_plddt_metric` | Target monomer quality |
| `stage_error_metric` | Count of stage-order or execution errors |

## GPU sandbox images
This directory includes two image tracks:

1. root `Dockerfile`
   - Blackwell-oriented WIP image validated on the remote RTX 6000 host
2. `experimental_h200_sandbox/`
   - experimental H200-targeted bundle for future GPU sandbox work
   - uses a stable CUDA 12.4 / PyTorch 2.4 stack instead of the Blackwell nightly stack
   - includes entrypoint and smoke-test scripts plus a README describing the remaining sandbox requirements

The current environment does **not** require either image yet. Today it still executes over SSH on the remote RTX 6000 host.
