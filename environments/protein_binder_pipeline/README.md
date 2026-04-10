# protein-binder-pipeline

Linear sandbox-backed protein binder pipeline scaffold built around the exact node order:

`Target protein sequence -> AlphaFold2 -> Target structure PDB -> RFdiffusion -> Binder backbone structure -> ProteinMPNN -> Binder sequence -> AlphaFold-Multimer -> Reward`

## Overview
- **Environment ID**: `protein-binder-pipeline`
- **Type**: `SandboxEnv` with high-level structured tools
- **Goal**: pick one RFdiffusion backbone mode and one ProteinMPNN sampling temperature, run the full pipeline in order, and submit the generated binder sequence
- **Reward**: `1` when the final AlphaFold-Multimer `structural_plausibility` is strictly greater than the task threshold, else `0`

## Why this environment exists
This is the smallest sandbox-native version of the original linear protein-design diagram.
It keeps the tool chain explicit while making the task cheap and deterministic enough for repeated evals and later RL.

## Episode contract
Each task exposes:
- a target protein sequence
- a structural plausibility threshold
- the required stage order
- two agent-controlled choices:
  - `RFdiffusion.design_mode`: `helix | beta | balanced`
  - `ProteinMPNN.sampling_temperature`: `low | medium | high`

The environment creates a per-rollout Prime sandbox workspace and materializes pipeline artifacts there:
- `target_sequence.fasta`
- `target_structure.pdb`
- `binder_backbone.pdb`
- `binder_sequence.fasta`
- `multimer_report.json`

The agent should:
1. run `AlphaFold2`
2. inspect the returned structure summary
3. choose an RFdiffusion mode
4. choose a ProteinMPNN sampling temperature
5. run `AlphaFold-Multimer`
6. submit the generated binder sequence inside `<sequence>...</sequence>`

## Tools
- `run_alphafold2()`
  - produces the target structure PDB and a structure summary
- `run_rfdiffusion(design_mode)`
  - produces the binder backbone PDB
- `run_proteinmpnn(sampling_temperature)`
  - produces the binder sequence FASTA
- `run_alphafold_multimer()`
  - returns structural plausibility, threshold, and pass/fail

Stages are enforced left to right. Out-of-order calls return explicit JSON errors.

## Metrics
| Metric | Meaning |
| --- | --- |
| `reward` | Main scalar reward (`1` if plausibility > threshold and submitted sequence matches) |
| `submitted_sequence_matches_candidate` | Final `<sequence>` matches the generated binder sequence |
| `pipeline_completed_metric` | All four stages were run in order |
| `structural_plausibility_metric` | Final AlphaFold-Multimer plausibility score |
| `threshold_metric` | Per-task plausibility threshold |
| `pass_margin_metric` | `plausibility - threshold` |
| `passes_threshold_metric` | Binary pass/fail outcome before answer formatting |
| `used_optimal_backbone_metric` | Whether the chosen RFdiffusion mode matched the hidden target preference |
| `used_optimal_sampling_metric` | Whether the chosen ProteinMPNN temperature matched the hidden target preference |
| `stage_error_metric` | Count of stage-order or execution errors |

## Quickstart
Install locally:

```bash
prime env install protein-binder-pipeline
```

Smoke eval:

```bash
prime eval run protein-binder-pipeline -m gpt-4.1-mini -n 5
```

Config-driven eval:

```bash
prime eval run configs/eval/protein-binder-pipeline-baseline.toml -A
```

## Prime sandboxes and images
By default the environment uses the stock `python:3.11-slim` sandbox image and copies the synthetic runner into `/workspace` during setup.

A ready-to-build `Dockerfile` is included so the runner can be preloaded into a Prime Image for faster startup:

```bash
prime images push protein-binder-pipeline:v0.1.0 --context ./environments/protein_binder_pipeline
```

Then evaluate with the prebuilt image:

```bash
prime eval run protein-binder-pipeline \
  -m openai/gpt-4.1-mini \
  -x '{"docker_image":"cmdyatgoq008xao8o5m8bkh20/protein-binder-pipeline:v0.1.0"}'
```

## Environment arguments
| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `num_train_examples` | int | `96` | Number of synthetic train tasks |
| `num_eval_examples` | int | `24` | Number of synthetic eval tasks |
| `train_seed` | int | `11` | Train RNG seed |
| `eval_seed` | int | `23` | Eval RNG seed |
| `max_turns` | int | `6` | Maximum assistant turns |
| `docker_image` | str | `"python:3.11-slim"` | Sandbox image used for each rollout |
