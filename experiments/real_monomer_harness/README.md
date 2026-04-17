# real-monomer-harness

A real-tool protein-binder harness that keeps the pipeline **monomer-only**:

1. run **ColabFold monomer** on the target sequence
2. run **RFdiffusion** against the target monomer structure to generate binder backbones
3. run **ProteinMPNN** to design binder sequences for those backbones
4. run **ColabFold monomer** on each designed binder sequence
5. rank candidates with a **monomer plausibility heuristic** and a stricter **quality gate**

## Why this harness exists
The current Blackwell machine can run real AlphaFold-monomer/ColabFold, RFdiffusion, and ProteinMPNN, but the project direction is now explicitly **monomer-only**.

That means the harness should optimize for what can be measured without a multimer scorer:
- target monomer structure generation
- interface geometry proposed by RFdiffusion
- binder sequence generation via ProteinMPNN
- binder foldability as a standalone monomer
- agreement between the RFdiffusion binder backbone and the monomer-predicted binder structure
- hotspot/interface proxy retention from the RFdiffusion design

## What it scores
For each candidate binder sequence, the harness records:
- `binder_mean_plddt` from ColabFold monomer
- `binder_ptm` from ColabFold monomer
- `binder_distance_rmse`
  - RMSE between the pairwise CA distance maps of:
    - the RFdiffusion binder backbone chain, and
    - the ColabFold monomer prediction for the designed binder sequence
- `interface_residue_contacts`
  - how many binder residues contact the target chain in the RFdiffusion complex backbone
- `hotspot_contacts`
  - how many requested hotspot residues are contacted in that RFdiffusion complex backbone
  - current proxy uses a slightly relaxed `6Å` any-atom cutoff so the monomer-only screen is less brittle to small backbone shifts
- `monomer_plausibility_score`
  - a heuristic ranking score that combines monomer confidence, backbone agreement, hotspot coverage, and interface-contact density
- `passes_quality_gate`
  - whether the candidate clears a stricter monomer-only threshold contract

This is **not a binding affinity score**. It is a monomer-only filter for selecting candidates that look internally consistent before any later complex-aware step.

## Quality gate defaults
A candidate passes only if it clears all of these defaults:
- target mean pLDDT >= `80`
- binder mean pLDDT >= `80`
- binder distance RMSE <= `1.5`
- hotspot fraction >= `0.33`
- interface residue contacts >= `10`
- monomer plausibility score >= `0.72`

These thresholds are intentionally stricter than the original smoke-test heuristic so that foldable-but-interface-weak candidates do not rank as successes.

## Tool-ready stage model
The runner is now designed to support **tool-style rollout calls**.

It supports both:
- a single end-to-end `pipeline` command, and
- individual stage commands that operate on a persistent `run-dir`

Stage commands:
- `init-run`
- `target-monomer`
- `rfdiffusion`
- `proteinmpnn`
- `binder-monomer`
- `summarize`

That means a future environment can call each stage as a separate tool while keeping artifacts isolated per rollout directory.

## Parallelization hooks
Heavy stages still default to **safe single-GPU execution**, but the harness now exposes a batching interface for binder scoring:
- `--candidate-batch-size`
- `--max-concurrent-binder-batches`

By default:
- candidates are grouped into batches of `16`
- only `1` binder batch runs at a time

This keeps current Blackwell runs safe while making it straightforward to scale out later across more GPUs or rollout workers.

## Files
- `run_monomer_pipeline.py`
  - main orchestration CLI and stage commands
- `run_rfdiffusion_blackwell.py`
  - wrapper that restores `torch.load(..., weights_only=False)` behavior needed by RFdiffusion under newer PyTorch
- `scout_ronig_dataset.py`
  - scouts `ronig/protein_binding_sequences` for monomer-harness-compatible target/peptide pairs
  - reports staged filter counts, length-window tradeoffs, and an optional structure-backed hotspot viability sample
- `curate_ronig_dataset.py`
  - turns a conservative-but-broader ronig subset into bundled environment tasks with sequence-backed targets and sequential hotspot labels
  - applies a second-pass structural filter, derives top-contact hotspots, and now allows up to two tasks per exact receptor sequence and up to two tasks per exact peptide sequence so the bundled library can reach useful scale without collapsing diversity

## Dataset scouting example
Use this before wiring a larger public dataset into the environment:

```bash
python experiments/real_monomer_harness/scout_ronig_dataset.py \
  --min-peptide-length 30 \
  --max-peptide-length 50 \
  --structure-sample-size 20 \
  --output-jsonl ./artifacts/dataset_scout/ronig_30_50_candidates.jsonl
```

This gives a conservative candidate manifest for the current monomer harness and estimates how often filtered examples still expose at least a few receptor interface residues that can be turned into sequential hotspot prompts.

To materialize the bundled ronig task library used by `protein-binder-monomer-real`:

```bash
python experiments/real_monomer_harness/curate_ronig_dataset.py \
  --max-tasks 100 \
  --selection-seed 7 \
  --output ./environments/protein_binder_monomer_real/protein_binder_monomer_real/data/ronig_curated_tasks.json
```

## Remote assumptions
Default paths assume the Blackwell host has already been prepared with:
- `~/protein-runtime/colabfold-cache`
- `~/protein-runtime/RFdiffusion-src`
- `~/protein-runtime/rfdiffusion-bw-venv/bin/python`
- `~/protein-runtime/rfdiffusion-models`
- `~/protein-runtime/ProteinMPNN-src`
- `~/protein-runtime/proteinmpnn-test-venv/bin/python`

## End-to-end example
The old invocation style still works and is treated as the `pipeline` command.

```bash
python experiments/real_monomer_harness/run_monomer_pipeline.py \
  --run-dir ./artifacts/real_monomer_harness/insulin_run \
  --target-pdb ~/protein-runtime/work/input_pdbs/insulin_target.pdb \
  --target-chain A \
  --hotspots A59,A83,A91 \
  --binder-length-min 40 \
  --binder-length-max 50 \
  --num-designs 8 \
  --num-seqs-per-backbone 8 \
  --candidate-batch-size 16 \
  --overwrite
```

## Stage-by-stage example
```bash
python experiments/real_monomer_harness/run_monomer_pipeline.py init-run \
  --run-dir ./artifacts/real_monomer_harness/insulin_tools \
  --target-pdb ~/protein-runtime/work/input_pdbs/insulin_target.pdb \
  --target-chain A \
  --hotspots A59,A83,A91 \
  --binder-length-min 40 \
  --binder-length-max 50 \
  --num-designs 8 \
  --num-seqs-per-backbone 8 \
  --overwrite

python experiments/real_monomer_harness/run_monomer_pipeline.py target-monomer --run-dir ./artifacts/real_monomer_harness/insulin_tools
python experiments/real_monomer_harness/run_monomer_pipeline.py rfdiffusion --run-dir ./artifacts/real_monomer_harness/insulin_tools
python experiments/real_monomer_harness/run_monomer_pipeline.py proteinmpnn --run-dir ./artifacts/real_monomer_harness/insulin_tools
python experiments/real_monomer_harness/run_monomer_pipeline.py binder-monomer --run-dir ./artifacts/real_monomer_harness/insulin_tools
python experiments/real_monomer_harness/run_monomer_pipeline.py summarize --run-dir ./artifacts/real_monomer_harness/insulin_tools
```

## Outputs
The run directory contains:
- `inputs/` — target FASTA plus binder candidate batch FASTAs
- `target_monomer/` — target ColabFold monomer outputs
- `rfdiffusion/` — RFdiffusion backbones and `.trb` files
- `proteinmpnn/` — ProteinMPNN FASTA outputs per backbone
- `binder_monomer/` — ColabFold monomer outputs for designed binders, grouped by batch
- `state/` — persistent stage metadata (`run_config.json`, `backbones.json`, `candidates.json`, etc.)
- `summary/run_summary.json` — full structured summary
- `summary/ranked_candidates.csv` — flat ranking table

## Current verdict
This harness is a good **real-tool experimentation scaffold**, but it is still a **screening harness**, not a final binder evaluator.

Use it to:
- validate the real pipeline end to end
- sweep backbone lengths / hotspots / ProteinMPNN samples
- filter for foldable binders that preserve the designed backbone geometry
- drive tool-style rollout stages off a persistent artifact directory

Do **not** confuse the monomer plausibility score with true binding quality.
