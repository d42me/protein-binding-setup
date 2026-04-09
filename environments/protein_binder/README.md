# protein-binder

Synthetic peptide-binder design environment for bootstrapping protein-binding RL workflows.

## What this environment is
`protein-binder` is an intentionally lightweight proxy task: the model receives a synthetic target-pocket specification and must propose a short peptide sequence that should bind well under a handcrafted biochemical heuristic.

This is **not** a docking environment and should not be interpreted as a physically valid protein-design benchmark. It is a fast iteration scaffold for:
- shaping a sequence-design interface,
- validating tool-use behavior,
- testing reward decomposition,
- and debugging RL/eval pipelines before adding more realistic biology.

## Task contract
- **Type**: tool-use, single-agent, short-horizon
- **Action**: output one peptide sequence in `<sequence>...</sequence>`
- **Observation**: synthetic target description with position-wise residue-class preferences, approximate composition targets, charge target, and residue constraints
- **Tools**:
  - `amino_acid_reference(residue)`
  - `sequence_profile(sequence)`
  - `compare_to_target(sequence, target_spec_json)`

## Reward
The main reward is a weighted proxy composed of:
- position-wise class matching,
- composition matching,
- charge matching,
- anchor-position quality,
- simple stability/constraint checks,
- and a small XML-format bonus.

Metrics are also logged for:
- `valid_sequence_metric`
- `position_metric`
- `composition_metric`
- `charge_metric`
- `anchor_metric`
- `reference_similarity_metric`

## Synthetic dataset method
Each example is generated procedurally:
1. Sample a target-pocket template with peptide length, position preferences, anchors, and forbidden residues.
2. Search for a high-scoring latent reference sequence by repeated constrained sampling plus local mutation.
3. Derive visible prompt targets from that latent sequence: class counts, target charge, and anchor positions.
4. Hide the exact reference sequence from the model and use it only as a metric for inspection.

This gives us a small, reproducible dataset with:
- multiple valid solutions,
- continuous reward instead of brittle exact match,
- and meaningful tool affordances.

## Quickstart
Install locally:

```bash
prime env install protein-binder
```

Quick smoke eval:

```bash
prime eval run protein-binder -m openai/gpt-4.1-mini -n 5
```

Saved-results eval:

```bash
prime eval run protein-binder -m openai/gpt-4.1-mini -n 12 -s
```

## Environment arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `96` | Number of synthetic training examples |
| `num_eval_examples` | int | `24` | Number of synthetic eval examples |
| `train_seed` | int | `7` | Train split RNG seed |
| `eval_seed` | int | `17` | Eval split RNG seed |
| `max_turns` | int | `4` | Maximum assistant turns including tool-use turns |

## Recommended next steps
1. Replace the proxy with a stronger surrogate scorer (e.g. sequence LM + structure-aware classifier).
2. Add held-out motif families and harder OOD splits.
3. Introduce pairwise ranking tasks or mutation trajectories.
4. Eventually move to a stateful environment with external structure/scoring tools.
