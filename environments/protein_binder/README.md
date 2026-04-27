# protein-binder

Budgeted peptide-binder redesign environment for the first realistic protein-design loop.

## Scope
This environment implements **scope 0.5** of the broader protein-binder roadmap:
- start from a weak seed binder,
- redesign under a fixed compute budget,
- use stateful tools to create and screen variants,
- submit the best screened candidate ID and exact sequence.

It is intentionally earlier-stage than a full de novo pipeline. The goal is to train and evaluate:
- strategic search,
- budget-aware tool use,
- candidate tracking,
- and shortlist selection.

## Environment type
- **Type**: `StatefulToolEnv`
- **Horizon**: multi-turn, longer budgeted search
- **Final answer**: `<answer>C0003</answer><sequence>ACDEFGHIK</sequence>`

## Episode design
Each task contains:
- a synthetic target-pocket specification,
- one weak seed candidate `C0000`,
- a fixed redesign budget of `18.0` units,
- residue-class and charge constraints.

The hidden ground-truth scorer is the same synthetic binder proxy used to create the task, but the agent only interacts through staged tools and noisy screens. The final answer must include both the selected candidate ID and its exact sequence so the externally visible output is biologically meaningful while still preserving search bookkeeping.

## Tools
- `list_candidates()`
  - inspect the current candidate table, remaining budget, and the exact sequence for the recommended submission candidate, while separating quick-screened vs full-screened finalists
- `design_variants(parent_id, strategy, num_variants)`
  - create new variants from an existing candidate
- `quick_screen(candidate_ids)`
  - cheap/noisy scoring for triage
- `full_screen(candidate_ids)`
  - stronger and more expensive scoring with a fuller metric breakdown

Supported design strategies:
- `balanced`
- `anchor`
- `composition`
- `charge`
- `explore`

## Reward
Main reward favors:
- selecting a **screened** candidate,
- high hidden true quality,
- improvement over the initial seed,
- and some budget efficiency.

Additional metrics:
- `chosen_true_score`
- `chosen_improvement`
- `screened_selection_metric`
- `chose_full_screened`
- `submitted_sequence_matches_candidate`
- `budget_efficiency_metric`
- `best_screened_score_metric`
- `candidate_count_metric`
- `design_calls_metric`
- `quick_screen_calls_metric`
- `full_screen_calls_metric`

## Why this is useful
This is the smallest version that already looks like a real design campaign:
- stateful candidate table
- redesign instead of one-shot generation
- multi-fidelity screening
- compute-budget tradeoffs
- shortlist-style decision making

That makes it a better stepping stone toward realistic binder environments than direct sequence emission.

## Quickstart
Install locally:

```bash
prime env install protein-binder
```

Smoke eval:

```bash
prime eval run protein-binder -m openai/gpt-4.1-mini -n 5
```

Config-driven eval:

```bash
prime eval run configs/eval/protein-binder-baseline.toml -A
```

## Environment arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `96` | Number of synthetic train episodes |
| `num_eval_examples` | int | `24` | Number of synthetic eval episodes |
| `train_seed` | int | `7` | Train RNG seed |
| `eval_seed` | int | `17` | Eval RNG seed |
| `max_turns` | int | `14` | Maximum assistant turns |

## Next step after scope 0.5
Move from synthetic redesign to structure-guided redesign with stronger surrogate scoring and held-out target splits.
