---
name: pi-replay
description: Prepare Prime eval and RL runs for later inspection in pi-replay. Use when an environment change should preserve rollout outputs, reward decomposition, and import paths for replay/debugging.
---

# pi-replay

Use this skill when working on environments that should later be inspectable with the local `../pi-replay` project.

## Goal
Keep environment iteration compatible with `prime eval` today and `pi-replay` imports later, without coupling the environment to a speculative replay schema.

## Current local project
`../pi-replay` currently supports:
- `prime-replay import-verifiers`
- `prime-replay import-prime-rl`
- `prime-replay import-latest-eval`
- static run viewers, compare pages, and fork bundles

It is intentionally an adapter/storage layer, not an environment standard.

## Rules
1. Prefer `prime eval run` as the canonical eval path.
2. Preserve reward decomposition as rubric terms or metrics instead of collapsing everything into one scalar.
3. Save results for meaningful evals with `-s` so imports have a stable source.
4. When useful, save extra state columns that help replay/debugging.
5. Do not redesign the environment around a replay format; keep replay compatibility additive.

## Good environment patterns for replay readiness
1. Keep final reward broken into explicit rubric funcs and metrics.
2. Keep tool outputs structured and compact JSON when possible.
3. Store key episode bookkeeping in state if it helps debug decisions:
   - budget remaining
   - chosen candidate id
   - best discovered score
   - tool call counts
4. Prefer stable candidate/task identifiers over free-form prose references.
5. Keep artifact references deterministic when files are produced.

## Local workflow
Run evals normally first:
```bash
prime eval run my-env -m openai/gpt-4.1-mini -n 20 -s
```

Then import later from the pi-replay repo:
```bash
cd ../pi-replay
prime-replay import-latest-eval --search-root ../protein-binder --output-root runs
prime-replay browse --runs-root runs
```

## What not to do yet
- Do not block environment progress on a full trajectory schema.
- Do not patch upstream verifiers/prime-rl unless the user explicitly asks.
- Do not invent snapshot semantics unless the environment genuinely needs them.

## Deliverable mindset
For each meaningful milestone, leave behind:
1. a runnable environment,
2. a saved eval result,
3. clear reward decomposition,
4. enough metadata for later import into pi-replay.
