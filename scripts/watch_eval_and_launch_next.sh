#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <eval_id> <next_eval_name> [task_library] [num_examples]" >&2
  exit 2
fi

EVAL_ID="$1"
NEXT_EVAL_NAME="$2"
TASK_LIBRARY="${3:-ronig}"
NUM_EXAMPLES="${4:-2}"

cd /Users/dominikscherm/Documents/PI/Dev/protein-binder
source .venv/bin/activate

while true; do
  STATUS="$({ prime eval get "$EVAL_ID" --plain 2>/dev/null || true; } | rg '"status"' | head -n1 | sed -E 's/.*"status": "([A-Z_]+)".*/\1/')"
  echo "$(date -u +%FT%TZ) status=${STATUS:-UNKNOWN} eval=$EVAL_ID"

  if [ "$STATUS" = "COMPLETED" ]; then
    prime eval run d42me/protein-binder-monomer-real@0.2.0 \
      -m Qwen/Qwen3-30B-A3B-Instruct-2507 \
      -n "$NUM_EXAMPLES" -r 1 -c 1 \
      --hosted --timeout-minutes 180 \
      --custom-secrets "$(python - <<'PYJSON'
import json
import os

required = {
    'PROTEIN_BINDER_API_TOKEN': os.environ.get('PROTEIN_BINDER_API_TOKEN'),
    'PROTEIN_BINDER_REMOTE_API_BASE_URL': os.environ.get('PROTEIN_BINDER_REMOTE_API_BASE_URL'),
}
missing = [name for name, value in required.items() if not value]
if missing:
    raise SystemExit('Missing required environment variables: ' + ', '.join(missing))
print(json.dumps({
    **required,
    'PROTEIN_BINDER_TASK_LIBRARY': os.environ.get('PROTEIN_BINDER_TASK_LIBRARY', '$TASK_LIBRARY'),
    'PROTEIN_BINDER_KEEP_REMOTE_ARTIFACTS': os.environ.get('PROTEIN_BINDER_KEEP_REMOTE_ARTIFACTS', 'true'),
}))
PYJSON
      )" \
      --eval-name "$NEXT_EVAL_NAME" -A
    exit 0
  fi

  if [ "$STATUS" = "FAILED" ] || [ "$STATUS" = "CANCELLED" ]; then
    echo "terminal_status=$STATUS"
    exit 1
  fi

  sleep 120
done
