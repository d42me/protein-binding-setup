#!/usr/bin/env bash
set -euo pipefail

: "${PROTEIN_RUNTIME:=/opt/protein-runtime}"
: "${COLABFOLD_CACHE:=/cache/colabfold}"
: "${HF_HOME:=/cache/huggingface}"
: "${XDG_CACHE_HOME:=/cache/xdg}"
: "${RFDIFFUSION_MODELS:=/models/rfdiffusion}"

mkdir -p \
  "${PROTEIN_RUNTIME}" \
  "${COLABFOLD_CACHE}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${RFDIFFUSION_MODELS}" \
  /workspace

if ! find "${RFDIFFUSION_MODELS}" -mindepth 1 -maxdepth 1 -print -quit >/dev/null 2>&1; then
  echo "[protein-sandbox-entrypoint] warning: ${RFDIFFUSION_MODELS} is empty; mount RFdiffusion model weights before running design stages." >&2
fi

exec "$@"
