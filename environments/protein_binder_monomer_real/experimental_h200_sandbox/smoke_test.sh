#!/usr/bin/env bash
set -euo pipefail

python3 --version
nvidia-smi

/opt/protein-runtime/rfdiffusion-h200-venv/bin/python - <<'PY'
import torch, dgl, rfdiffusion
print("rfdiffusion torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("rfdiffusion import ok")
PY

/opt/protein-runtime/proteinmpnn-h200-venv/bin/python - <<'PY'
import torch
print("proteinmpnn torch", torch.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("proteinmpnn import ok")
PY

python3 /opt/protein-runtime/support/run_monomer_pipeline.py --help >/dev/null

echo "protein sandbox smoke test passed"
