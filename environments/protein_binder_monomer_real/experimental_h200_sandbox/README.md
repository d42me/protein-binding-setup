# experimental_h200_sandbox

Experimental H200-oriented GPU sandbox bundle for `protein-binder-monomer-real`.

## What this is
This folder packages the extra pieces needed to try the real monomer pipeline inside a **GPU sandbox image** instead of the current SSH-backed RTX 6000 workflow.

It is intentionally separate from the main environment runtime because the current production path is still:
- local verifiers env
- tool calls over SSH
- remote run directory on `ubuntu@154.54.100.216`

## Why a separate H200 bundle
The root `Dockerfile` in this environment was shaped around the Blackwell host and therefore uses nightly `cu128` PyTorch.

For H200/Hopper sandboxes we should prefer a simpler, more reproducible stack:
- stable CUDA 12.4 wheels
- stable PyTorch 2.4.0
- matching DGL 2.4.0+cu124
- the same packaged monomer harness support scripts

That is exactly what this folder contains.

## Files
- `Dockerfile`
  - H200-targeted image based on `ghcr.io/sokrypton/colabfold:1.6.0-cuda12`
  - installs RFdiffusion and ProteinMPNN source trees
  - creates stable `rfdiffusion-h200-venv` and `proteinmpnn-h200-venv`
- `sandbox_entrypoint.sh`
  - prepares writable cache/model/workspace paths
  - warns when RFdiffusion weights are not mounted
- `smoke_test.sh`
  - verifies GPU visibility plus RFdiffusion / ProteinMPNN imports

## What a GPU sandbox still needs beyond the image
Containerizing the toolchain is necessary but not sufficient. A real sandbox rollout path also needs:

1. **GPU resource request**
   - the sandbox runtime must request at least `1x H200`
   - the environment should expose that requirement explicitly when we convert from SSH to sandbox-native execution

2. **Model weights mount**
   - RFdiffusion weights are still expected at:
     - `${RFDIFFUSION_MODELS}`
   - current default in this image:
     - `/models/rfdiffusion`

3. **Writable cache volumes**
   - ColabFold params / caches should land in a writable persistent path:
     - `${COLABFOLD_CACHE}` → `/cache/colabfold`
   - HuggingFace / XDG caches should also be writable:
     - `${HF_HOME}` → `/cache/huggingface`
     - `${XDG_CACHE_HOME}` → `/cache/xdg`

4. **Sandbox-native run contract**
   - today `protein_binder_monomer_real.py` assumes SSH and a remote host
   - to use GPU sandboxes, we still need a second environment path that replaces:
     - `remote_host`
     - `remote_support_dir`
     - `remote_run_root`
   - with sandbox-local equivalents such as:
     - `docker_image`
     - sandbox workspace path
     - mounted model/cache roots

5. **MSA/network policy decision**
   - if target monomer runs with `mmseqs2_uniref_env`, the sandbox needs outbound access for the ColabFold/MMseqs path
   - if outbound access is not allowed, we need either:
     - precomputed MSAs, or
     - a local database strategy

6. **Image publication**
   - after validation, the image should be pushed as a Prime image and referenced by the sandbox-native env

## Build locally from the environment root
From `environments/protein_binder_monomer_real/`:

```bash
docker build -f experimental_h200_sandbox/Dockerfile -t protein-binder-monomer-real-h200:wip .
```

## Example runtime contract
```bash
docker run --rm --gpus all \
  -e COLABFOLD_CACHE=/cache/colabfold \
  -e HF_HOME=/cache/huggingface \
  -e XDG_CACHE_HOME=/cache/xdg \
  -e RFDIFFUSION_MODELS=/models/rfdiffusion \
  -v /host/colabfold-cache:/cache/colabfold \
  -v /host/hf-cache:/cache/huggingface \
  -v /host/xdg-cache:/cache/xdg \
  -v /host/rfdiffusion-models:/models/rfdiffusion \
  protein-binder-monomer-real-h200:wip \
  protein-sandbox-smoke-test
```

## Recommended next step
If we want to actually execute this on Prime GPU sandboxes, the next implementation should be a **sandbox-native sibling environment** that keeps the same five staged tools but swaps the SSH orchestration layer for a sandbox workspace runner.
