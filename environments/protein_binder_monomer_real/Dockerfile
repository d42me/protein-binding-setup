FROM ghcr.io/sokrypton/colabfold:1.6.0-cuda12

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTEIN_RUNTIME=/opt/protein-runtime \
    COLABFOLD_CACHE=/opt/protein-runtime/colabfold-cache \
    RFDIFFUSION_MODELS=/opt/protein-runtime/rfdiffusion-models

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3-venv \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${PROTEIN_RUNTIME} ${COLABFOLD_CACHE} ${RFDIFFUSION_MODELS} /workspace
WORKDIR ${PROTEIN_RUNTIME}

# Real tool sources used by the current Blackwell workflow.
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git RFdiffusion-src \
 && git clone https://github.com/dauparas/ProteinMPNN.git ProteinMPNN-src

# RFdiffusion Blackwell-compatible environment.
RUN python3 -m venv ${PROTEIN_RUNTIME}/rfdiffusion-bw-venv \
 && source ${PROTEIN_RUNTIME}/rfdiffusion-bw-venv/bin/activate \
 && python -m pip install --upgrade pip setuptools wheel \
 && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
 && pip install 'dgl==2.4.0+cu124' -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html --no-deps \
 && pip install \
      numpy==2.2.6 \
      scipy==1.15.3 \
      pandas==2.3.3 \
      hydra-core==1.3.2 \
      omegaconf==2.3.0 \
      e3nn==0.3.3 \
      pyrsistent==0.20.0 \
      pynvml==11.0.0 \
      psutil==7.2.2 \
      pydantic==2.12.5 \
      requests==2.33.1 \
      tqdm==4.67.3 \
 && pip install ${PROTEIN_RUNTIME}/RFdiffusion-src/env/SE3Transformer \
 && pip install ${PROTEIN_RUNTIME}/RFdiffusion-src

# ProteinMPNN Blackwell-compatible environment.
RUN python3 -m venv ${PROTEIN_RUNTIME}/proteinmpnn-bw-venv \
 && source ${PROTEIN_RUNTIME}/proteinmpnn-bw-venv/bin/activate \
 && python -m pip install --upgrade pip setuptools wheel \
 && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 \
 && pip install numpy==2.2.6

# Packaged harness support scripts. These are the same tools the rollout env syncs to the remote RTX 6000 host.
COPY protein_binder_monomer_real/support ${PROTEIN_RUNTIME}/support

WORKDIR /workspace

# WIP image notes:
# - ColabFold is inherited from the base image.
# - RFdiffusion model weights are expected to be mounted into ${RFDIFFUSION_MODELS}.
# - This image targets a future GPU sandbox path; the current environment still uses direct SSH access to the RTX 6000 host.
CMD ["bash"]
