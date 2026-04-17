FROM ghcr.io/sokrypton/colabfold:1.6.0-cuda12

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PROTEIN_RUNTIME=/opt/protein-runtime \
    COLABFOLD_CACHE=/cache/colabfold \
    HF_HOME=/cache/huggingface \
    XDG_CACHE_HOME=/cache/xdg \
    RFDIFFUSION_MODELS=/models/rfdiffusion

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    python3-venv \
    rsync \
    wget \
 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ${PROTEIN_RUNTIME} ${COLABFOLD_CACHE} ${HF_HOME} ${XDG_CACHE_HOME} ${RFDIFFUSION_MODELS} /workspace
WORKDIR ${PROTEIN_RUNTIME}

# H200/Hopper should work with a stable cu124 stack, so this image avoids the
# Blackwell-only nightly torch workaround used by the RTX 6000 path.
RUN git clone https://github.com/RosettaCommons/RFdiffusion.git RFdiffusion-src \
 && git clone https://github.com/dauparas/ProteinMPNN.git ProteinMPNN-src

RUN python3 -m venv ${PROTEIN_RUNTIME}/rfdiffusion-h200-venv \
 && source ${PROTEIN_RUNTIME}/rfdiffusion-h200-venv/bin/activate \
 && python -m pip install --upgrade pip setuptools wheel \
 && pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
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

RUN python3 -m venv ${PROTEIN_RUNTIME}/proteinmpnn-h200-venv \
 && source ${PROTEIN_RUNTIME}/proteinmpnn-h200-venv/bin/activate \
 && python -m pip install --upgrade pip setuptools wheel \
 && pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124 \
 && pip install numpy==2.2.6

COPY protein_binder_monomer_real/support ${PROTEIN_RUNTIME}/support
COPY experimental_h200_sandbox/sandbox_entrypoint.sh /usr/local/bin/protein-sandbox-entrypoint
COPY experimental_h200_sandbox/smoke_test.sh /usr/local/bin/protein-sandbox-smoke-test

RUN chmod +x /usr/local/bin/protein-sandbox-entrypoint /usr/local/bin/protein-sandbox-smoke-test

WORKDIR /workspace
ENTRYPOINT ["protein-sandbox-entrypoint"]
CMD ["bash"]
