#!/usr/bin/env python3
"""Run RFdiffusion with a PyTorch compatibility patch for Blackwell hosts.

RFdiffusion still expects torch.load(..., weights_only=False) semantics from older
PyTorch versions. Newer PyTorch defaults changed, so this wrapper restores the
older behavior before delegating to RFdiffusion's inference entrypoint.
"""

from __future__ import annotations

import runpy
import sys

import torch

_ORIGINAL_TORCH_LOAD = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


def main() -> None:
    torch.load = _patched_torch_load
    sys.argv = ["scripts/run_inference.py", *sys.argv[1:]]
    runpy.run_path("scripts/run_inference.py", run_name="__main__")


if __name__ == "__main__":
    main()
