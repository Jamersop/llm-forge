"""Detect what this machine can train on, and which remote backends are configured."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

# GPU types offered for remote (Modal) jobs, with memory for the UI's fit hints.
MODAL_GPUS = [
    {"id": "T4", "memory_gb": 16}, {"id": "L4", "memory_gb": 24}, {"id": "A10G", "memory_gb": 24},
    {"id": "L40S", "memory_gb": 48}, {"id": "A100-40GB", "memory_gb": 40},
    {"id": "A100-80GB", "memory_gb": 80}, {"id": "H100", "memory_gb": 80},
]


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str | None = None):
    import torch
    device = device or pick_device()
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32  # MPS mixed-precision training is not reliable yet; CPU needs fp32


def local_info() -> dict:
    import torch
    device = pick_device()
    info = {"device": device, "name": "CPU", "memory_gb": None,
            "methods": ["lora", "full"] + (["qlora"] if device == "cuda" else [])}
    if device == "cuda":
        p = torch.cuda.get_device_properties(0)
        info.update(name=p.name, memory_gb=round(p.total_memory / 2**30, 1),
                    count=torch.cuda.device_count())
    elif device == "mps":
        info["name"] = "Apple Silicon GPU (MPS)"
    return info


def modal_info() -> dict:
    if importlib.util.find_spec("modal") is None:
        return {"available": False, "reason": "pip install modal, then `modal setup`"}
    has_token = bool(os.environ.get("MODAL_TOKEN_ID")) or (Path.home() / ".modal.toml").exists()
    if not has_token:
        return {"available": False, "reason": "run `modal setup` to log in"}
    return {"available": True, "gpus": MODAL_GPUS,
            "note": "Deploy once with `modal deploy forge/remote_modal.py`"}


def compute_info() -> dict:
    return {"local": local_info(), "modal": modal_info()}
