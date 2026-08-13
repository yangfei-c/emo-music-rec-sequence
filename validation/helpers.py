"""Shared assertions and device selection for validation scripts."""

import torch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"[FAIL] {message}")


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，改用 CPU")
        return torch.device("cpu")
    return torch.device(name)
