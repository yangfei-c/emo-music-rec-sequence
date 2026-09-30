"""Reproducible training records and data-partition checks."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    value = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        value.update(key.encode())
        value.update(str(array.dtype).encode())
        value.update(str(array.shape).encode())
        value.update(array.tobytes())
    return value.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Windows readers without FILE_SHARE_DELETE can briefly block replacement.
    # Keep the previous complete JSON visible and bound retries for real ACL errors.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or attempt == 9:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 0.5))


def environment() -> dict:
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": str(np.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_identity_splits(identities: np.ndarray, splits: np.ndarray, name: str) -> None:
    seen: dict[str, str] = {}
    for identity, split in zip(identities.astype(str), splits.astype(str), strict=True):
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown split in {name}: {split}")
        if identity in seen and seen[identity] != split:
            raise ValueError(f"Recording crosses partitions in {name}: {identity}")
        seen[identity] = split


def training_provenance(model: torch.nn.Module, seed: int, config: dict) -> dict:
    return {
        **environment(), "seed": seed, "config": config,
        "initialization": (
            "fresh_random_clip_style_adapters"
            if config.get("model", {}).get("adapter_kind") == "clip_style"
            else "fresh_random_with_zero_output_residual_adapters"
        ),
        "initial_checkpoint": None, "initial_state_sha256": state_sha256(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "backbone_optimized": False,
    }
