from __future__ import annotations

import json
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO

import numpy as np
import torch
import yaml


class Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, content: str) -> int:
        for stream in self.streams:
            stream.write(content)
        return len(content)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(
            getattr(stream, "isatty", lambda: False)() for stream in self.streams
        )


@contextmanager
def logged_output(path: Path) -> Iterator[None]:
    stdout, stderr = sys.stdout, sys.stderr
    with path.open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout, sys.stderr = Tee(stdout, log), Tee(stderr, log)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout, sys.stderr = stdout, stderr


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"YAML 顶层必须是映射：{path}")
    return config


def save_yaml_once(path: Path, content: dict) -> None:
    if path.exists():
        return
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(content, file, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，自动改用 CPU", flush=True)
        return torch.device("cpu")
    return torch.device(name)


def move_batch(
    batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return tuple(
        batch[key].to(device, non_blocking=non_blocking)
        for key in ("features", "segment_mask", "targets")
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr: float,
):
    if total_steps <= 0 or not 0 <= warmup_ratio < 1:
        raise ValueError("total_steps 必须大于 0，warmup_ratio 必须位于 [0,1)")
    base_lr, warmup_steps = optimizer.param_groups[0]["lr"], int(
        total_steps * warmup_ratio
    )
    min_ratio = min(1.0, min_lr / base_lr)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0e-8)
        progress = min(
            1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        )
        return min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * progress)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def scalar_metrics(metrics: dict) -> dict:
    result = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (int, np.integer)):
            result[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            value = float(value)
            result[key] = value if math.isfinite(value) else None
    return result


def atomic_json_save(path: Path, content: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(content, file, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def gpu_memory(device: torch.device) -> str:
    return (
        f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f}GB"
        if device.type == "cuda"
        else "CPU"
    )


def nonfinite_batch_message(
    batch: dict, predictions: torch.Tensor, targets: torch.Tensor
) -> str:
    problems = []
    if not torch.isfinite(predictions).all().item():
        problems.append("predictions 含 NaN/Inf")
    if not torch.isfinite(targets).all().item():
        problems.append("targets 含 NaN/Inf")
    return f"歌曲 IDs={batch.get('ids', [])}; " + (
        "；".join(problems) if problems else "输入本身为有限值"
    )


def nonfinite_gradient_names(model: torch.nn.Module, limit: int = 8) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ][:limit]
