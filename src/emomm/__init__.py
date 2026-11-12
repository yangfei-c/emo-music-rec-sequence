"""Shared paths and small I/O helpers for EmoMM-MusicRec."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "emomm_musicrec_fresh_v1"


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def run_id(value="auto"):
    if value == "auto":
        value = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value):
        raise ValueError("ID 只能包含字母、数字、下划线和连字符。")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_npz(path, **arrays):
    import numpy as np
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def require_device(name):
    import torch
    if name not in {"cpu", "cuda"}:
        raise ValueError("device 必须是 cpu 或 cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请使用 msmmr 环境，或明确将 device 改为 cpu。")
    return torch.device(name)
