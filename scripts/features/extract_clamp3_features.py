"""Extract full frozen CLaMP3 embeddings from an affect manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.encoding.clamp3 import (
    Clamp3TextAudioEncoder,
    collate_audio_tokens,
    load_audio_tokens,
)
from emomm_sequence.training.utilities import load_yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(config: dict) -> Path:
    value = Path(config["clamp3"]["checkpoint"])
    return value if value.is_absolute() else ROOT / value


def encode_texts(
    texts: list[str],
    model: Clamp3TextAudioEncoder,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    chunks, owners, weights = [], [], []
    for owner, text in enumerate(texts):
        ids = tokenizer(str(text), add_special_tokens=True, truncation=False)["input_ids"]
        if not ids:
            ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
        sample_chunks = [ids[start : start + max_length] for start in range(0, len(ids), max_length)]
        sample_chunks[-1] = ids[-max_length:]
        full, remainder = divmod(len(ids), max_length)
        sample_weights = [max_length] * full + ([remainder] if remainder else [])
        if not sample_weights:
            sample_weights = [len(ids)]
        if len(sample_weights) != len(sample_chunks):
            raise RuntimeError("CLaMP3 text chunk accounting mismatch")
        for chunk, weight in zip(sample_chunks, sample_weights):
            chunks.append(chunk)
            owners.append(owner)
            weights.append(weight)
    totals = torch.zeros((len(texts), 768), dtype=torch.float32)
    denominators = torch.zeros((len(texts), 1), dtype=torch.float32)
    for start in tqdm(range(0, len(chunks), batch_size), desc="CLaMP3 text", unit="batch"):
        batch_chunks = chunks[start : start + batch_size]
        width = max(len(item) for item in batch_chunks)
        input_ids = torch.full(
            (len(batch_chunks), width), tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros((len(batch_chunks), width), dtype=torch.bool)
        for index, item in enumerate(batch_chunks):
            input_ids[index, : len(item)] = torch.tensor(item, dtype=torch.long)
            mask[index, : len(item)] = True
        with torch.inference_mode():
            encoded = model.encode_text(input_ids.to(device), mask.to(device)).float().cpu()
        batch_owners = torch.tensor(owners[start : start + len(batch_chunks)], dtype=torch.long)
        batch_weights = torch.tensor(weights[start : start + len(batch_chunks)], dtype=torch.float32).unsqueeze(1)
        totals.index_add_(0, batch_owners, encoded * batch_weights)
        denominators.index_add_(0, batch_owners, batch_weights)
    return F.normalize(totals / denominators, dim=-1).numpy().astype(np.float32)


def _load_precomputed(path: str) -> np.ndarray | None:
    if not path or not Path(path).is_file():
        return None
    value = np.load(path, allow_pickle=False).reshape(-1, 768).mean(axis=0)
    if not np.isfinite(value).all():
        raise ValueError(f"precomputed embedding contains NaN/Inf: {path}")
    norm = np.linalg.norm(value)
    if norm <= 0:
        raise ValueError(f"zero precomputed embedding: {path}")
    return (value / norm).astype(np.float32)


def encode_audio_rows(
    frame: pd.DataFrame,
    model: Clamp3TextAudioEncoder,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    identities = frame["global_identity"].astype(str).tolist()
    unique_rows = {}
    for index, identity in enumerate(identities):
        unique_rows.setdefault(identity, index)
    cache: dict[str, np.ndarray] = {}
    pending = []
    for identity, index in unique_rows.items():
        precomputed = _load_precomputed(
            str(frame.iloc[index].get("clamp3_embedding_path", ""))
        )
        if precomputed is not None:
            cache[identity] = precomputed
        else:
            pending.append((identity, str(frame.iloc[index]["feature_path"])))
    for start in tqdm(range(0, len(pending), batch_size), desc="CLaMP3 audio", unit="batch"):
        batch_items = pending[start : start + batch_size]
        tokens = [load_audio_tokens(path, max_length=max_length) for _, path in batch_items]
        values, mask = collate_audio_tokens(tokens)
        with torch.inference_mode():
            encoded = model.encode_audio(values.to(device), mask.to(device)).float()
            encoded = F.normalize(encoded, dim=-1).cpu().numpy().astype(np.float32)
        for (identity, _), value in zip(batch_items, encoded):
            cache[identity] = value
    return np.stack([cache[identity] for identity in identities]).astype(np.float32)


def targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    has_va = frame["has_va"].astype(bool).to_numpy()
    values = np.full((len(frame), 2), np.nan, dtype=np.float32)
    if has_va.any():
        values[has_va, 0] = pd.to_numeric(frame.loc[has_va, "valence"]).to_numpy(np.float32)
        values[has_va, 1] = pd.to_numeric(frame.loc[has_va, "arousal"]).to_numpy(np.float32)
    return values, has_va


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--modality", choices=("text", "audio", "pair"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--split", help="exact manifest split value to include")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")
    config = load_yaml(args.config)
    ckpt = checkpoint_path(config)
    expected_hash = str(config["clamp3"]["checkpoint_sha256"]).lower()
    actual_hash = file_sha256(ckpt)
    if actual_hash != expected_hash:
        raise ValueError(f"CLaMP3 checkpoint SHA-256 mismatch: {actual_hash}")
    frame = pd.read_csv(manifest, keep_default_na=False)
    if args.split:
        frame = frame[frame["split"].astype(str) == args.split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"manifest has no rows for split={args.split!r}")
    id_column = "pair_id" if "pair_id" in frame else "sample_id"
    if frame[id_column].duplicated().any():
        raise ValueError("manifest has duplicate sample IDs")
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model = Clamp3TextAudioEncoder(
        ckpt,
        text_model_name=config["clamp3"]["text_model"],
        max_audio_length=int(config["clamp3"]["max_audio_length"]),
        normalize=False,
        freeze=True,
    ).to(device)
    model.eval()
    payload: dict[str, np.ndarray] = {}
    if args.modality in {"text", "pair"}:
        tokenizer = model.tokenizer(config["clamp3"]["text_model"])
        payload["text_embeddings"] = encode_texts(
            frame["text"].astype(str).tolist(),
            model,
            tokenizer,
            device,
            int(config["data"]["batch_size_text_embedding"]),
            int(config["clamp3"]["max_text_length"]),
        )
    if args.modality in {"audio", "pair"}:
        payload["audio_embeddings"] = encode_audio_rows(
            frame,
            model,
            device,
            int(config["data"]["batch_size_audio_embedding"]),
            int(config["clamp3"]["max_audio_length"]),
        )
    va_targets, has_va = targets(frame)
    payload.update(
        ids=frame[id_column].astype(str).to_numpy(dtype=str),
        splits=frame["split"].astype(str).to_numpy(dtype=str),
        datasets=frame["dataset"].astype(str).to_numpy(dtype=str),
        global_identities=frame["global_identity"].astype(str).to_numpy(dtype=str),
        va_targets=va_targets,
        has_va=has_va,
        manifest_sha256=np.asarray(file_sha256(manifest)),
        checkpoint_sha256=np.asarray(actual_hash),
    )
    for name, value in payload.items():
        if name.endswith("_embeddings"):
            if value.shape != (len(frame), 768) or not np.isfinite(value).all():
                raise ValueError(f"invalid {name}: {value.shape}")
            norms = np.linalg.norm(value, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                raise ValueError(f"{name} is not unit normalized")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    metadata = {
        "manifest": manifest.as_posix(),
        "manifest_sha256": payload["manifest_sha256"].item(),
        "checkpoint_sha256": actual_hash,
        "modality": args.modality,
        "samples": len(frame),
        "selected_split": args.split,
        "split_counts": frame["split"].value_counts().sort_index().to_dict(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "checkpoint_epoch": model.checkpoint_epoch,
        "checkpoint_loss": model.checkpoint_loss,
        "output": output.as_posix(),
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
