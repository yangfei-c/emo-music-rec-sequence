"""Extract contextual CLaMP3 window embeddings for dense semantic reranking."""

from __future__ import annotations

import argparse
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
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.utilities import load_yaml


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unique_tracks(manifests: list[Path]) -> pd.DataFrame:
    frames = []
    for manifest in manifests:
        frame = pd.read_csv(manifest, keep_default_na=False)
        required = {"global_identity", "feature_path", "dataset", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{manifest} lacks columns: {sorted(missing)}")
        selected = frame[list(required)].copy()
        selected["source_manifest"] = manifest.as_posix()
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    for identity, group in combined.groupby("global_identity", sort=False):
        if group["feature_path"].astype(str).nunique() != 1:
            raise ValueError(f"conflicting audio features for {identity}")
        if group["split"].astype(str).nunique() != 1:
            raise ValueError(f"identity crosses data partitions: {identity}")
    return combined.drop_duplicates("global_identity", keep="first").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    manifests = [resolve(path) for path in args.manifests]
    output = resolve(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace: {output}")
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    config = load_yaml(resolve(args.config))
    checkpoint = resolve(config["clamp3"]["checkpoint"])
    actual_hash = sha256(checkpoint)
    if actual_hash != str(config["clamp3"]["checkpoint_sha256"]).lower():
        raise ValueError("CLaMP3 checkpoint SHA-256 mismatch")
    frame = unique_tracks(manifests)
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    batch_size = args.batch_size or int(config["data"]["batch_size_audio_embedding"])
    max_length = int(config["clamp3"]["max_audio_length"])
    model = Clamp3TextAudioEncoder(
        checkpoint,
        text_model_name=config["clamp3"]["text_model"],
        max_audio_length=max_length,
        normalize=False,
        freeze=True,
    ).to(device)
    model.eval()

    extracted: list[np.ndarray] = []
    for start in tqdm(
        range(0, len(frame), batch_size), desc="CLaMP3 segments", unit="batch"
    ):
        batch = frame.iloc[start : start + batch_size]
        tokens = [
            load_audio_tokens(resolve(path), max_length=max_length)
            for path in batch["feature_path"].astype(str)
        ]
        values, mask = collate_audio_tokens(tokens)
        with torch.inference_mode():
            segments, segment_mask = model.encode_audio_segments(
                values.to(device), mask.to(device), exclude_boundary_tokens=True
            )
            segments = F.normalize(segments.float(), dim=-1).cpu()
            segment_mask = segment_mask.cpu()
        extracted.extend(
            segments[index, segment_mask[index]].numpy().astype(np.float32)
            for index in range(len(batch))
        )

    lengths = np.asarray([len(value) for value in extracted], dtype=np.int64)
    if (lengths <= 0).any():
        raise ValueError("a track produced no real CLaMP3 windows")
    width = int(lengths.max())
    embeddings = np.zeros((len(extracted), width, 768), dtype=np.float16)
    segment_mask = np.zeros((len(extracted), width), dtype=bool)
    for index, value in enumerate(extracted):
        embeddings[index, : len(value)] = value.astype(np.float16)
        segment_mask[index, : len(value)] = True
    stored = embeddings[segment_mask].astype(np.float32)
    norms = np.linalg.norm(stored, axis=1)
    if not np.isfinite(stored).all() or not np.allclose(norms, 1.0, atol=1.0e-3):
        raise ValueError("invalid contextual segment embeddings")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        track_ids=frame["global_identity"].astype(str).to_numpy(dtype=str),
        embeddings=embeddings,
        mask=segment_mask,
        lengths=lengths,
        datasets=frame["dataset"].astype(str).to_numpy(dtype=str),
        splits=frame["split"].astype(str).to_numpy(dtype=str),
        feature_paths=frame["feature_path"].astype(str).to_numpy(dtype=str),
        checkpoint_sha256=np.asarray(actual_hash),
    )
    metadata = {
        "method": "frozen CLaMP3 audio-transformer contextual windows",
        "claim_boundary": (
            "the dense reranker adds no new encoder or trainable parameter; "
            "boundary tokens are excluded and all real 30-second windows are retained"
        ),
        "manifests": [
            {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
            for path in manifests
        ],
        "checkpoint_sha256": actual_hash,
        "tracks": len(frame),
        "maximum_windows": width,
        "mean_windows": float(lengths.mean()),
        "stored_dtype": "float16",
        "device": str(device),
        "output": output.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
