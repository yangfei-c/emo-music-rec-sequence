"""Validation/test evaluation for global plus query-conditioned segment matching."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.evaluation.pairs import grouped_retrieval_metrics_from_scores
from emomm_sequence.retrieval.segments import query_conditioned_segment_similarity
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--local-weights", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3]
    )
    parser.add_argument(
        "--temperatures", type=float, nargs="+", default=[0.03, 0.07, 0.1]
    )
    parser.add_argument(
        "--top-k", type=int, nargs="+", default=[1, 2, 3],
        help="Number of windows; use 0 for soft attention over every real window.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--track-chunk-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def encode_rows(
    model: FrozenRetriever,
    modality: str,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    output = []
    for start in range(0, len(values), batch_size):
        features = torch.from_numpy(values[start : start + batch_size].astype(np.float32)).to(device)
        output.append(model.encode_factor(modality, "semantic", features).cpu())
    return torch.cat(output)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.track_chunk_size <= 0:
        raise ValueError("batch sizes must be positive")
    if any(not 0.0 <= value <= 1.0 for value in args.local_weights):
        raise ValueError("local weights must be in [0,1]")
    if any(value <= 0 for value in args.temperatures):
        raise ValueError("temperatures must be positive")
    if any(value < 0 for value in args.top_k):
        raise ValueError("top-k must be non-negative")

    pair_path, segment_path, checkpoint_path = map(
        resolve, (args.pairs, args.segments, args.checkpoint)
    )
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    with np.load(pair_path, allow_pickle=False) as archive:
        split_mask = archive["splits"].astype(str) == args.split
        text_raw = archive["text_embeddings"][split_mask].astype(np.float32)
        music_rows = archive["audio_embeddings"][split_mask].astype(np.float32)
        identities = archive["global_identities"][split_mask].astype(str)
        datasets = archive["datasets"][split_mask].astype(str)
    if not split_mask.any():
        raise ValueError(f"pair archive contains no {args.split} rows")
    unique_ids, first = np.unique(identities, return_index=True)
    order = np.argsort(first)
    unique_ids, first = unique_ids[order], first[order]
    music_raw = music_rows[first]

    with np.load(segment_path, allow_pickle=False) as archive:
        catalog_ids = archive["track_ids"].astype(str)
        catalog_segments = archive["embeddings"]
        catalog_mask = archive["mask"].astype(bool)
    if len(set(catalog_ids.tolist())) != len(catalog_ids):
        raise ValueError("segment catalog has duplicate track identities")
    catalog_index = {identity: index for index, identity in enumerate(catalog_ids)}
    missing = [identity for identity in unique_ids if identity not in catalog_index]
    if missing:
        raise ValueError(f"segment catalog lacks pair identities: {missing[:3]}")
    positions = np.asarray([catalog_index[value] for value in unique_ids], dtype=np.int64)
    segments_raw = catalog_segments[positions].astype(np.float32)
    segment_mask = catalog_mask[positions]

    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = FrozenRetriever(checkpoint_path).to(device)
    model.eval()
    with torch.inference_mode():
        text = encode_rows(model, "text", text_raw, device, args.batch_size)
        music = encode_rows(model, "music", music_raw, device, args.batch_size)
        global_scores = text @ music.T
        flat = segments_raw.reshape(-1, segments_raw.shape[-1])
        adapted_flat = encode_rows(model, "music", flat, device, args.batch_size)
        adapted_segments = adapted_flat.reshape(*segments_raw.shape[:2], -1)

    results = []
    for top_k_value in args.top_k:
        top_k = None if top_k_value == 0 else top_k_value
        for temperature in args.temperatures:
            local_chunks = []
            for start in range(0, len(unique_ids), args.track_chunk_size):
                local, _ = query_conditioned_segment_similarity(
                    text,
                    adapted_segments[start : start + args.track_chunk_size],
                    torch.from_numpy(
                        segment_mask[start : start + args.track_chunk_size]
                    ),
                    temperature=temperature,
                    top_k=top_k,
                )
                local_chunks.append(local)
            local_scores = torch.cat(local_chunks, dim=1)
            for local_weight in args.local_weights:
                mixed = (
                    (1.0 - local_weight) * global_scores
                    + local_weight * local_scores
                ).numpy()
                metrics = grouped_retrieval_metrics_from_scores(
                    mixed, identities, unique_ids, ks=(1, 5, 10)
                )
                selection = 0.5 * (
                    metrics["text_to_music"]["mrr"]
                    + metrics["music_to_text"]["mrr"]
                )
                results.append(
                    {
                        "local_weight": float(local_weight),
                        "segment_temperature": float(temperature),
                        "segment_top_k": top_k,
                        "selection_score": float(selection),
                        "metrics": metrics,
                    }
                )
    best = max(
        results,
        key=lambda item: (
            item["selection_score"],
            -item["local_weight"],
            -(item["segment_top_k"] or segments_raw.shape[1]),
        ),
    )
    report = {
        "protocol": (
            "hyperparameters may be selected on validation only; a test run must "
            "reuse the locked validation setting without another sweep"
        ),
        "split": args.split,
        "datasets": sorted(set(datasets.tolist())),
        "captions": len(identities),
        "tracks": len(unique_ids),
        "inputs": {
            "pairs": pair_path.relative_to(ROOT).as_posix(),
            "pairs_sha256": sha256(pair_path),
            "segments": segment_path.relative_to(ROOT).as_posix(),
            "segments_sha256": sha256(segment_path),
            "checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256(checkpoint_path),
        },
        "best": best,
        "grid": results,
        "device": str(device),
    }
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
