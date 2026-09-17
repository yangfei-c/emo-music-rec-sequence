"""Build quality-filtered train-only MERGE lyric views with frozen CLaMP3.

This adapts RobustCLAP's multi-view supervision idea to the paired data that
actually exists in this repository.  The generated views are lyric excerpts,
not LLM paraphrases, and are never added to validation or test partitions.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.encoding.clamp3 import Clamp3TextAudioEncoder
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.semantic_views import (
    lyric_window_candidates,
    rank_view_candidates,
)
from emomm_sequence.training.utilities import load_yaml

from extract_clamp3_features import encode_texts


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests/pairs_merge.csv")
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=Path("artifacts/features/clamp3/clean/pairs_merge_clean.npz"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/features/clamp3/augmentation/merge_train_views"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--window-words", type=int, default=96)
    parser.add_argument("--candidate-count", type=int, default=5)
    parser.add_argument("--selected-views", type=int, default=2)
    parser.add_argument("--min-lyrics-words", type=int, default=64)
    parser.add_argument("--audio-weight", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = resolve(args.manifest)
    base_path = resolve(args.base)
    output_directory = resolve(args.output_directory)
    config_path = resolve(args.config)
    if not 1 <= args.selected_views <= args.candidate_count:
        raise ValueError("selected-views must be in [1, candidate-count]")
    expected_outputs = [
        output_directory / f"rank_{rank}.npz"
        for rank in range(1, args.selected_views + 1)
    ]
    metadata_path = output_directory / "metadata.json"
    existing = [path for path in [*expected_outputs, metadata_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "augmentation output exists; pass --overwrite to replace: "
            + ", ".join(str(path) for path in existing)
        )

    frame = pd.read_csv(manifest, keep_default_na=False)
    frame = frame[frame["split"].astype(str) == "train"].reset_index(drop=True)
    manifest_train_rows = len(frame)
    if frame.empty:
        raise ValueError("MERGE manifest contains no train rows")
    if frame["pair_id"].duplicated().any():
        raise ValueError("MERGE train manifest contains duplicate pair IDs")

    with np.load(base_path, allow_pickle=False) as archive:
        base = {key: archive[key] for key in archive.files}
    base_ids = base["ids"].astype(str)
    if len(set(base_ids.tolist())) != len(base_ids):
        raise ValueError("base MERGE archive contains duplicate pair IDs")
    base_index = {pair_id: index for index, pair_id in enumerate(base_ids)}
    missing = sorted(set(frame["pair_id"].astype(str)) - set(base_index))
    if missing:
        raise ValueError(f"train IDs missing from base embeddings: {missing[:3]}")
    row_indices = np.asarray(
        [base_index[pair_id] for pair_id in frame["pair_id"].astype(str)],
        dtype=np.int64,
    )
    # The content-deduplication protocol can conservatively move an original
    # manifest train row to validation/test.  The cleaned split is authoritative
    # and such rows must never be emitted as augmentation.
    clean_train = base["splits"][row_indices].astype(str) == "train"
    excluded_clean_holdout_ids = frame.loc[~clean_train, "pair_id"].astype(str).tolist()
    frame = frame.loc[clean_train].reset_index(drop=True)
    row_indices = row_indices[clean_train]
    if not np.all(base["splits"][row_indices].astype(str) == "train"):
        raise RuntimeError("clean split filtering failed")
    if not np.array_equal(
        base["global_identities"][row_indices].astype(str),
        frame["global_identity"].astype(str).to_numpy(),
    ):
        raise ValueError("manifest/base identity disagreement")

    records: list[tuple[int, int, str]] = []
    candidate_counts = []
    for row, text in enumerate(frame["text"].astype(str)):
        candidates = lyric_window_candidates(
            text,
            window_words=args.window_words,
            candidate_count=args.candidate_count,
            min_lyrics_words=args.min_lyrics_words,
        )
        candidate_counts.append(len(candidates))
        records.extend((row, index, candidate) for index, candidate in enumerate(candidates))
    if not records:
        raise ValueError("no eligible lyric views were generated")

    config = load_yaml(config_path)
    checkpoint = resolve(Path(config["clamp3"]["checkpoint"]))
    actual_checkpoint_hash = sha256(checkpoint)
    expected_checkpoint_hash = str(config["clamp3"]["checkpoint_sha256"]).lower()
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("CLaMP3 checkpoint SHA-256 mismatch")
    device = torch.device(
        config.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    model = Clamp3TextAudioEncoder(
        checkpoint,
        text_model_name=config["clamp3"]["text_model"],
        max_audio_length=int(config["clamp3"]["max_audio_length"]),
        normalize=False,
        freeze=True,
    ).to(device)
    model.eval()
    tokenizer = model.tokenizer(config["clamp3"]["text_model"])
    candidate_embeddings = encode_texts(
        [record[2] for record in records],
        model,
        tokenizer,
        device,
        int(config["data"]["batch_size_text_embedding"]),
        int(config["clamp3"]["max_text_length"]),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    by_row: dict[int, list[int]] = {}
    for embedding_index, (row, _, _) in enumerate(records):
        by_row.setdefault(row, []).append(embedding_index)
    full_text = base["text_embeddings"][row_indices].astype(np.float32)
    audio = base["audio_embeddings"][row_indices].astype(np.float32)
    selected: list[list[int]] = [[] for _ in range(args.selected_views)]
    scores: list[dict] = []
    for row, embedding_indices in by_row.items():
        views = candidate_embeddings[embedding_indices]
        audio_similarity = (views @ audio[row]).astype(float).tolist()
        text_similarity = (views @ full_text[row]).astype(float).tolist()
        ranking = rank_view_candidates(
            audio_similarity,
            text_similarity,
            audio_weight=args.audio_weight,
        )
        for rank, local_index in enumerate(ranking[: args.selected_views]):
            selected[rank].append(embedding_indices[local_index])
        scores.append(
            {
                "pair_id": str(frame.iloc[row]["pair_id"]),
                "candidate_count": len(embedding_indices),
                "ranked_candidate_indices": [
                    int(records[embedding_indices[index]][1]) for index in ranking
                ],
                "candidate_audio_cosine": audio_similarity,
                "candidate_full_text_cosine": text_similarity,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    output_records = []
    for rank, embedding_indices in enumerate(selected, start=1):
        source_rows = np.asarray([records[index][0] for index in embedding_indices])
        source_base_indices = row_indices[source_rows]
        output_path = output_directory / f"rank_{rank}.npz"
        payload = {
            "text_embeddings": candidate_embeddings[embedding_indices],
            "audio_embeddings": base["audio_embeddings"][source_base_indices].astype(
                np.float32
            ),
            "ids": np.asarray(
                [
                    f"{base['ids'][base_index_value]}:lyric_view_rank_{rank}"
                    for base_index_value in source_base_indices
                ],
                dtype=str,
            ),
            "splits": np.asarray(["train"] * len(source_rows), dtype=str),
            "datasets": np.asarray(["merge"] * len(source_rows), dtype=str),
            "global_identities": base["global_identities"][source_base_indices].astype(
                str
            ),
            "va_targets": base["va_targets"][source_base_indices].astype(np.float32),
            "has_va": base["has_va"][source_base_indices].astype(bool),
            "source_pair_ids": base["ids"][source_base_indices].astype(str),
            "view_rank": np.full(len(source_rows), rank, dtype=np.int64),
            "manifest_sha256": np.asarray(sha256(manifest)),
            "base_sha256": np.asarray(sha256(base_path)),
            "checkpoint_sha256": np.asarray(actual_checkpoint_hash),
        }
        norms = np.linalg.norm(payload["text_embeddings"], axis=1)
        if not np.isfinite(payload["text_embeddings"]).all() or not np.allclose(
            norms, 1.0, atol=1.0e-4
        ):
            raise ValueError(f"rank-{rank} text embeddings are invalid")
        np.savez_compressed(output_path, **payload)
        output_records.append(
            {
                "rank": rank,
                "path": str(output_path.relative_to(ROOT)),
                "sha256": sha256(output_path),
                "samples": len(source_rows),
                "mean_audio_cosine": float(
                    np.mean(
                        np.sum(
                            payload["text_embeddings"] * payload["audio_embeddings"],
                            axis=1,
                        )
                    )
                ),
                "mean_full_text_cosine": float(
                    np.mean(
                        np.sum(payload["text_embeddings"] * full_text[source_rows], axis=1)
                    )
                ),
            }
        )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "frozen-CLaMP3 quality-filtered MERGE lyric excerpt views",
        "claim_boundary": (
            "train-only lyric excerpts inspired by multi-view contrastive training; "
            "these are not LLM paraphrases and not a RobustCLAP reproduction"
        ),
        "sources": {
            "manifest": str(manifest.relative_to(ROOT)),
            "manifest_sha256": sha256(manifest),
            "base": str(base_path.relative_to(ROOT)),
            "base_sha256": sha256(base_path),
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_sha256": actual_checkpoint_hash,
        },
        "parameters": {
            "window_words": args.window_words,
            "candidate_count": args.candidate_count,
            "selected_views": args.selected_views,
            "min_lyrics_words": args.min_lyrics_words,
            "audio_weight": args.audio_weight,
        },
        "input_manifest_train_rows": manifest_train_rows,
        "clean_train_rows": len(frame),
        "excluded_clean_holdout_ids": excluded_clean_holdout_ids,
        "eligible_rows": len(by_row),
        "skipped_rows": len(frame) - len(by_row),
        "candidate_views": len(records),
        "candidate_count_distribution": {
            str(value): int(candidate_counts.count(value))
            for value in sorted(set(candidate_counts))
        },
        "outputs": output_records,
        "selection_scores": scores,
        "partition_policy": (
            "only rows marked train in both the original manifest and the cleaned "
            "content-deduplicated archive are emitted"
        ),
        "literature": {
            "name": "Do Audio-Language Models Understand Linguistic Variations?",
            "method": "RobustCLAP multi-view contrastive learning",
            "venue": "NAACL 2025",
            "url": "https://aclanthology.org/2025.naacl-short.76/",
        },
        "device": str(device),
    }
    write_json(metadata_path, metadata)
    print(json.dumps({key: value for key, value in metadata.items() if key != "selection_scores"}, indent=2))


if __name__ == "__main__":
    main()
