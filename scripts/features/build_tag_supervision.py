"""Build leakage-aware Ours factor supervision from official MTG splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.encoding.clamp3 import Clamp3TextAudioEncoder
from emomm_sequence.data.tags import (
    read_mtg_autotagging_tsv,
    select_supported_labels,
    tag_prompt,
)
from emomm_sequence.training.utilities import load_yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def encode_prompts(
    prompts: list[str],
    encoder: Clamp3TextAudioEncoder,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    parts = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        tokens = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        parts.append(
            encoder.encode_text(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            ).float().cpu().numpy()
        )
    return np.concatenate(parts).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "training" / "retriever.yaml"
    )
    parser.add_argument(
        "--backbone-config",
        type=Path,
        default=ROOT / "configs" / "data.yaml",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=ROOT / "artifacts" / "catalog" / "music.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "features" / "retrieval" / "mtg_factor_supervision.npz",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    backbone_config_path = (
        args.backbone_config if args.backbone_config.is_absolute() else ROOT / args.backbone_config
    )
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Ours supervision already exists: {output}")
    config, backbone_config = load_yaml(config_path), load_yaml(backbone_config_path)
    split_dir = Path(backbone_config["paths"]["mtg_split_dir"])
    source_paths = {
        split: split_dir / f"autotagging-{split}.tsv"
        for split in ("train", "validation", "test")
    }
    source_tracks = []
    for split, path in source_paths.items():
        source_tracks.extend(read_mtg_autotagging_tsv(path, split))
    identities = [row.track_id for row in source_tracks]
    if len(identities) != len(set(identities)):
        raise ValueError("an MTG track occurs in more than one official split")

    with np.load(index_path, allow_pickle=False) as index:
        index_track_ids = index["track_ids"].astype(str)
    index_position = {track_id: position for position, track_id in enumerate(index_track_ids)}
    tracks = [row for row in source_tracks if row.track_id in index_position]
    supervision = config["supervision"]
    min_tracks = int(supervision["min_train_tracks_per_label"])
    min_artists = int(supervision["min_train_artists_per_label"])
    preference_labels, preference_support = select_supported_labels(
        tracks,
        tuple(supervision["preference_prefixes"]),
        min_train_tracks=min_tracks,
        min_train_artists=min_artists,
    )
    affect_labels, affect_support = select_supported_labels(
        tracks,
        tuple(supervision["affect_prefixes"]),
        min_train_tracks=min_tracks,
        min_train_artists=min_artists,
    )
    preference_index = {label: index for index, label in enumerate(preference_labels)}
    affect_index = {label: index for index, label in enumerate(affect_labels)}
    preference_targets = np.zeros((len(tracks), len(preference_labels)), dtype=bool)
    affect_targets = np.zeros((len(tracks), len(affect_labels)), dtype=bool)
    for row_index, row in enumerate(tracks):
        for tag in row.tags:
            if tag in preference_index:
                preference_targets[row_index, preference_index[tag]] = True
            if tag in affect_index:
                affect_targets[row_index, affect_index[tag]] = True

    clamp = backbone_config["clamp3"]
    checkpoint = ROOT / clamp["checkpoint"]
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    encoder = Clamp3TextAudioEncoder(
        checkpoint_path=checkpoint,
        text_model_name=clamp["text_model"],
        hidden_size=int(clamp["embedding_dim"]),
        audio_layers=int(clamp.get("audio_layers", 12)),
        max_audio_length=int(clamp["max_audio_length"]),
        normalize=True,
        freeze=True,
    ).to(device).eval()
    tokenizer = encoder.tokenizer(clamp["text_model"])
    preference_prompts = [tag_prompt(label) for label in preference_labels]
    affect_prompts = [tag_prompt(label) for label in affect_labels]
    preference_embeddings = encode_prompts(
        preference_prompts,
        encoder,
        tokenizer,
        device,
        int(args.batch_size),
        int(clamp["max_text_length"]),
    )
    affect_embeddings = encode_prompts(
        affect_prompts,
        encoder,
        tokenizer,
        device,
        int(args.batch_size),
        int(clamp["max_text_length"]),
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        track_ids=np.asarray([row.track_id for row in tracks], dtype=str),
        index_positions=np.asarray([index_position[row.track_id] for row in tracks], dtype=np.int64),
        artist_ids=np.asarray([row.artist_id for row in tracks], dtype=str),
        album_ids=np.asarray([row.album_id for row in tracks], dtype=str),
        splits=np.asarray([row.split for row in tracks], dtype=str),
        preference_labels=np.asarray(preference_labels, dtype=str),
        preference_prompts=np.asarray(preference_prompts, dtype=str),
        preference_label_embeddings=preference_embeddings,
        preference_targets=preference_targets,
        affect_labels=np.asarray(affect_labels, dtype=str),
        affect_prompts=np.asarray(affect_prompts, dtype=str),
        affect_label_embeddings=affect_embeddings,
        affect_targets=affect_targets,
        source_index_sha256=np.asarray(sha256(index_path)),
        backbone_checkpoint_sha256=np.asarray(clamp["checkpoint_sha256"]),
    )
    split_counts = {
        split: int(sum(row.split == split for row in tracks))
        for split in ("train", "validation", "test")
    }
    report = {
        "tracks_in_official_tsv": len(source_tracks),
        "tracks_in_clamp3_index": len(index_track_ids),
        "matched_tracks": len(tracks),
        "missing_tsv_tracks_from_index": len(source_tracks) - len(tracks),
        "split_counts": split_counts,
        "preference_labels": len(preference_labels),
        "affect_labels": len(affect_labels),
        "rows_with_preference": int(preference_targets.any(axis=1).sum()),
        "rows_with_affect": int(affect_targets.any(axis=1).sum()),
        "preference_support": preference_support,
        "affect_support": affect_support,
        "source_sha256": {split: sha256(path) for split, path in source_paths.items()},
        "index_sha256": sha256(index_path),
        "config_sha256": sha256(config_path),
        "backbone_config_sha256": sha256(backbone_config_path),
        "output": output.as_posix(),
        "output_sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
