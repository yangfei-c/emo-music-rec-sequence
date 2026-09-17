"""Evaluate MTG label-to-track retrieval by frozen train-frequency groups."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name.strip(), resolve(raw_path.strip())


@torch.inference_mode()
def per_label_metrics(
    checkpoint: Path,
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    factor: str,
    split: str,
    batch_size: int,
    ks: tuple[int, ...],
    device: torch.device,
) -> dict:
    model = FrozenRetriever(checkpoint).to(device).eval()
    split_mask = mtg["splits"].astype(str) == split
    raw_text = mtg[f"{factor}_label_embeddings"].astype(np.float32)
    raw_music = music_index[mtg["index_positions"][split_mask]].astype(np.float32)
    positives = mtg[f"{factor}_targets"][split_mask].astype(bool).T
    text_parts, music_parts = [], []
    for start in range(0, len(raw_text), batch_size):
        batch = torch.from_numpy(raw_text[start : start + batch_size]).to(device)
        text_parts.append(
            model.encode_factor("text", factor, batch).float().cpu().numpy()
        )
    for start in range(0, len(raw_music), batch_size):
        batch = torch.from_numpy(raw_music[start : start + batch_size]).to(device)
        music_parts.append(
            model.encode_factor("music", factor, batch).float().cpu().numpy()
        )
    scores = np.concatenate(text_parts) @ np.concatenate(music_parts).T
    return label_to_track_metrics(scores, positives, ks)


def mean_metrics(rows: list[dict], metric_names: list[str]) -> dict[str, float]:
    return {
        metric: float(np.mean([float(row[metric]) for row in rows]))
        for metric in metric_names
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--factor", choices=("preference", "affect"), default="preference")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--groups", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/retriever.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.groups != 3:
        parser.error("this report currently defines exactly tail/mid/head thirds")
    checkpoints = dict(args.checkpoint)
    if len(checkpoints) != len(args.checkpoint):
        parser.error("checkpoint names must be unique")
    for path in checkpoints.values():
        if not path.is_file():
            parser.error(f"checkpoint does not exist: {path}")
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mtg_path = resolve(config["supervision"]["mtg_factors"])
    music_path = resolve(config["supervision"]["music_index"])
    with np.load(mtg_path, allow_pickle=False) as data:
        mtg = {key: data[key] for key in data.files}
    with np.load(music_path, allow_pickle=False) as data:
        music_index = data["embeddings"].astype(np.float32)

    factor = args.factor
    train_mask = mtg["splits"].astype(str) == "train"
    frequencies = mtg[f"{factor}_targets"][train_mask].sum(axis=0).astype(int)
    labels = mtg[f"{factor}_labels"].astype(str)
    # The secondary label key makes tied-frequency grouping deterministic.
    ordered = sorted(range(len(labels)), key=lambda index: (frequencies[index], labels[index]))
    chunks = np.array_split(np.asarray(ordered, dtype=int), 3)
    group_indices = {name: chunk.tolist() for name, chunk in zip(("tail", "mid", "head"), chunks)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ks = tuple(int(value) for value in config["evaluation"]["tag_ks"])
    output = {
        "partition": args.split,
        "test_evaluated": args.split == "test",
        "factor": factor,
        "group_definition": (
            "Equal label-count thirds ordered by (train positive count, label name); "
            "groups are frozen before checkpoint evaluation."
        ),
        "sources": {
            "config": {"path": str(config_path), "sha256": sha256(config_path)},
            "mtg": {"path": str(mtg_path), "sha256": sha256(mtg_path)},
            "music_index": {"path": str(music_path), "sha256": sha256(music_path)},
            "checkpoints": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in checkpoints.items()
            },
        },
        "groups": {},
        "checkpoints": {},
    }
    for group, indices in group_indices.items():
        output["groups"][group] = {
            "label_count": len(indices),
            "train_frequency_min": int(frequencies[indices].min()),
            "train_frequency_max": int(frequencies[indices].max()),
            "train_positive_total": int(frequencies[indices].sum()),
            "labels": [labels[index] for index in indices],
        }

    metric_names: list[str] | None = None
    for name, checkpoint in checkpoints.items():
        metrics = per_label_metrics(
            checkpoint,
            mtg,
            music_index,
            factor,
            args.split,
            args.batch_size,
            ks,
            device,
        )
        rows_by_index = {int(row["query_index"]): row for row in metrics["per_query"]}
        if metric_names is None:
            metric_names = [
                key
                for key in metrics["per_query"][0]
                if key not in {"query_index", "positives"}
            ]
        groups = {}
        for group, indices in group_indices.items():
            rows = [rows_by_index[index] for index in indices if index in rows_by_index]
            groups[group] = {
                "evaluated_labels": len(rows),
                "metrics": mean_metrics(rows, metric_names),
            }
        output["checkpoints"][name] = {
            "overall": metrics["macro"],
            "groups": groups,
            "per_label": [
                {
                    "label": labels[index],
                    "train_frequency": int(frequencies[index]),
                    **rows_by_index[index],
                }
                for index in ordered
                if index in rows_by_index
            ],
        }

    names = list(checkpoints)
    if len(names) == 2:
        baseline, candidate = names
        output["comparison"] = {
            "candidate": candidate,
            "baseline": baseline,
            "candidate_minus_baseline": {
                "overall": {
                    metric: output["checkpoints"][candidate]["overall"][metric]
                    - output["checkpoints"][baseline]["overall"][metric]
                    for metric in metric_names
                },
                "groups": {
                    group: {
                        metric: output["checkpoints"][candidate]["groups"][group]["metrics"][metric]
                        - output["checkpoints"][baseline]["groups"][group]["metrics"][metric]
                        for metric in metric_names
                    }
                    for group in group_indices
                },
            },
        }
    output_path = resolve(args.output)
    if output_path.exists():
        raise FileExistsError(output_path)
    write_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
