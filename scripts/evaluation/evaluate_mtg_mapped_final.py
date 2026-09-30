"""One-shot MTG test evaluation after validation-only model selection.

The script deliberately separates preparation from execution. ``prepare`` locks
every input and checkpoint hash; ``run`` refuses to overwrite an existing test
result. This prevents accidental test-set model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json


RUN_ID = "mtg_mapped_final_test_20260914"
OUTPUT = ROOT / "artifacts/runs" / RUN_ID / "evaluation"
MAPPED = ROOT / "artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz"
QWEN = ROOT / "artifacts/runs/diagnosis_20260914_tower_freeze/evaluation/query_form_embeddings.npz"
ALIGNMENT = ROOT / "artifacts/runs/diagnosis_20260914_query_alignment/evaluation/execution.json"
BASELINES = ROOT / "artifacts/runs/mtg_mapped_baselines_20260914/evaluation/execution.json"
CONFIG = ROOT / "configs/training/retriever.yaml"
SELECTED_VARIANT = "mapped_three_view"
METRICS = (
    "ndcg_at_10",
    "average_precision",
    "macro_roc_auc",
    "precision_at_10",
    "recall_at_10",
)


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, labels: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, labels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(value)


class MLPProbe(nn.Module):
    def __init__(self, embedding_dim: int, labels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, labels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def locked_checkpoints() -> dict[str, list[dict]]:
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    baselines = json.loads(BASELINES.read_text(encoding="utf-8"))
    chosen = [row for row in alignment if row["variant"] == SELECTED_VARIANT]
    result = {
        "ours_mapped_three_view": [
            {"seed": row["seed"], "path": row["checkpoint"], "sha256": row["checkpoint_sha256"]}
            for row in chosen
        ]
    }
    for method in ("linear_bce_probe", "mlp_bce_probe", "triplet_metric"):
        unique: dict[int, dict] = {}
        for row in baselines:
            if row["method"] == method:
                unique[int(row["seed"])] = {
                    "seed": int(row["seed"]),
                    "path": row["checkpoint"],
                    "sha256": row["checkpoint_sha256"],
                    "best_epoch": int(row["best_epoch"]),
                }
        result[method] = [unique[seed] for seed in sorted(unique)]
    if any(len(rows) != 3 for rows in result.values()):
        raise RuntimeError("expected exactly three locked seeds per trained method")
    return result


def prepare() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    checkpoints = locked_checkpoints()
    plan = {
        "run_id": RUN_ID,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "test",
        "one_shot": True,
        "selected_on_validation": SELECTED_VARIANT,
        "selection_rule": (
            "highest mean geometric(NDCG@10, AP) across canonical and Qwen runtime "
            "queries; test was not used"
        ),
        "query_forms": ["canonical", "qwen_runtime"],
        "metrics": list(METRICS),
        "same_protocol": (
            "same official split, mapped label identities, unchanged target matrix, candidate "
            "tracks, CLaMP3 audio features, and metric implementation"
        ),
        "interface_boundary": (
            "Ours, frozen CLaMP3 and triplet metric accept open text embeddings; probes and "
            "positive centroids are closed-vocabulary references"
        ),
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (MAPPED, QWEN, ALIGNMENT, BASELINES, CONFIG, Path(__file__))
        },
        "checkpoints": checkpoints,
    }
    write_json(OUTPUT / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def verify_plan(plan: dict) -> None:
    for relative, expected in plan["inputs"].items():
        path = ROOT / relative
        if sha256(path) != expected:
            raise RuntimeError(f"locked protocol input changed: {path}")
    for rows in plan["checkpoints"].values():
        for row in rows:
            path = ROOT / row["path"]
            if sha256(path) != row["sha256"]:
                raise RuntimeError(f"locked checkpoint changed: {path}")


def load_data() -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    with np.load(MAPPED, allow_pickle=False) as archive:
        mtg = {key: archive[key] for key in archive.files}
    with np.load(ROOT / config["supervision"]["music_index"], allow_pickle=False) as archive:
        catalog = archive["embeddings"].astype(np.float32)
    with np.load(QWEN, allow_pickle=False) as archive:
        qwen = {key: archive[key] for key in archive.files}
    if not np.array_equal(mtg["preference_labels"].astype(str), qwen["labels"].astype(str)):
        raise ValueError("Qwen query order differs from MTG label order")
    queries = {
        "canonical": mtg["preference_label_embeddings"].astype(np.float32),
        "qwen_runtime": qwen["qwen_preference"].astype(np.float32),
    }
    return mtg, catalog, queries


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[str, list[int]]:
    train = mtg["splits"].astype(str) == "train"
    frequency = mtg["preference_targets"][train].sum(axis=0)
    labels = mtg["preference_labels"].astype(str)
    order = sorted(range(len(labels)), key=lambda index: (frequency[index], labels[index]))
    return {
        name: values.tolist()
        for name, values in zip(("tail", "mid", "head"), np.array_split(order, 3))
    }


def evaluate_scores(
    scores: np.ndarray,
    positives: np.ndarray,
    groups: dict[str, list[int]],
    labels: np.ndarray,
) -> dict:
    retrieval = label_to_track_metrics(scores, positives, (5, 10, 20))
    rows = {int(row["query_index"]): dict(row) for row in retrieval["per_query"]}
    auc = {
        index: float(roc_auc_score(positives[index], scores[index]))
        for index in range(len(positives))
        if np.unique(positives[index]).size == 2
    }
    for index, row in rows.items():
        row["label"] = str(labels[index])
        row["roc_auc"] = auc[index]

    def summarize(indices: list[int]) -> dict[str, float]:
        return {
            "ndcg_at_10": float(np.mean([rows[index]["ndcg_at_10"] for index in indices])),
            "average_precision": float(np.mean([rows[index]["average_precision"] for index in indices])),
            "macro_roc_auc": float(np.mean([rows[index]["roc_auc"] for index in indices])),
            "precision_at_10": float(np.mean([rows[index]["precision_at_10"] for index in indices])),
            "recall_at_10": float(np.mean([rows[index]["recall_at_10"] for index in indices])),
        }

    return {
        "overall": summarize(list(range(len(positives)))),
        "frequency_groups": {name: summarize(indices) for name, indices in groups.items()},
        "per_label": [rows[index] for index in range(len(rows))],
    }


@torch.inference_mode()
def adapted_scores(
    checkpoint: Path,
    raw_music: np.ndarray,
    query: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model = FrozenRetriever(checkpoint).to(device).eval()
    music_parts = []
    for start in range(0, len(raw_music), 1024):
        value = torch.from_numpy(raw_music[start : start + 1024]).to(device)
        music_parts.append(model.encode_factor("music", "preference", value).float().cpu().numpy())
    text = model.encode_factor(
        "text", "preference", torch.from_numpy(query).to(device)
    ).float().cpu().numpy()
    return text @ np.concatenate(music_parts).T


@torch.inference_mode()
def probe_scores(
    checkpoint: Path,
    raw_music: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model: nn.Module
    if payload["model_type"] == "linear_bce_probe":
        model = LinearProbe(int(payload["embedding_dim"]), int(payload["labels"]))
    else:
        model = MLPProbe(int(payload["embedding_dim"]), int(payload["labels"]))
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    parts = []
    for start in range(0, len(raw_music), 1024):
        value = torch.from_numpy(raw_music[start : start + 1024]).to(device)
        parts.append(model(value).float().cpu().numpy())
    return np.concatenate(parts).T


def aggregate(evaluations: list[dict], interface: str) -> dict:
    def metric_values(location: str, name: str) -> list[float]:
        return [float(row[location][name]) for row in evaluations]

    result = {
        "runs": len(evaluations),
        "query_interface": interface,
        "overall": {},
        "frequency_groups": {},
    }
    for metric in METRICS:
        values = metric_values("overall", metric)
        result["overall"][metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    for group in ("tail", "mid", "head"):
        result["frequency_groups"][group] = {}
        for metric in METRICS:
            values = [float(row["frequency_groups"][group][metric]) for row in evaluations]
            result["frequency_groups"][group][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
    averaged_per_label = []
    for index in range(len(evaluations[0]["per_label"])):
        source = evaluations[0]["per_label"][index]
        row = {"query_index": index, "label": source["label"], "positives": source["positives"]}
        for key in ("ndcg_at_10", "average_precision", "roc_auc", "precision_at_10", "recall_at_10"):
            row[key] = float(np.mean([item["per_label"][index][key] for item in evaluations]))
        averaged_per_label.append(row)
    result["per_label_mean"] = averaged_per_label
    return result


def paired_label_bootstrap(ours: dict, baseline: dict, samples: int = 10000) -> dict:
    rng = np.random.default_rng(20260914)
    result = {}
    key_map = {"macro_roc_auc": "roc_auc"}
    labels = len(ours["per_label_mean"])
    draws = rng.integers(0, labels, size=(samples, labels))
    for metric in METRICS:
        key = key_map.get(metric, metric)
        ours_values = np.asarray([row[key] for row in ours["per_label_mean"]])
        base_values = np.asarray([row[key] for row in baseline["per_label_mean"]])
        difference = ours_values - base_values
        sampled = difference[draws].mean(axis=1)
        result[metric] = {
            "mean_delta": float(difference.mean()),
            "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
            "labels_improved": int((difference > 0).sum()),
            "labels_total": int(labels),
        }
    return result


def run() -> None:
    result_path = OUTPUT / "test_execution.json"
    if result_path.exists():
        raise FileExistsError(f"one-shot test result already exists: {result_path}")
    plan = json.loads((OUTPUT / "plan.json").read_text(encoding="utf-8"))
    verify_plan(plan)
    mtg, catalog, queries = load_data()
    mask_train = mtg["splits"].astype(str) == "train"
    mask_test = mtg["splits"].astype(str) == "test"
    raw_test = catalog[mtg["index_positions"][mask_test]].astype(np.float32)
    targets_train = mtg["preference_targets"][mask_train].astype(bool)
    targets_test = mtg["preference_targets"][mask_test].astype(bool).T
    labels = mtg["preference_labels"].astype(str)
    groups = frequency_groups(mtg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods: dict[str, dict] = {}

    for form, query in queries.items():
        frozen = evaluate_scores(query @ raw_test.T, targets_test, groups, labels)
        methods[f"frozen_clamp3/{form}"] = aggregate([frozen], "open_text")

    raw_train = catalog[mtg["index_positions"][mask_train]].astype(np.float32)
    centroids = np.asarray(
        [raw_train[targets_train[:, index]].mean(axis=0) for index in range(len(labels))],
        dtype=np.float32,
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1.0e-8)
    methods["positive_centroid/closed_vocabulary_label_id"] = aggregate(
        [evaluate_scores(centroids @ raw_test.T, targets_test, groups, labels)],
        "closed_vocabulary",
    )

    checkpoints = plan["checkpoints"]
    for method in ("linear_bce_probe", "mlp_bce_probe"):
        evaluations = [
            evaluate_scores(
                probe_scores(ROOT / row["path"], raw_test, device),
                targets_test,
                groups,
                labels,
            )
            for row in checkpoints[method]
        ]
        methods[f"{method}/closed_vocabulary_label_id"] = aggregate(
            evaluations, "closed_vocabulary"
        )

    for method, name in (
        ("triplet_metric", "triplet_metric"),
        ("ours_mapped_three_view", "ours_mapped_three_view"),
    ):
        for form, query in queries.items():
            evaluations = [
                evaluate_scores(
                    adapted_scores(ROOT / row["path"], raw_test, query, device),
                    targets_test,
                    groups,
                    labels,
                )
                for row in checkpoints[method]
            ]
            methods[f"{name}/{form}"] = aggregate(evaluations, "open_text")

    significance = {
        form: paired_label_bootstrap(
            methods[f"ours_mapped_three_view/{form}"],
            methods[f"frozen_clamp3/{form}"],
        )
        for form in queries
    }
    output = {
        "run_id": RUN_ID,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "test",
        "test_tracks": int(mask_test.sum()),
        "labels": int(len(labels)),
        "selected_variant": SELECTED_VARIANT,
        "methods": methods,
        "paired_label_bootstrap_ours_vs_frozen": significance,
    }
    write_json(result_path, output)
    print(json.dumps({"test_tracks": int(mask_test.sum()), "significance": significance}, ensure_ascii=False, indent=2))


def summarize() -> None:
    source = json.loads((OUTPUT / "test_execution.json").read_text(encoding="utf-8"))
    methods = source["methods"]
    compact = []
    for name, method in sorted(
        methods.items(),
        key=lambda item: item[1]["overall"]["ndcg_at_10"]["mean"],
        reverse=True,
    ):
        compact.append(
            {
                "method": name,
                "interface": method["query_interface"],
                **{
                    metric: {
                        "mean": method["overall"][metric]["mean"],
                        "std": method["overall"][metric]["std"],
                    }
                    for metric in METRICS
                },
            }
        )
    summary = {
        "run_id": RUN_ID,
        "partition": "test",
        "selected_without_test": True,
        "result_interpretation": (
            "Only rows with query_interface=open_text are interface-equivalent. Closed-vocabulary "
            "rows are diagnostic classifiers, not arbitrary-language retrieval systems."
        ),
        "ranking": compact,
        "paired_label_bootstrap_ours_vs_frozen": source[
            "paired_label_bootstrap_ours_vs_frozen"
        ],
    }
    write_json(OUTPUT / "summary.json", summary)
    with (OUTPUT / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "interface", *[f"{metric}_mean" for metric in METRICS], *[f"{metric}_std" for metric in METRICS]]
        )
        for row in compact:
            writer.writerow(
                [
                    row["method"],
                    row["interface"],
                    *[row[metric]["mean"] for metric in METRICS],
                    *[row[metric]["std"] for metric in METRICS],
                ]
            )
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
