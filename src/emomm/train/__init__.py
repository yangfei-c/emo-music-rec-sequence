"""Reproducible training records and data-partition checks."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
import random
import yaml

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
        "initialization": "fresh_random_with_zero_output_residual_adapters",
        "initial_checkpoint": None, "initial_state_sha256": state_sha256(model),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "backbone_optimized": False,
    }


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return config

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import math
def retrieval_selection_score(pair_mrr: float, preference_ndcg: float, affect_ndcg: float) -> float:
    """Give the three retrieval tasks equal weight; never use auxiliary VA scores."""
    values = [float(pair_mrr), float(preference_ndcg), float(affect_ndcg)]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Nonfinite validation retrieval score")
    return sum(values) / len(values)


def tag_balanced_selection_score(ndcg_at_10: float, average_precision: float) -> float:
    """Balance early-rank quality and global ranking with a geometric mean."""

    values = [float(ndcg_at_10), float(average_precision)]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Tag validation metrics must be finite and non-negative")
    return math.sqrt(values[0] * values[1])

def _summary(ranks: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or len(ranks) == 0 or (ranks < 1).any():
        raise ValueError("retrieval ranks must be a non-empty positive vector")
    result = {
        "queries": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    result.update({f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks})
    return result


def grouped_retrieval_metrics(
    text_embeddings: np.ndarray,
    music_embeddings: np.ndarray,
    identities: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    text = np.asarray(text_embeddings, dtype=np.float64)
    music = np.asarray(music_embeddings, dtype=np.float64)
    identities = np.asarray(identities).astype(str)
    if text.shape != music.shape or text.ndim != 2 or len(text) != len(identities):
        raise ValueError("paired embeddings/identities have inconsistent shapes")
    text /= np.linalg.norm(text, axis=1, keepdims=True).clip(min=1.0e-12)
    music /= np.linalg.norm(music, axis=1, keepdims=True).clip(min=1.0e-12)
    unique, first = np.unique(identities, return_index=True)
    order = np.argsort(first)
    unique, first = unique[order], first[order]
    unique_music = music[first]
    identity_to_music = {identity: index for index, identity in enumerate(unique)}

    text_scores = text @ unique_music.T
    correct = np.asarray([identity_to_music[value] for value in identities])
    correct_scores = text_scores[np.arange(len(text)), correct]
    text_ranks = 1 + (text_scores > correct_scores[:, None]).sum(axis=1)

    music_scores = unique_music @ text.T
    music_ranks = []
    for row, identity in enumerate(unique):
        relevant = identities == identity
        best = music_scores[row, relevant].max()
        music_ranks.append(1 + int((music_scores[row] > best).sum()))
    music_ranks = np.asarray(music_ranks)
    positive = np.sum(text * music, axis=1)
    return {
        "captions": int(len(text)),
        "tracks": int(len(unique)),
        "text_to_music": _summary(text_ranks, ks),
        "music_to_text": _summary(music_ranks, ks),
        "mean_positive_cosine": float(positive.mean()),
        "std_positive_cosine": float(positive.std(ddof=1)) if len(positive) > 1 else 0.0,
    }



def _dcg(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2, dtype=np.float64))
    return float(np.sum(relevance.astype(np.float64) / discounts))

def label_to_track_metrics(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Macro retrieval metrics for label queries with many relevant tracks."""

    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive_mask must share a [labels, tracks] shape")
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must be positive")
    per_query = []
    for query in range(scores.shape[0]):
        truth = positive_mask[query]
        positives = int(truth.sum())
        if positives == 0:
            continue
        order = np.argsort(-scores[query], kind="stable")
        ranked = truth[order]
        positive_ranks = np.flatnonzero(ranked) + 1
        precisions = np.cumsum(ranked)[positive_ranks - 1] / positive_ranks
        row = {
            "query_index": query,
            "positives": positives,
            "average_precision": float(precisions.mean()),
            "reciprocal_rank": float(1.0 / positive_ranks[0]),
        }
        for k in ks:
            cutoff = min(int(k), len(ranked))
            hits = int(ranked[:cutoff].sum())
            ideal = np.ones(min(positives, cutoff), dtype=bool)
            row[f"precision_at_{k}"] = float(hits / cutoff)
            row[f"recall_at_{k}"] = float(hits / positives)
            row[f"ndcg_at_{k}"] = float(_dcg(ranked[:cutoff]) / max(_dcg(ideal), 1.0e-12))
        per_query.append(row)
    if not per_query:
        raise ValueError("no label has a positive track in this split")
    keys = [key for key in per_query[0] if key not in {"query_index", "positives"}]
    macro = {key: float(np.mean([row[key] for row in per_query])) for key in keys}
    return {
        "queries": len(per_query),
        "tracks": int(scores.shape[1]),
        "mean_positives_per_query": float(np.mean([row["positives"] for row in per_query])),
        "macro": macro,
        "per_query": per_query,
    }

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def _to_numpy(value) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _check_arrays(predictions, targets) -> tuple[np.ndarray, np.ndarray]:
    predictions = _to_numpy(predictions).astype(np.float64)
    targets = _to_numpy(targets).astype(np.float64)
    if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
        raise ValueError(f"VA predictions/targets must be [N,2], got {predictions.shape}/{targets.shape}")
    if len(predictions) < 2 or not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError("VA arrays need at least two finite samples")
    return predictions, targets


def concordance_correlation_coefficient(
    targets: np.ndarray, predictions: np.ndarray, eps: float = 1.0e-12
) -> float:
    target_mean, prediction_mean = targets.mean(), predictions.mean()
    target_var = np.mean((targets - target_mean) ** 2)
    prediction_var = np.mean((predictions - prediction_mean) ** 2)
    covariance = np.mean((targets - target_mean) * (predictions - prediction_mean))
    denominator = target_var + prediction_var + (target_mean - prediction_mean) ** 2
    if denominator <= eps:
        return 1.0 if np.allclose(targets, predictions, atol=eps, rtol=0.0) else 0.0
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))


def _pearson(targets: np.ndarray, predictions: np.ndarray, eps: float = 1.0e-12) -> float:
    if targets.std() <= eps or predictions.std() <= eps:
        return 1.0 if np.allclose(targets, predictions, atol=eps, rtol=0.0) else 0.0
    return float(np.clip(np.corrcoef(targets, predictions)[0, 1], -1.0, 1.0))


def compute_va_metrics(predictions, targets) -> dict[str, float]:
    predictions, targets = _check_arrays(predictions, targets)
    ccc = [concordance_correlation_coefficient(targets[:, axis], predictions[:, axis]) for axis in range(2)]
    pearson = [_pearson(targets[:, axis], predictions[:, axis]) for axis in range(2)]
    rmse = np.sqrt(mean_squared_error(targets, predictions, multioutput="raw_values"))
    mae = mean_absolute_error(targets, predictions, multioutput="raw_values")
    r2 = r2_score(targets, predictions, multioutput="raw_values", force_finite=True)
    return {
        "valence_ccc": ccc[0], "arousal_ccc": ccc[1], "va_score": float(np.mean(ccc)),
        "valence_pearson": pearson[0], "arousal_pearson": pearson[1], "mean_pearson": float(np.mean(pearson)),
        "valence_r2": float(r2[0]), "arousal_r2": float(r2[1]), "mean_r2": float(np.mean(r2)),
        "valence_rmse": float(rmse[0]), "arousal_rmse": float(rmse[1]), "mean_rmse": float(rmse.mean()),
        "valence_mae": float(mae[0]), "arousal_mae": float(mae[1]), "mean_mae": float(mae.mean()),
    }
