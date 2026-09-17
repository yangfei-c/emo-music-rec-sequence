"""Retrieval metrics with explicit support for multiple positives."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from emomm_sequence.retrieval.model import RetrieverModel
from emomm_sequence.retrieval.factorization import mean_positive_label_embedding


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


@torch.inference_mode()
def counterfactual_factor_metrics(
    model: RetrieverModel,
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    split: str,
    device: torch.device,
    batch_size: int,
    max_queries: int,
    seed: int,
) -> dict:
    """Evaluate controlled affect/preference discrimination on held-out tracks."""

    split_mask = mtg["splits"].astype(str) == split
    preference_targets = mtg["preference_targets"][split_mask].astype(bool)
    affect_targets = mtg["affect_targets"][split_mask].astype(bool)
    valid = preference_targets.any(axis=1) & affect_targets.any(axis=1)
    available = np.flatnonzero(valid)
    if len(available) == 0:
        raise ValueError(f"no joint affect/preference queries in MTG {split}")
    rng = np.random.default_rng(seed)
    query_indices = (
        np.sort(rng.choice(available, size=max_queries, replace=False))
        if len(available) > max_queries
        else available
    )
    raw_music = music_index[mtg["index_positions"][split_mask]].astype(np.float32)
    model.to(device).eval()
    preference_labels = model.encode_factor("text", "preference",
        torch.from_numpy(mtg["preference_label_embeddings"].astype(np.float32)).to(device)
    )
    affect_labels = model.encode_factor("text", "affect",
        torch.from_numpy(mtg["affect_label_embeddings"].astype(np.float32)).to(device)
    )
    music_affect, music_preference = [], []
    for start in range(0, len(raw_music), batch_size):
        value = torch.from_numpy(raw_music[start : start + batch_size]).to(device)
        factors = model.encode_music(value)
        music_affect.append(factors["affect"].cpu())
        music_preference.append(factors["preference"].cpu())
    music_affect_tensor = torch.cat(music_affect).to(device)
    music_preference_tensor = torch.cat(music_preference).to(device)
    query_preference, _ = mean_positive_label_embedding(
        torch.from_numpy(preference_targets[query_indices]).to(device), preference_labels
    )
    query_affect, _ = mean_positive_label_embedding(
        torch.from_numpy(affect_targets[query_indices]).to(device), affect_labels
    )
    weights = model.factor_weights()[1:]
    weights = weights / weights.sum()
    scores = (
        weights[0] * (query_affect @ music_affect_tensor.T)
        + weights[1] * (query_preference @ music_preference_tensor.T)
    ).float().cpu().numpy()
    shared_preference = (
        preference_targets[query_indices].astype(np.int16)
        @ preference_targets.astype(np.int16).T
    ) > 0
    shared_affect = (
        affect_targets[query_indices].astype(np.int16) @ affect_targets.astype(np.int16).T
    ) > 0
    positives = shared_preference & shared_affect
    positives[np.arange(len(query_indices)), query_indices] = False
    affect_negatives = shared_preference & ~shared_affect
    preference_negatives = shared_affect & ~shared_preference

    def aucs(negatives: np.ndarray) -> list[float]:
        values = []
        for row in range(len(query_indices)):
            include = positives[row] | negatives[row]
            labels = positives[row, include].astype(np.int8)
            if np.unique(labels).size < 2:
                continue
            values.append(float(roc_auc_score(labels, scores[row, include])))
        return values

    affect_auc = aucs(affect_negatives)
    preference_auc = aucs(preference_negatives)
    return {
        "requested_queries": int(max_queries),
        "sampled_queries": int(len(query_indices)),
        "affect_control_queries": len(affect_auc),
        "preference_control_queries": len(preference_auc),
        "affect_control_auc": float(np.mean(affect_auc)) if affect_auc else float("nan"),
        "preference_control_auc": float(np.mean(preference_auc)) if preference_auc else float("nan"),
    }
