"""Explicit validation-only checkpoint selection for controlled experiments."""

from __future__ import annotations

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
