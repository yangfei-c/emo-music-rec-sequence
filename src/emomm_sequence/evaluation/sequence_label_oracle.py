"""Evaluator-only exact VA-position references; never a recommendation policy."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def positional_va_oracles(va, target_path, start, target):
    """Exact minimum position MAE, with and without the policy's fixed endpoints.

    Each row is a distinct audio-content identity. This objective is a linear
    assignment, so the rectangular assignment solver enforces no replacement.
    It deliberately excludes preference and transition costs. Its optimum is
    not an optimum of the Frozen or robust-VA playlist objective.
    """
    va = np.asarray(va, dtype=np.float64)
    target_path = np.asarray(target_path, dtype=np.float64)
    if va.ndim != 2 or va.shape[1] != 2 or target_path.ndim != 2 or target_path.shape[1] != 2:
        raise ValueError("expected song and position arrays with two VA coordinates")
    n, m = len(target_path), len(va)
    if n < 2 or m < n or start == target or not (0 <= start < m and 0 <= target < m):
        raise ValueError("need distinct valid endpoints and at least N unique songs")
    if not np.isfinite(va).all() or not np.isfinite(target_path).all():
        raise ValueError("VA oracle requires finite labels and targets")
    if min(va.min(), target_path.min()) < -1e-7 or max(va.max(), target_path.max()) > 1 + 1e-7:
        raise ValueError("VA oracle expects the declared [0,1] label scale")
    costs = np.abs(target_path[:, None] - va[None]).mean(axis=2)
    positions, free_path = linear_sum_assignment(costs)
    if not np.array_equal(positions, np.arange(n)):
        raise AssertionError("assignment did not cover every position")
    interior = np.array([i for i in range(m) if i not in (start, target)], dtype=int)
    fixed_path = np.array([start, target], dtype=int)
    if n > 2:
        _, selected = linear_sum_assignment(costs[1:-1, interior])
        fixed_path = np.r_[start, interior[selected], target]
    free_cost = float(costs[np.arange(n), free_path].mean())
    fixed_cost = float(costs[np.arange(n), fixed_path].mean())
    if fixed_cost < free_cost - 1e-10:
        raise AssertionError("constrained assignment below unconstrained optimum")
    return {"free_endpoint_min_va_mae": free_cost, "fixed_endpoint_min_va_mae": fixed_cost,
            "fixed_endpoint_va_penalty": fixed_cost - free_cost,
            "free_endpoint_path_indices": free_path.tolist(), "fixed_endpoint_path_indices": fixed_path.tolist(),
            "certified_for": "unique_song_position_VA_MAE_only", "uses_true_labels": True}
