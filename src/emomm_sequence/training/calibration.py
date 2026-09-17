"""Validation-only temperature calibration for relation probabilities."""

from __future__ import annotations

import numpy as np
import torch


def multiclass_nll(log_probs: np.ndarray, labels: np.ndarray) -> float:
    rows = np.arange(len(labels))[:, None]
    axes = np.arange(labels.shape[1])[None, :]
    return float(-np.mean(log_probs[rows, axes, labels]))


def fit_axis_temperatures(
    log_prob_factory,
    deltas: torch.Tensor,
    labels: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    """Select V and A temperatures independently on calibration songs."""

    if grid is None:
        grid = np.exp(np.linspace(np.log(0.10), np.log(10.0), 121))
    best = np.ones(2, dtype=np.float64)
    for axis in range(2):
        scores = []
        for value in grid:
            temperature = torch.ones(2, dtype=deltas.dtype, device=deltas.device)
            temperature[axis] = float(value)
            with torch.inference_mode():
                log_probs = log_prob_factory(deltas, temperature).detach().cpu().numpy()
            chosen = log_probs[:, axis, :]
            scores.append(float(-np.mean(chosen[np.arange(len(labels)), labels[:, axis]])))
        best[axis] = float(grid[int(np.argmin(scores))])
    return best


def calibration_bins(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 10
) -> dict[str, list[dict[str, float | int]]]:
    """Return separate reliability bins for valence and arousal."""

    edges = np.linspace(0.0, 1.0, bins + 1)
    result: dict[str, list[dict[str, float | int]]] = {}
    for axis, name in enumerate(("valence", "arousal")):
        confidence = probabilities[:, axis].max(axis=-1)
        prediction = probabilities[:, axis].argmax(axis=-1)
        truth = labels[:, axis]
        axis_result = []
        for index in range(bins):
            mask = (confidence >= edges[index]) & (
                confidence <= edges[index + 1]
                if index == bins - 1
                else confidence < edges[index + 1]
            )
            if not mask.any():
                continue
            axis_result.append(
                {
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "count": int(mask.sum()),
                    "mean_confidence": float(confidence[mask].mean()),
                    "accuracy": float((prediction[mask] == truth[mask]).mean()),
                }
            )
        result[name] = axis_result
    return result
