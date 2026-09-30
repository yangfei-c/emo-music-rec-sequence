"""Valence-arousal evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch
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
