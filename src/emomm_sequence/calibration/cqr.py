"""Continuous music-VA quantiles and song-level, joint-axis CQR calibration.

The intervals concern aggregate music-expression labels. They are marginal for
an exchangeable new song from the calibrated source; selection by a playlist
planner and transfer to a different catalog do not inherit that guarantee.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..models.music_va import QUANTILE_LEVELS, VAQuantileRegressor
from ..training.records import sha256


def _checked_arrays(quantiles: np.ndarray, targets: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    quantiles = np.asarray(quantiles, dtype=np.float64)
    if quantiles.ndim != 3 or quantiles.shape[1:] != (2, 3):
        raise ValueError(f"quantiles must have shape [N,2,3], got {quantiles.shape}")
    if not np.isfinite(quantiles).all() or np.any(quantiles < -1e-7) or np.any(quantiles > 1 + 1e-7):
        raise ValueError("quantiles must be finite and in [0,1]")
    if np.any(np.diff(quantiles, axis=-1) < -1e-7):
        raise ValueError("quantiles must be ordered low <= median <= high")
    if targets is None:
        return quantiles, None
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != quantiles.shape[:2] or not np.isfinite(targets).all():
        raise ValueError(f"targets must be finite [N,2], got {targets.shape}")
    if np.any(targets < 0) or np.any(targets > 1):
        raise ValueError("targets must be in [0,1]")
    return quantiles, targets


def joint_nonconformity(quantiles: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """One nonnegative score per song, jointly across V and A."""
    quantiles, targets = _checked_arrays(quantiles, targets)
    below = quantiles[:, :, 0] - targets
    above = targets - quantiles[:, :, 2]
    return np.maximum(0.0, np.maximum(below, above).max(axis=1))


def cqr_rank(n: int, alpha: float) -> int:
    """One-based split-conformal rank; reject unattainable finite coverage."""
    if n < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("n must be positive and alpha must lie in (0,1)")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        raise ValueError(f"finite CQR quantile unavailable for n={n}, alpha={alpha}")
    return rank


def fit_joint_cqr(
    quantiles: np.ndarray, targets: np.ndarray, sources: Sequence[str], alpha: float = 0.10
) -> dict:
    """Fit separate corrections by source using independent calibration songs."""
    scores = joint_nonconformity(quantiles, targets)
    sources = np.asarray(sources).astype(str)
    if sources.shape != (len(scores),):
        raise ValueError("sources must contain one entry per calibration song")
    corrections: dict[str, dict] = {}
    for source in sorted(set(sources)):
        selected = np.sort(scores[sources == source])
        rank = cqr_rank(len(selected), alpha)
        corrections[source] = {
            "n_songs": int(len(selected)),
            "rank_one_based": rank,
            "correction": float(selected[rank - 1]),
        }
    if not corrections:
        raise ValueError("calibration data are empty")
    return {
        "alpha": float(alpha),
        "quantile_levels": list(QUANTILE_LEVELS),
        "calibration_unit": "song_joint_valence_arousal",
        "score": "max(0,q05_V-y_V,y_V-q95_V,q05_A-y_A,y_A-q95_A)",
        "source_corrections": corrections,
        "default_correction": max(item["correction"] for item in corrections.values()),
        "default_scope": "maximum source correction for deployment only; no cross-domain coverage guarantee",
    }


def apply_joint_cqr(
    quantiles: np.ndarray, calibration: dict, source: str | Sequence[str] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median, lower, upper as [N,2] arrays in the music VA scale."""
    quantiles, _ = _checked_arrays(quantiles)
    n = len(quantiles)
    if source is None:
        corrections = np.full((n, 1), float(calibration["default_correction"]))
    elif isinstance(source, str):
        corrections = np.full((n, 1), float(calibration["source_corrections"][source]["correction"]))
    else:
        sources = np.asarray(source).astype(str)
        if sources.shape != (n,):
            raise ValueError("source list must contain one entry per song")
        corrections = np.array(
            [float(calibration["source_corrections"][name]["correction"]) for name in sources],
            dtype=np.float64,
        )[:, None]
    if not np.isfinite(corrections).all() or (corrections < 0).any():
        raise ValueError("CQR corrections must be finite and nonnegative")
    median = quantiles[:, :, 1]
    lower = np.maximum(0.0, quantiles[:, :, 0] - corrections)
    upper = np.minimum(1.0, quantiles[:, :, 2] + corrections)
    # Retain float64 arithmetic: rounding a fitted boundary inward to float32
    # can incorrectly mark a calibration song lying exactly on it as uncovered.
    return median, lower, upper


class VAUncertaintyModel:
    """Runtime wrapper around one trained quantile seed and its own calibration."""

    def __init__(self, model: VAQuantileRegressor, calibration: dict, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval().requires_grad_(False)
        self.calibration = calibration

    @classmethod
    def from_artifacts(
        cls, checkpoint_path: str | Path, calibration_path: str | Path, device: str | torch.device = "cpu"
    ) -> "VAUncertaintyModel":
        checkpoint_path, calibration_path = Path(checkpoint_path), Path(calibration_path)
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        expected = calibration.get("checkpoint_sha256")
        if expected is not None and sha256(checkpoint_path) != expected:
            raise ValueError("quantile checkpoint does not match CQR calibration")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = VAQuantileRegressor(**payload["model_config"])
        model.load_state_dict(payload["model"], strict=True)
        return cls(model, calibration, device)

    @torch.inference_mode()
    def predict_quantiles(self, embeddings: np.ndarray, batch_size: int = 512) -> np.ndarray:
        features = np.asarray(embeddings, dtype=np.float32)
        input_dim = self.model.network[0].normalized_shape[0]
        if features.ndim != 2 or features.shape[1] != input_dim:
            raise ValueError(f"embeddings must have shape [N,{input_dim}], got {features.shape}")
        if not np.isfinite(features).all() or batch_size < 1:
            raise ValueError("embeddings must be finite and batch_size positive")
        if len(features) == 0:
            return np.empty((0, 2, 3), dtype=np.float32)
        return np.concatenate([
            self.model(torch.from_numpy(features[i:i + batch_size]).to(self.device)).cpu().numpy()
            for i in range(0, len(features), batch_size)
        ]).astype(np.float32)

    def predict(
        self, embeddings: np.ndarray, source: str | Sequence[str] | None = None, batch_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return apply_joint_cqr(self.predict_quantiles(embeddings, batch_size), self.calibration, source)
