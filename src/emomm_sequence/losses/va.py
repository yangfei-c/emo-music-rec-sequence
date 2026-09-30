"""Loss used by the Music-VA and cross-modal affect heads."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..models.music_va import QUANTILE_LEVELS


def pinball_loss(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss across songs, both VA axes, and three quantiles."""
    if quantiles.ndim != 3 or quantiles.shape[1:] != (2, 3):
        raise ValueError(f"quantiles must have shape [N,2,3], got {tuple(quantiles.shape)}")
    if targets.shape != quantiles.shape[:2]:
        raise ValueError(f"targets must have shape [N,2], got {tuple(targets.shape)}")
    levels = quantiles.new_tensor(QUANTILE_LEVELS)
    residual = targets.unsqueeze(-1) - quantiles
    return torch.maximum(levels * residual, (levels - 1.0) * residual).mean()


class VALoss(nn.Module):
    """Axis-balanced Smooth-L1 with an optional batch-CCC penalty."""

    def __init__(self, beta: float = 0.1, ccc_weight: float = 0.0, eps: float = 1.0e-8) -> None:
        super().__init__()
        if beta < 0 or ccc_weight < 0 or eps <= 0:
            raise ValueError("invalid VA loss parameter")
        self.beta, self.ccc_weight, self.eps = beta, ccc_weight, eps

    def ccc(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prediction_mean, target_mean = predictions.mean(0), targets.mean(0)
        prediction_centered = predictions - prediction_mean
        target_centered = targets - target_mean
        covariance = (prediction_centered * target_centered).mean(0)
        denominator = (
            prediction_centered.square().mean(0)
            + target_centered.square().mean(0)
            + (prediction_mean - target_mean).square()
        )
        return (2.0 * covariance / denominator.clamp_min(self.eps)).clamp(-1.0, 1.0)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError(
                f"predictions and targets must both be [B,2], got {tuple(predictions.shape)}/{tuple(targets.shape)}"
            )
        predictions, targets = predictions.float(), targets.float()
        huber = F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction="none").mean(0)
        ccc_penalty = 1.0 - self.ccc(predictions, targets)
        axis = huber + self.ccc_weight * ccc_penalty
        total = axis.mean()
        if not return_components:
            return total
        return {
            "total": total,
            "valence": axis[0],
            "arousal": axis[1],
            "huber": huber.mean(),
            "ccc_penalty": ccc_penalty.mean(),
        }
