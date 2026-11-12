"""Ordered music valence/arousal quantile regressor."""

from __future__ import annotations

import torch
from torch import nn


QUANTILE_LEVELS = (0.05, 0.50, 0.95)


class VAQuantileRegressor(nn.Module):
    """Return ordered [low, median, high] quantiles for V and A in [0, 1]."""

    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        raw = self.network(embeddings).reshape(*embeddings.shape[:-1], 2, 3)
        raw_low, raw_median, raw_high = raw.unbind(dim=-1)
        median = torch.sigmoid(raw_median)
        low = median * torch.sigmoid(raw_low)
        high = median + (1.0 - median) * torch.sigmoid(raw_high)
        return torch.stack((low, median, high), dim=-1)
