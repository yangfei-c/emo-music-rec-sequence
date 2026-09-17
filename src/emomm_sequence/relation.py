"""Frozen calibrated music-relation ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class RegressionRelation(nn.Module):
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.network = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def coordinates(self, music: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder.network(music))


def _log_cdf_difference(upper: torch.Tensor, lower: torch.Tensor) -> torch.Tensor:
    log_upper = F.logsigmoid(upper)
    log_lower = F.logsigmoid(lower)
    ratio = torch.exp(log_lower - log_upper).clamp(max=1.0 - 1.0e-7)
    return log_upper + torch.log1p(-ratio)


def relation_log_probabilities(
    delta: torch.Tensor,
    small_bound: torch.Tensor,
    big_bound: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    z0 = (-big_bound - delta) / temperature
    z1 = (-small_bound - delta) / temperature
    z2 = (small_bound - delta) / temperature
    z3 = (big_bound - delta) / temperature
    return torch.stack(
        (
            F.logsigmoid(z0),
            _log_cdf_difference(z1, z0),
            _log_cdf_difference(z2, z1),
            _log_cdf_difference(z3, z2),
            F.logsigmoid((delta - big_bound) / temperature),
        ),
        dim=-1,
    )


class RelationEnsemble:
    def __init__(self, root: Path, config: dict, device: torch.device) -> None:
        self.device = device
        self.models: list[RegressionRelation] = []
        self.calibrations: list[dict] = []
        for seed in config["seeds"]:
            checkpoint = root / config["checkpoint_template"].format(seed=seed)
            calibration = root / config["calibration_template"].format(seed=seed)
            payload = torch.load(checkpoint, map_location=device, weights_only=True)
            model = RegressionRelation(
                hidden_dim=int(config["hidden_dim"]),
                dropout=float(config["dropout"]),
            ).to(device)
            model.load_state_dict(payload.get("model", payload), strict=True)
            model.requires_grad_(False)
            self.models.append(model.eval())
            self.calibrations.append(json.loads(calibration.read_text(encoding="utf-8")))

    @torch.inference_mode()
    def probabilities(self, embeddings: np.ndarray) -> np.ndarray:
        features = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(self.device)
        outputs = []
        for model, calibration in zip(self.models, self.calibrations, strict=True):
            coordinates = model.coordinates(features)
            delta = coordinates[None, :, :] - coordinates[:, None, :]
            temperature = torch.tensor(
                [calibration["temperature_v"], calibration["temperature_a"]],
                device=self.device,
            )
            small = torch.tensor(calibration["fixed_small_bounds"], device=self.device)
            big = torch.tensor(calibration["fixed_big_bounds"], device=self.device)
            outputs.append(relation_log_probabilities(delta, small, big, temperature).exp().cpu().numpy())
        return np.stack(outputs).astype(np.float32)
