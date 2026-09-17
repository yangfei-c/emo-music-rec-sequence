"""Inference loader for the same Ours model used by training."""

from __future__ import annotations

from pathlib import Path

import torch
from emomm_sequence.retrieval.model import FACTORS, ResidualAdapter, RetrieverModel


class FrozenRetriever(RetrieverModel):
    def __init__(self, checkpoint: str | Path, embedding_dim: int = 768, bottleneck_dim: int = 64, dropout: float = 0.1) -> None:
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        config = payload.get("model_config", {})
        super().__init__(
            embedding_dim=embedding_dim,
            bottleneck_dim=int(config.get("bottleneck_dim", bottleneck_dim)),
            dropout=float(config.get("dropout", dropout)),
            adapter_drop_probability=float(
                config.get("adapter_drop_probability", 0.0)
            ),
            adapter_drop_probabilities=config.get("adapter_drop_probabilities"),
            initial_adapter_scale=float(config.get("initial_adapter_scale", 0.1)),
            initial_factor_weights=tuple(
                float(value)
                for value in config.get("initial_factor_weights", (0.34, 0.33, 0.33))
            ),
            initial_temperature=float(config.get("initial_temperature", 0.07)),
            train_factor_weights=bool(config.get("train_factor_weights", False)),
            adapter_sharing=str(config.get("adapter_sharing", "none")),
            adapter_kind=str(config.get("adapter_kind", "residual")),
            adapter_blend=float(config.get("adapter_blend", 0.2)),
            semantic_output_blend=float(config.get("semantic_output_blend", 1.0)),
            affect_output_blend=float(config.get("affect_output_blend", 1.0)),
            preference_output_blend=float(config.get("preference_output_blend", 1.0)),
        )
        self.load_state_dict(payload.get("model", payload), strict=True)
        self.requires_grad_(False)
        self.eval()

    def encode_text(self, factor: str, features: torch.Tensor) -> torch.Tensor:
        return self.encode_factor("text", factor, features)

    def predict_text_va(self, features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.text_va_head(self.text_adapters["affect"](features)))
