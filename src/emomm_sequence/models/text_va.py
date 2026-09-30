"""EmoBank VA calibration on the selected, frozen text-affect representation."""
from pathlib import Path

import torch
from torch import nn


class TextVAHead(nn.Module):
    """A trainable linear head; coordinates are returned in [-1, 1]."""

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, affect_features: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(affect_features))


def load_text_va(checkpoint: str | Path, retriever_sha256: str) -> TextVAHead:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["retriever_checkpoint_sha256"] != retriever_sha256:
        raise ValueError("Text VA was fitted to a different affect Adapter")
    head = TextVAHead(**payload["model_config"])
    head.load_state_dict(payload["model"], strict=True)
    return head.eval().requires_grad_(False)
