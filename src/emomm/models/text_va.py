"""EmoBank VA regression directly on frozen CLaMP3 text embeddings."""
from pathlib import Path

import torch
from torch import nn

INPUT_REPRESENTATION = "clamp3_text"


class TextVAHead(nn.Module):
    """A trainable linear head; coordinates are returned in [-1, 1]."""

    def __init__(self, embedding_dim: int = 768):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, 2)

    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        if text_features.ndim != 2 or text_features.shape[1] != self.linear.in_features:
            raise ValueError("TextVA expects [B, embedding_dim] CLaMP3 text features")
        return torch.tanh(self.linear(text_features))


def load_text_va(checkpoint: str | Path, clamp3_sha256: str | None = None) -> TextVAHead:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("input_representation") != INPUT_REPRESENTATION:
        raise ValueError("This TextVA checkpoint does not use direct CLaMP3 text features; retrain TextVA")
    if clamp3_sha256 is not None and payload["backbone_checkpoint_sha256"] != clamp3_sha256:
        raise ValueError("TextVA was fitted to features from a different CLaMP3 checkpoint")
    head = TextVAHead(**payload["model_config"])
    head.load_state_dict(payload["model"], strict=True)
    return head.eval().requires_grad_(False)
