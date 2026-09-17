"""Frozen CLaMP3 text encoder used by the production recommender."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer


def _prefixed_state(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    selected = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(f"CLaMP3 checkpoint lacks {prefix!r}")
    return selected


class FrozenTextEncoder(nn.Module):
    """Load only the text branch required at inference time."""

    def __init__(self, checkpoint: str | Path, text_model: str) -> None:
        super().__init__()
        self.text_model_name = str(text_model)
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.projection = nn.Linear(self.text_model.config.hidden_size, 768)
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        state = payload.get("model")
        if not isinstance(state, dict):
            raise ValueError("invalid CLaMP3 checkpoint")
        self.text_model.load_state_dict(_prefixed_state(state, "text_model."), strict=True)
        self.projection.load_state_dict(_prefixed_state(state, "text_proj."), strict=True)
        self.requires_grad_(False)
        self.eval()

    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.text_model_name)

    @torch.inference_mode()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        weights = attention_mask.unsqueeze(-1).to(hidden.dtype)
        count = weights.sum(dim=1)
        if (count == 0).any().item():
            raise ValueError("empty tokenized affect or preference text")
        pooled = (hidden * weights).sum(dim=1) / count
        return F.normalize(self.projection(pooled), dim=-1)
