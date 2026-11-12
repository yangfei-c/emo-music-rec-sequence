"""Six residual adapters: text/music × semantic/affect/preference."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

FACTORS = ("semantic", "affect", "preference")


class FrozenCLaMP3Retriever(nn.Module):
    """Zero-shot reference: normalized cached backbone features, no Adapter."""

    def encode_factor(self, modality, factor, features):
        if factor not in FACTORS or modality not in {"text", "music", "audio"}:
            raise ValueError("Unknown retrieval modality or factor")
        return F.normalize(features.float(), dim=-1)


class ResidualAdapter(nn.Module):
    def __init__(self, embedding_dim=768, bottleneck_dim=64, dropout=0.1, initial_scale=0.1):
        super().__init__()
        if min(embedding_dim, bottleneck_dim) <= 0 or not 0 <= dropout < 1 or initial_scale < 0:
            raise ValueError("Invalid adapter dimensions, dropout or scale")
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.norm.normalized_shape[0]:
            raise ValueError(f"Adapter expects [B,D], got {tuple(features.shape)}")
        residual = self.up(self.dropout(F.gelu(self.down(self.norm(features)))))
        # AMP linear layers return FP16. Keep the learned scalar multiplication
        # in FP32 so its batch/channel gradient reduction cannot overflow FP16.
        return F.normalize(features.float() + self.scale * residual.float(), dim=-1)


class RetrieverModel(nn.Module):
    """Trainable S/A/P spaces; VA heads are trained in separate modules."""
    def __init__(self, embedding_dim=768, bottleneck_dim=64, dropout=0.1,
                 initial_adapter_scale=0.1, initial_temperature=0.07,
                 semantic_output_blend=1.0, affect_output_blend=1.0, preference_output_blend=1.0):
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("Temperature must be positive")
        for factor, blend in zip(FACTORS, (semantic_output_blend, affect_output_blend, preference_output_blend)):
            if not 0 <= blend <= 1:
                raise ValueError("Output blend must be in [0,1]")
            setattr(self, f"{factor}_output_blend", float(blend))
        self.embedding_dim = int(embedding_dim)
        def adapters():
            return nn.ModuleDict({f: ResidualAdapter(embedding_dim, bottleneck_dim, dropout, initial_adapter_scale)
                                  for f in FACTORS})
        self.text_adapters = adapters()
        self.music_adapters = adapters()
        scale = torch.tensor(float(1 / initial_temperature)).log()
        self.logit_scales = nn.ParameterDict({f: nn.Parameter(scale.clone()) for f in FACTORS})

    def encode_factor(self, modality, factor, features):
        if factor not in FACTORS:
            raise ValueError(f"Unknown factor: {factor}")
        if modality == "text":
            adapter = self.text_adapters[factor]
        elif modality in {"music", "audio"}:
            adapter = self.music_adapters[factor]
        else:
            raise ValueError(f"Unknown modality: {modality}")
        adapted = adapter(features)
        blend = getattr(self, f"{factor}_output_blend")
        if blend == 1:
            return adapted
        return F.normalize((1 - blend) * F.normalize(features, dim=-1) + blend * adapted, dim=-1)

    def encode_text(self, features):
        return {f: self.encode_factor("text", f, features) for f in FACTORS}

    def encode_music(self, features):
        return {f: self.encode_factor("music", f, features) for f in FACTORS}

    def factor_similarity(self, factor, text_features, music_features, *, scaled=False):
        text = self.encode_factor("text", factor, text_features)
        music = self.encode_factor("music", factor, music_features)
        with torch.amp.autocast(device_type=text.device.type, enabled=False):
            score = text.float() @ music.float().T
            return score * self.logit_scales[factor].exp().clamp(max=100) if scaled else score


def load_retriever(checkpoint, freeze=True):
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    model = RetrieverModel(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.eval().requires_grad_(not freeze)
