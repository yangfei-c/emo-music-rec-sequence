"""Three-branch text/music retriever (Ours).

Ours treats CLaMP3 as a frozen feature extractor rather than as the final
retriever.  Small residual adapters learn three task-specific spaces while
preserving the pretrained geometry at initialization:

* semantic: open-vocabulary audio--caption correspondence;
* affect: affective meaning and continuous VA supervision;
* preference: genre/instrument preference correspondence.

The implementation in this file is original to EmoMM-Sequence.  The external
retrieval repositories under ``third_party/research`` are study material and
are deliberately not imported here.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from emomm_sequence.retrieval.segments import query_conditioned_segment_similarity


FACTORS = ("semantic", "affect", "preference")


class ResidualAdapter(nn.Module):
    """A zero-initialized bottleneck adapter with an identity residual path."""

    def __init__(
        self,
        embedding_dim: int = 768,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        adapter_drop_probability: float = 0.0,
        initial_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if (
            not 0.0 <= dropout < 1.0
            or not 0.0 <= adapter_drop_probability < 1.0
            or initial_scale < 0.0
        ):
            raise ValueError("invalid adapter dropout/scale")
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.adapter_drop_probability = float(adapter_drop_probability)
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.norm.normalized_shape[0]:
            raise ValueError(f"adapter expects [B,D], got {tuple(features.shape)}")
        residual = self.up(self.dropout(F.gelu(self.down(self.norm(features)))))
        if self.training and self.adapter_drop_probability > 0.0:
            keep = (
                torch.rand((), device=residual.device)
                >= self.adapter_drop_probability
            ).to(residual.dtype)
            residual = residual * keep / (1.0 - self.adapter_drop_probability)
        return F.normalize(features + self.scale * residual, dim=-1)


class ClipStyleAdapter(nn.Module):
    """CLIP-Adapter feature block transferred to music/text (not a paper replication).

    Two bias-free linear layers, ReLU after each, fixed residual blending.
    Original reference: https://arxiv.org/abs/2110.04544 and the authors'
    gaopengcuhk/CLIP-Adapter/clip_adapter.py. Our experiment changes modalities,
    supervision and optimizer, and enables both towers, explicitly documented.
    """

    def __init__(self, embedding_dim: int, bottleneck_dim: int, blend: float = 0.2):
        super().__init__()
        if embedding_dim <= 0 or bottleneck_dim <= 0 or not 0.0 <= blend <= 1.0:
            raise ValueError("Invalid CLIP-style adapter dimensions or blend")
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, bottleneck_dim, bias=False), nn.ReLU(),
            nn.Linear(bottleneck_dim, embedding_dim, bias=False), nn.ReLU(),
        )
        self.blend = float(blend)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize((1.0 - self.blend) * features + self.blend * self.fc(features), dim=-1)


class RetrieverModel(nn.Module):
    """Factorized trainable retrieval head over frozen CLaMP3 embeddings."""

    def __init__(
        self,
        embedding_dim: int = 768,
        bottleneck_dim: int = 64,
        dropout: float = 0.1,
        adapter_drop_probability: float = 0.0,
        adapter_drop_probabilities: Mapping[str, float] | None = None,
        initial_adapter_scale: float = 0.1,
        initial_factor_weights: tuple[float, float, float] = (0.34, 0.33, 0.33),
        initial_temperature: float = 0.07,
        train_factor_weights: bool = False,
        adapter_sharing: str = "none",
        adapter_kind: str = "residual",
        adapter_blend: float = 0.2,
        semantic_output_blend: float = 1.0,
        affect_output_blend: float = 1.0,
        preference_output_blend: float = 1.0,
    ) -> None:
        super().__init__()
        if len(initial_factor_weights) != len(FACTORS):
            raise ValueError("one initial weight is required per Ours factor")
        weights = torch.as_tensor(initial_factor_weights, dtype=torch.float32)
        if (weights <= 0).any().item() or initial_temperature <= 0:
            raise ValueError("factor weights and temperature must be positive")
        if adapter_sharing not in {"none", "across_factors"}:
            raise ValueError("adapter_sharing must be 'none' or 'across_factors'")
        factor_drop_probabilities = {
            factor: float(adapter_drop_probability) for factor in FACTORS
        }
        if adapter_drop_probabilities is not None:
            unknown_factors = set(adapter_drop_probabilities).difference(FACTORS)
            if unknown_factors:
                raise ValueError(
                    "unknown adapter-drop factors: "
                    + ", ".join(sorted(unknown_factors))
                )
            factor_drop_probabilities.update(
                {
                    factor: float(probability)
                    for factor, probability in adapter_drop_probabilities.items()
                }
            )
        if any(
            not 0.0 <= probability < 1.0
            for probability in factor_drop_probabilities.values()
        ):
            raise ValueError("adapter-drop probabilities must be in [0, 1)")
        if (
            adapter_sharing == "across_factors"
            and len(set(factor_drop_probabilities.values())) != 1
        ):
            raise ValueError(
                "factor-specific adapter drop requires independent factor adapters"
            )
        output_blends = {
            "semantic": float(semantic_output_blend),
            "affect": float(affect_output_blend),
            "preference": float(preference_output_blend),
        }
        if any(not 0.0 <= value <= 1.0 for value in output_blends.values()):
            raise ValueError("factor output blends must be in [0, 1]")
        if adapter_kind not in {"residual", "clip_style"}:
            raise ValueError(f"Unknown adapter kind: {adapter_kind}")

        def make_adapter(factor: str) -> nn.Module:
            if adapter_kind == "clip_style":
                return ClipStyleAdapter(embedding_dim, bottleneck_dim, adapter_blend)
            return ResidualAdapter(
                embedding_dim=embedding_dim,
                bottleneck_dim=bottleneck_dim,
                dropout=dropout,
                adapter_drop_probability=factor_drop_probabilities[factor],
                initial_scale=initial_adapter_scale,
            )

        def make_adapters() -> nn.ModuleDict:
            if adapter_sharing == "across_factors":
                shared = make_adapter(FACTORS[0])
                return nn.ModuleDict({factor: shared for factor in FACTORS})
            return nn.ModuleDict(
                {factor: make_adapter(factor) for factor in FACTORS}
            )

        self.text_adapters = make_adapters()
        self.music_adapters = make_adapters()
        self.text_va_head = nn.Linear(embedding_dim, 2)
        self.music_va_head = nn.Linear(embedding_dim, 2)
        self.factor_logits = nn.Parameter(weights.log(), requires_grad=bool(train_factor_weights))
        initial_logit_scale = torch.tensor(float(1.0 / initial_temperature)).log()
        self.logit_scales = nn.ParameterDict(
            {factor: nn.Parameter(initial_logit_scale.clone()) for factor in FACTORS}
        )
        self.embedding_dim = int(embedding_dim)
        self.adapter_sharing = adapter_sharing
        self.adapter_kind = adapter_kind
        self.adapter_drop_probabilities = factor_drop_probabilities
        self.semantic_output_blend = output_blends["semantic"]
        self.affect_output_blend = output_blends["affect"]
        self.preference_output_blend = output_blends["preference"]

    def encode_factor(
        self, modality: str, factor: str, features: torch.Tensor
    ) -> torch.Tensor:
        """Encode one factor, optionally ensembling frozen and adapted features.

        A blend below one preserves part of the frozen CLaMP3 geometry.  Each
        factor's fixed blend is selected only on its validation task.
        """

        if factor not in FACTORS:
            raise ValueError(f"unknown Ours factor: {factor}")
        if modality == "text":
            adapter = self.text_adapters[factor]
        elif modality in {"music", "audio"}:
            adapter = self.music_adapters[factor]
        else:
            raise ValueError(f"unknown Ours modality: {modality}")
        adapted = adapter(features)
        alpha = float(getattr(self, f"{factor}_output_blend"))
        if alpha >= 1.0:
            return adapted
        frozen = F.normalize(features, dim=-1)
        return F.normalize((1.0 - alpha) * frozen + alpha * adapted, dim=-1)

    def _encode(
        self, modality: str, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        return {
            factor: self.encode_factor(modality, factor, features)
            for factor in FACTORS
        }

    def encode_text(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._encode("text", features)

    def encode_music(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._encode("music", features)

    def predict_va(self, modality: str, features: torch.Tensor) -> torch.Tensor:
        encoded = self._encode(modality, features)["affect"]
        head = self.text_va_head if modality == "text" else self.music_va_head
        return torch.tanh(head(encoded))

    def factor_weights(self) -> torch.Tensor:
        return F.softmax(self.factor_logits, dim=0)

    def factor_similarity(
        self,
        factor: str,
        text_features: torch.Tensor,
        music_features: torch.Tensor,
        *,
        scaled: bool = False,
    ) -> torch.Tensor:
        if factor not in FACTORS:
            raise ValueError(f"unknown Ours factor: {factor}")
        text = self.encode_factor("text", factor, text_features)
        music = self.encode_factor("music", factor, music_features)
        similarity = text @ music.T
        if scaled:
            similarity = similarity * self.logit_scales[factor].exp().clamp(max=100.0)
        return similarity

    def semantic_similarity(
        self,
        text_features: torch.Tensor,
        music_features: torch.Tensor,
        *,
        music_segments: torch.Tensor | None = None,
        music_segment_mask: torch.Tensor | None = None,
        local_weight: float = 0.0,
        segment_temperature: float = 0.07,
        segment_top_k: int | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Combine global CLaMP3 matching with dense query-to-window matching.

        ``local_weight=0`` is exactly the historical global semantic path, so
        existing checkpoints and deployments remain compatible.  The dense
        branch has no new trainable parameters: it reuses the semantic text and
        music adapters and is tuned only on the validation split.
        """

        if not 0.0 <= local_weight <= 1.0:
            raise ValueError("local_weight must be in [0, 1]")
        global_score = self.factor_similarity(
            "semantic", text_features, music_features, scaled=False
        )
        components: dict[str, torch.Tensor] = {"global": global_score}
        if local_weight == 0.0:
            return (global_score, components) if return_components else global_score
        if music_segments is None or music_segment_mask is None:
            raise ValueError("dense semantic matching requires segments and their mask")
        if music_segments.ndim != 3 or music_segments.shape[0] != music_features.shape[0]:
            raise ValueError("music segments must be [M,T,D] and align with music rows")
        if music_segments.shape[2] != self.embedding_dim:
            raise ValueError("music segment embedding dimension differs from the model")

        query = self.encode_factor("text", "semantic", text_features)
        flat_segments = music_segments.reshape(-1, self.embedding_dim)
        adapted_segments = self.encode_factor(
            "music", "semantic", flat_segments
        ).reshape_as(music_segments)
        local_score, attention = query_conditioned_segment_similarity(
            query,
            adapted_segments,
            music_segment_mask,
            temperature=segment_temperature,
            top_k=segment_top_k,
        )
        score = (1.0 - local_weight) * global_score + local_weight * local_score
        components.update(local=local_score, attention=attention)
        return (score, components) if return_components else score

    def score_structured_query(
        self,
        joint_text: torch.Tensor,
        affect_text: torch.Tensor,
        preference_text: torch.Tensor,
        music_features: torch.Tensor,
        *,
        music_segments: torch.Tensor | None = None,
        music_segment_mask: torch.Tensor | None = None,
        semantic_local_weight: float = 0.0,
        segment_temperature: float = 0.07,
        segment_top_k: int | None = None,
        preference_only: bool = False,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
        """Score structured queries against a music batch.

        All text tensors have shape ``[Q, D]`` and music has shape ``[M, D]``.
        ``joint_text`` is the frozen embedding of the complete query, while the
        other two tensors are embeddings of the separated fields.
        """

        if preference_only:
            score = self.factor_similarity(
                "preference", preference_text, music_features, scaled=False
            )
            components = {"preference": score}
        else:
            components = {
                "semantic": self.semantic_similarity(
                    joint_text,
                    music_features,
                    music_segments=music_segments,
                    music_segment_mask=music_segment_mask,
                    local_weight=semantic_local_weight,
                    segment_temperature=segment_temperature,
                    segment_top_k=segment_top_k,
                ),
                "affect": self.factor_similarity(
                    "affect", affect_text, music_features, scaled=False
                ),
                "preference": self.factor_similarity(
                    "preference", preference_text, music_features, scaled=False
                ),
            }
            weights = self.factor_weights()
            score = sum(weights[index] * components[factor] for index, factor in enumerate(FACTORS))
        if return_components:
            return score, components
        return score
