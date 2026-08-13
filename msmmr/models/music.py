from collections.abc import Sequence

import torch
from torch import nn


def _check_pool_inputs(x: torch.Tensor, mask: torch.Tensor) -> None:
    if x.ndim != 3 or mask.shape != x.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(
            f"x/mask 形状或类型错误：{tuple(x.shape)}/{tuple(mask.shape)}/{mask.dtype}"
        )
    if (mask.sum(1) == 0).any().item():
        raise ValueError("存在没有有效 segment 的歌曲")


class MaskedMeanPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _check_pool_inputs(x, mask)
        weight = mask.unsqueeze(-1).to(x.dtype)
        return (x * weight).sum(1) / weight.sum(1)


class MaskedMeanStdPooling(nn.Module):
    """对有效分段计算均值和总体标准差。"""

    def __init__(self, eps: float = 1.0e-5) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps 必须大于 0，实际为 {eps}")
        self.eps = eps

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _check_pool_inputs(x, mask)
        weight = mask.unsqueeze(-1).to(x.dtype)
        count = weight.sum(1)
        mean = (x * weight).sum(1) / count
        variance = ((x - mean.unsqueeze(1)).square() * weight).sum(1) / count
        return torch.cat((mean, (variance + self.eps).sqrt()), dim=-1)


class MusicMultiTaskModel(nn.Module):
    """冻结 MERT 特征上的轻量歌曲编码器与四个任务头。"""

    def __init__(
        self,
        layer_indices: Sequence[int] = (5, 6),
        hidden_dim: int = 256,
        dropout: float = 0.2,
        pooling: str = "mean_std",
        pooling_eps: float = 1.0e-5,
    ) -> None:
        super().__init__()
        indices, pooling = tuple(map(int, layer_indices)), pooling.lower()
        if (
            not indices
            or len(indices) != len(set(indices))
            or any(i not in range(12) for i in indices)
        ):
            raise ValueError(
                f"layer_indices 必须是不重复的 0-based [0,11] 索引：{indices}"
            )
        if hidden_dim <= 0 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim 必须大于 0，dropout 必须位于 [0,1)")
        if pooling not in {"mean", "mean_std"}:
            raise ValueError("pooling 只支持 mean 或 mean_std")

        self.hidden_dim, self.pooling_name = hidden_dim, pooling
        self.register_buffer(
            "layer_indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        input_dim = len(indices) * 768
        self.segment_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if pooling == "mean":
            self.song_pooling, pooled_dim = MaskedMeanPooling(), hidden_dim
        else:
            self.song_pooling, pooled_dim = (
                MaskedMeanStdPooling(pooling_eps),
                hidden_dim * 2,
            )
        self.song_fusion = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.genre_head = nn.Linear(hidden_dim, 87)
        self.instrument_head = nn.Linear(hidden_dim, 40)
        self.mood_head = nn.Linear(hidden_dim, 56)
        self.va_head = nn.Linear(hidden_dim, 2)

    @staticmethod
    def check_inputs(features: torch.Tensor, mask: torch.Tensor) -> None:
        if features.ndim != 4 or tuple(features.shape[2:]) != (12, 768):
            raise ValueError(
                f"features 应为 [B,S,12,768]，实际为 {tuple(features.shape)}"
            )
        if not features.is_floating_point():
            raise TypeError(f"features 应为浮点类型，实际为 {features.dtype}")
        if mask.shape != features.shape[:2] or mask.dtype != torch.bool:
            raise ValueError(
                f"mask 应为 bool {tuple(features.shape[:2])}，实际为 "
                f"{tuple(mask.shape)}/{mask.dtype}"
            )
        if (mask.sum(1) == 0).any().item():
            raise ValueError("存在没有有效 segment 的歌曲")

    def encode(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.check_inputs(features, mask)
        segments = features.index_select(2, self.layer_indices).flatten(2)
        return self.song_fusion(
            self.song_pooling(self.segment_projection(segments), mask)
        )

    def forward(
        self, features: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        embedding = self.encode(features, mask)
        genre, instrument, mood = (
            self.genre_head(embedding),
            self.instrument_head(embedding),
            self.mood_head(embedding),
        )
        return {
            "song_embedding": embedding,
            "genre_logits": genre,
            "instrument_logits": instrument,
            "mood_logits": mood,
            "tag_logits": torch.cat((genre, instrument, mood), dim=-1),
            "va_predictions": self.va_head(embedding),
        }
