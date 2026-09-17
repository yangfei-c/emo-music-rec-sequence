"""Memory-conscious loader for the official CLaMP3 text and audio branches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel


CHECKPOINT_PREFIXES = (
    "text_model.",
    "text_proj.",
    "audio_model.",
    "audio_proj.",
)


def _prefixed_state(state: dict[str, torch.Tensor], prefix: str) -> dict:
    selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
    if not selected:
        raise KeyError(f"CLaMP3 checkpoint lacks prefix {prefix!r}")
    return selected


def _masked_mean(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError(f"invalid features/mask: {tuple(features.shape)}/{tuple(mask.shape)}")
    weights = mask.unsqueeze(-1).to(features.dtype)
    counts = weights.sum(1)
    if (counts == 0).any().item():
        raise ValueError("a CLaMP3 sample has no valid token")
    return (features * weights).sum(1) / counts


class Clamp3TextAudioEncoder(nn.Module):
    """Official SAAS text/audio parameters without the unused symbolic branch."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        text_model_name: str = "FacebookAI/xlm-roberta-base",
        hidden_size: int = 768,
        audio_layers: int = 12,
        max_audio_length: int = 128,
        normalize: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, hidden_size)
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=audio_layers,
            num_attention_heads=hidden_size // 64,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_audio_length,
        )
        self.audio_model = BertModel(audio_config)
        self.audio_proj = nn.Linear(hidden_size, hidden_size)
        self.normalize = bool(normalize)
        self.max_audio_length = int(max_audio_length)

        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True, mmap=True
        )
        state = checkpoint.get("model")
        if not isinstance(state, dict):
            raise ValueError("invalid CLaMP3 checkpoint: missing model state")
        self.text_model.load_state_dict(_prefixed_state(state, "text_model."), strict=True)
        self.text_proj.load_state_dict(_prefixed_state(state, "text_proj."), strict=True)
        self.audio_model.load_state_dict(_prefixed_state(state, "audio_model."), strict=True)
        self.audio_proj.load_state_dict(_prefixed_state(state, "audio_proj."), strict=True)
        self.checkpoint_epoch = int(checkpoint.get("epoch", -1))
        self.checkpoint_loss = float(checkpoint.get("min_eval_loss", float("nan")))
        del checkpoint, state

        if freeze:
            self.requires_grad_(False)
            self.eval()

    @staticmethod
    def tokenizer(text_model_name: str = "FacebookAI/xlm-roberta-base"):
        return AutoTokenizer.from_pretrained(text_model_name)

    def _finish(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features, dim=-1) if self.normalize else features

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return self._finish(self.text_proj(_masked_mean(hidden, attention_mask.bool())))

    def encode_audio(
        self, audio_tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        hidden, mask = self._audio_hidden(audio_tokens, attention_mask)
        return self._finish(self.audio_proj(_masked_mean(hidden, mask)))

    def _audio_hidden(
        self, audio_tokens: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if audio_tokens.ndim != 3 or audio_tokens.shape[-1] != 768:
            raise ValueError(f"audio_tokens must be [B,T,768], got {tuple(audio_tokens.shape)}")
        if attention_mask.shape != audio_tokens.shape[:2]:
            raise ValueError(
                "audio attention mask must match the first two token dimensions"
            )
        if audio_tokens.shape[1] > self.max_audio_length:
            raise ValueError("audio token sequence exceeds CLaMP3 maximum")
        mask = attention_mask.bool()
        if (~mask.any(dim=1)).any().item():
            raise ValueError("a CLaMP3 audio sample has no valid token")
        hidden = self.audio_model(
            inputs_embeds=audio_tokens, attention_mask=mask
        ).last_hidden_state
        return hidden, mask

    def encode_audio_segments(
        self,
        audio_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        exclude_boundary_tokens: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return contextual CLaMP3 embeddings for the 30-second audio windows.

        ``load_audio_tokens`` follows the official extractor and places one zero
        boundary token at each side of the real MERT windows.  Global CLaMP3
        pooling keeps those tokens for exact compatibility with the released
        checkpoint; dense query-to-window matching excludes them by default.
        """

        hidden, mask = self._audio_hidden(audio_tokens, attention_mask)
        segment_mask = mask.clone()
        if exclude_boundary_tokens:
            for row in range(segment_mask.shape[0]):
                positions = torch.nonzero(segment_mask[row], as_tuple=False).flatten()
                if positions.numel() < 3:
                    raise ValueError(
                        "boundary exclusion requires at least one real audio segment"
                    )
                segment_mask[row, positions[0]] = False
                segment_mask[row, positions[-1]] = False
        projected = self.audio_proj(hidden)
        return self._finish(projected), segment_mask


def _segment_number(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"invalid segment filename: {path}") from error


def load_audio_tokens(path: str | Path, max_length: int = 128) -> torch.Tensor:
    """Load official MERT tokens or adapt existing [segment, layer, 768] features.

    CLaMP3's public extractor averages MERT hidden layers and adds zero boundary
    tokens. Existing EmoMM-Sequence features store one [12,768] tensor per 30-second
    window, so the same layer averaging is applied before the boundary tokens.
    """

    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("segment_*.npy"), key=_segment_number)
        if not files:
            raise FileNotFoundError(f"no MERT segments in {path}")
        tokens = []
        for file in files:
            value = np.load(file, allow_pickle=False)
            if value.shape == (1, 12, 768):
                value = value[0]
            if value.shape != (12, 768):
                raise ValueError(f"unexpected MERT shape {value.shape}: {file}")
            tokens.append(value.mean(axis=0, dtype=np.float32))
        array = np.stack(tokens)
    elif path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
        if value.shape[-1] != 768:
            raise ValueError(f"unexpected MERT last dimension {value.shape}: {path}")
        array = value.reshape(-1, 768).astype(np.float32, copy=False)
    else:
        raise ValueError(f"unsupported CLaMP3 audio feature path: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"MERT tokens contain NaN/Inf: {path}")
    array = array[: max(0, max_length - 2)]
    boundary = np.zeros((1, 768), dtype=np.float32)
    array = np.concatenate((boundary, array, boundary), axis=0)
    return torch.from_numpy(array)


def collate_audio_tokens(samples: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not samples:
        raise ValueError("empty audio batch")
    length = max(item.shape[0] for item in samples)
    batch = torch.zeros((len(samples), length, 768), dtype=torch.float32)
    mask = torch.zeros((len(samples), length), dtype=torch.bool)
    for index, item in enumerate(samples):
        batch[index, : item.shape[0]] = item
        mask[index, : item.shape[0]] = True
    return batch, mask
