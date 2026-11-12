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
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        self.text_model = AutoModel.from_pretrained(text_model_name, local_files_only=local_files_only)
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
    def tokenizer(text_model_name: str = "FacebookAI/xlm-roberta-base", local_files_only=True):
        return AutoTokenizer.from_pretrained(text_model_name, local_files_only=local_files_only)

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




from tqdm import tqdm


@torch.inference_mode()
def encode_audio(tokens, model, device):
    """Project newly computed MERT tokens, with CLaMP3's zero boundary tokens."""
    tokens = tokens[:model.max_audio_length - 2]
    tokens = torch.cat((torch.zeros(1, 768), tokens, torch.zeros(1, 768)))
    values = tokens.unsqueeze(0).to(device)
    mask = torch.ones(values.shape[:2], dtype=torch.bool, device=device)
    return F.normalize(model.encode_audio(values, mask), dim=-1)[0].float().cpu().numpy()

def encode_texts(
    texts: list[str],
    model: Clamp3TextAudioEncoder,
    tokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    chunks, owners, weights = [], [], []
    for owner, text in enumerate(texts):
        ids = tokenizer(str(text), add_special_tokens=True, truncation=False)["input_ids"]
        if not ids:
            ids = [tokenizer.cls_token_id, tokenizer.sep_token_id]
        sample_chunks = [ids[start : start + max_length] for start in range(0, len(ids), max_length)]
        sample_chunks[-1] = ids[-max_length:]
        full, remainder = divmod(len(ids), max_length)
        sample_weights = [max_length] * full + ([remainder] if remainder else [])
        if not sample_weights:
            sample_weights = [len(ids)]
        if len(sample_weights) != len(sample_chunks):
            raise RuntimeError("CLaMP3 text chunk accounting mismatch")
        for chunk, weight in zip(sample_chunks, sample_weights):
            chunks.append(chunk)
            owners.append(owner)
            weights.append(weight)
    totals = torch.zeros((len(texts), 768), dtype=torch.float32)
    denominators = torch.zeros((len(texts), 1), dtype=torch.float32)
    for start in tqdm(range(0, len(chunks), batch_size), desc="CLaMP3 text", unit="batch"):
        batch_chunks = chunks[start : start + batch_size]
        width = max(len(item) for item in batch_chunks)
        input_ids = torch.full(
            (len(batch_chunks), width), tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros((len(batch_chunks), width), dtype=torch.bool)
        for index, item in enumerate(batch_chunks):
            input_ids[index, : len(item)] = torch.tensor(item, dtype=torch.long)
            mask[index, : len(item)] = True
        with torch.inference_mode():
            encoded = model.encode_text(input_ids.to(device), mask.to(device)).float().cpu()
        batch_owners = torch.tensor(owners[start : start + len(batch_chunks)], dtype=torch.long)
        batch_weights = torch.tensor(weights[start : start + len(batch_chunks)], dtype=torch.float32).unsqueeze(1)
        totals.index_add_(0, batch_owners, encoded * batch_weights)
        denominators.index_add_(0, batch_owners, batch_weights)
    return F.normalize(totals / denominators, dim=-1).numpy().astype(np.float32)


