from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class XLMRobertaVAModel(nn.Module):
    """XLM-R `<s>` 表示、256 维文本 embedding 与二维 VA 回归头。"""

    checkpoint_state_kind = "trainable_only"

    def __init__(
        self,
        pretrained_name: str = "xlm-roberta-base",
        hidden_dim: int = 256,
        dropout: float = 0.2,
        max_length: int = 128,
        trainable_encoder_layers: int = 4,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or max_length <= 0 or not 0 <= dropout < 1:
            raise ValueError("hidden_dim/max_length 必须大于 0，dropout 必须位于 [0,1)")
        self.pretrained_name, self.max_length = pretrained_name, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_name, local_files_only=local_files_only
        )
        self.encoder = AutoModel.from_pretrained(
            pretrained_name, local_files_only=local_files_only
        )
        layers = self.encoder.encoder.layer
        if not 0 <= trainable_encoder_layers <= len(layers):
            raise ValueError(f"trainable_encoder_layers 应位于 [0,{len(layers)}]")
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for layer in (
            layers[-trainable_encoder_layers:] if trainable_encoder_layers else ()
        ):
            for parameter in layer.parameters():
                parameter.requires_grad = True

        encoder_dim = int(self.encoder.config.hidden_size)
        self.text_projection = nn.Sequential(
            nn.LayerNorm(encoder_dim),
            nn.Linear(encoder_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.va_head = nn.Linear(hidden_dim, 2)
        self.hidden_dim, self.trainable_encoder_layers = (
            hidden_dim,
            trainable_encoder_layers,
        )

    def tokenize(
        self, texts: list[str], device: torch.device
    ) -> dict[str, torch.Tensor]:
        if not texts or any(not isinstance(text, str) or not text for text in texts):
            raise ValueError("texts 必须是非空字符串列表")
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            key: value.to(device, non_blocking=device.type == "cuda")
            for key, value in encoded.items()
        }

    def forward(
        self, texts: list[str], device: torch.device | None = None
    ) -> dict[str, torch.Tensor]:
        device = device or next(self.parameters()).device
        encoded = self.tokenize(texts, device)
        cls = self.encoder(**encoded).last_hidden_state[:, 0]
        embedding = self.text_projection(cls)
        return {"text_embedding": embedding, "va_predictions": self.va_head(embedding)}

    def save_tokenizer(self, directory: str | Path) -> None:
        self.tokenizer.save_pretrained(directory)

    def checkpoint_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        return {
            name: value
            for name, value in self.state_dict().items()
            if name in trainable
        }

    def load_checkpoint_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        trainable = {
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if set(state) != trainable:
            missing, unexpected = sorted(trainable - set(state)), sorted(
                set(state) - trainable
            )
            raise ValueError(
                f"文本 checkpoint 参数不一致；缺少={missing[:5]}，多余={unexpected[:5]}"
            )
        self.load_state_dict(state, strict=False)
