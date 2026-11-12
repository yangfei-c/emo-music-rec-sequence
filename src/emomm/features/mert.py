"""Fresh MERT tokens from raw audio; intermediate tokens remain in memory."""
from pathlib import Path
import hashlib
import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, Wav2Vec2FeatureExtractor


class MERTEncoder:
    def __init__(self, config, device, local_files_only=True):
        self.config, self.device = config, device
        root = Path(snapshot_download(repo_id=config["model"], revision=config["revision"],
                    local_files_only=local_files_only,
                    allow_patterns=["*.json", "*.py", "pytorch_model.bin"]))
        cfg = AutoConfig.from_pretrained(root, trust_remote_code=True, local_files_only=local_files_only)
        if not hasattr(cfg, "conv_pos_batch_norm"):
            cfg.conv_pos_batch_norm = False
        model = AutoModel.from_config(cfg, trust_remote_code=True)
        state = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
        expected = model.state_dict()
        # Match PyTorch's current weight-normalization parameter names strictly.
        for old, new in (("weight_g", "parametrizations.weight.original0"),
                         ("weight_v", "parametrizations.weight.original1")):
            before, after = "encoder.pos_conv_embed.conv." + old, "encoder.pos_conv_embed.conv." + new
            if before in state and after in expected:
                state[after] = state.pop(before)
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False).to(device)
        self.processor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=config["sample_rate"],
            padding_value=0, return_attention_mask=True, do_normalize=True)

    @torch.inference_mode()
    def encode(self, path):
        cfg = self.config
        waveform, _ = librosa.load(path, sr=cfg["sample_rate"], mono=True)
        waveform = np.asarray(waveform, dtype=np.float32)
        if not np.isfinite(waveform).all() or len(waveform) < cfg["sample_rate"]:
            raise ValueError(f"音频不足一秒或含无效值: {path}")
        digest = hashlib.sha256(waveform.tobytes()).hexdigest()
        width = int(cfg["sample_rate"] * cfg["window_seconds"])
        if cfg["window_policy"] == "front_middle_end" and len(waveform) > width:
            starts = dict.fromkeys((0, (len(waveform) - width) // 2, len(waveform) - width))
            waveform = np.concatenate([waveform[s:s + width] for s in starts])
        elif cfg["window_policy"] not in {"front_middle_end", "full"}:
            raise ValueError("未知音频截取方式")
        width = int(cfg["sample_rate"] * cfg["chunk_seconds"])
        chunks = [waveform[i:i + width] for i in range(0, len(waveform), width)
                  if len(waveform[i:i + width]) >= cfg["sample_rate"]]
        parts = []
        # Encode equal-length batches so padding cannot change the temporal mean.
        for start in range(0, len(chunks), cfg["chunk_batch_size"]):
            batch = chunks[start:start + cfg["chunk_batch_size"]]
            groups = [batch] if len({len(x) for x in batch}) == 1 else [batch[:-1], batch[-1:]]
            for group in groups:
                if not group:
                    continue
                values = self.processor(group, sampling_rate=cfg["sample_rate"], return_tensors="pt", padding=True)
                hidden = self.model(values.input_values.to(self.device), output_hidden_states=True).hidden_states
                parts.append(torch.stack([h.mean(1) for h in hidden]).mean(0).float().cpu())
        tokens = torch.cat(parts)
        if tokens.ndim != 2 or tokens.shape[1] != 768 or not torch.isfinite(tokens).all():
            raise ValueError(f"MERT 特征无效: {path}")
        return tokens, digest
