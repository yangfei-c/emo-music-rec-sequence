"""Legacy weight-normalization names must never silently randomize MERT weights."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class TinyMert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.pos_conv_embed = nn.Module()
        self.encoder.pos_conv_embed.conv = nn.utils.parametrizations.weight_norm(nn.Conv1d(4, 4, 3), dim=2)


def loader(monkeypatch):
    spec = importlib.util.spec_from_file_location("mert_extract_test", ROOT / "scripts/features/extract_mert_tokens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "MusicHubertModel", TinyMert)
    monkeypatch.setattr(module.MusicHubertConfig, "from_pretrained", lambda *_: SimpleNamespace(conv_pos_batch_norm=False))
    return module


def test_legacy_mert_keys_load_exactly_and_freeze(tmp_path, monkeypatch):
    module = loader(monkeypatch)
    reference = TinyMert(None)
    legacy = {key.replace("parametrizations.weight.original0", "weight_g").replace("parametrizations.weight.original1", "weight_v"): value
              for key, value in reference.state_dict().items()}
    torch.save(legacy, tmp_path / "pytorch_model.bin")
    loaded = module.load_frozen_mert(tmp_path, torch.device("cpu"))
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key])
    assert not any(parameter.requires_grad for parameter in loaded.parameters())


def test_mert_missing_weights_fail_instead_of_randomizing(tmp_path, monkeypatch):
    module = loader(monkeypatch)
    torch.save({}, tmp_path / "pytorch_model.bin")
    with pytest.raises(RuntimeError, match="Missing key"):
        module.load_frozen_mert(tmp_path, torch.device("cpu"))
