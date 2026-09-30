import json
import numpy as np
import pytest
import torch

from emomm_sequence.losses.va import VALoss
from emomm_sequence.models.retriever import RetrieverModel, FACTORS
from emomm_sequence.losses.retrieval import symmetric_multi_positive_info_nce
from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.training.records import check_identity_splits, state_sha256
from emomm_sequence.inference.query import NaturalLanguageQueryParser, QueryParsingError


def test_training_changes_every_adapter_and_roundtrips(tmp_path):
    torch.manual_seed(42)
    model = RetrieverModel(embedding_dim=16, bottleneck_dim=4, dropout=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    x, y = torch.randn(8, 16), torch.randn(8, 16)
    initial = state_sha256(model)
    for _ in range(3):
        optimizer.zero_grad()
        loss = sum(symmetric_multi_positive_info_nce(model.text_adapters[f](x), model.music_adapters[f](y), torch.eye(8, dtype=torch.bool), logit_scale=torch.tensor(10.)) for f in FACTORS)
        loss.backward()
        optimizer.step()
    assert state_sha256(model) != initial
    for modality in (model.text_adapters, model.music_adapters):
        for factor in FACTORS:
            assert modality[factor].up.weight.abs().sum() > 0
    path = tmp_path / "new.pt"
    torch.save({"model": model.state_dict()}, path)
    loaded = FrozenRetriever(path, embedding_dim=16, bottleneck_dim=4, dropout=0)
    assert not any(p.requires_grad for p in loaded.parameters())
    for factor in FACTORS:
        torch.testing.assert_close(model.eval().text_adapters[factor](x), loaded.encode_text(factor, x))


def test_identity_overlap_rejected():
    with pytest.raises(ValueError, match="crosses"):
        check_identity_splits(np.asarray(["song", "song"]), np.asarray(["train", "test"]), "unit")


def test_text_va_training_roundtrip_and_retriever_binding(tmp_path):
    from emomm_sequence.models.text_va import TextVAHead, load_text_va
    model = TextVAHead(16)
    x, y = torch.randn(12, 16), torch.rand(12, 2) * 2 - 1
    before = state_sha256(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003)
    loss = VALoss(.1, .1)(model(x), y)
    loss.backward()
    optimizer.step()
    assert state_sha256(model) != before
    path = tmp_path / "text.pt"
    torch.save({"model": model.state_dict(), "model_config": {"embedding_dim": 16},
                "retriever_checkpoint_sha256": "expected"}, path)
    loaded = load_text_va(path, "expected")
    torch.testing.assert_close(model(x), loaded(x))
    assert not any(p.requires_grad for p in loaded.parameters())
    with pytest.raises(ValueError, match="different affect Adapter"):
        load_text_va(path, "obsolete")


def test_short_request_does_not_invent_preference(monkeypatch):
    parser = NaturalLanguageQueryParser()
    monkeypatch.setattr(parser, "model_digest", lambda: "test")
    values = {"current_affect": "annoyed", "target_affect": "", "music_preference": "", "current_affect_evidence": "很烦", "target_affect_evidence": "", "music_preference_evidence": "", "missing_fields": ["target_affect", "music_preference"]}
    monkeypatch.setattr(parser, "_request", lambda *args: {"message": {"content": json.dumps(values)}})
    parsed = parser.parse("我现在很烦")
    assert parsed.target_affect == "calm" and parsed.music_preference == ""
    assert parsed.defaulted_fields == ("target_affect",)


def test_parser_rejects_fabricated_evidence(monkeypatch):
    parser = NaturalLanguageQueryParser()
    monkeypatch.setattr(parser, "model_digest", lambda: "test")
    monkeypatch.setattr(parser, "_request", lambda *args: {"message": {"content": json.dumps({"current_affect": "annoyed", "current_affect_evidence": "invented text"})}})
    with pytest.raises(QueryParsingError, match="依据"):
        parser.parse("我现在很烦")
