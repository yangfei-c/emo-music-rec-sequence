from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from emomm_sequence.models.retriever import ClipStyleAdapter, RetrieverModel
from emomm_sequence.training.selection import (
    retrieval_selection_score,
    tag_balanced_selection_score,
)
from scripts.training.train_retriever import capture_factor_state


@pytest.mark.parametrize("modality", ["text", "music"])
def test_va_gradient_updates_only_own_affect_adapter_and_va_head(modality):
    torch.manual_seed(42)
    model = RetrieverModel(embedding_dim=8, bottleneck_dim=3, dropout=0.0)
    frozen_features = torch.randn(6, 8)  # cached backbone output has no autograd graph
    target = torch.rand(6, 2) * 2 - 1
    F.smooth_l1_loss(model.predict_va(modality, frozen_features), target).backward()
    gradients = {name: p.grad for name, p in model.named_parameters()}
    assert gradients[f"{modality}_adapters.affect.up.weight"].abs().sum() > 0
    assert gradients[f"{modality}_va_head.weight"].abs().sum() > 0
    other = "music" if modality == "text" else "text"
    for name, grad in gradients.items():
        if name.startswith(other) or ".semantic." in name or ".preference." in name:
            assert grad is None or torch.count_nonzero(grad) == 0
    assert frozen_features.grad is None


def test_zero_auxiliary_weight_does_not_create_va_gradient():
    model = RetrieverModel(embedding_dim=8, bottleneck_dim=3, dropout=0.0)
    (0.0 * model.predict_va("text", torch.randn(4, 8)).square().mean()).backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())


def test_clip_style_matches_published_feature_blending_formula():
    adapter = ClipStyleAdapter(8, 2, blend=0.2)
    x = torch.randn(3, 8)
    raw = F.relu(F.linear(F.relu(F.linear(x, adapter.fc[0].weight)), adapter.fc[2].weight))
    assert torch.allclose(adapter(x), F.normalize(0.8 * x + 0.2 * raw, dim=-1))
    assert sum(p.numel() for p in adapter.parameters()) == 32


def test_retrieval_selection_uses_three_equal_tasks_only():
    assert retrieval_selection_score(0.1, 0.2, 0.6) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        retrieval_selection_score(float("nan"), 0.2, 0.3)


def test_balanced_tag_selection_is_geometric_mean() -> None:
    assert tag_balanced_selection_score(0.25, 0.04) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        tag_balanced_selection_score(-0.1, 0.2)


def test_capacity_matched_shared_control_and_six_heads():
    ours = RetrieverModel()
    shared = RetrieverModel(bottleneck_dim=192, adapter_sharing="across_factors")
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert abs(count(shared) / count(ours) - 1) < 0.02
    assert shared.text_adapters["affect"] is shared.text_adapters["semantic"]
    assert ours.text_adapters["affect"] is not ours.text_adapters["semantic"]


def test_affect_checkpoint_keeps_epoch_matched_va_heads():
    model = RetrieverModel(embedding_dim=8, bottleneck_dim=3, dropout=0.0)
    affect = capture_factor_state(model, "affect")
    semantic = capture_factor_state(model, "semantic")
    assert "text_va_head.weight" in affect
    assert "music_va_head.bias" in affect
    assert "text_va_head.weight" not in semantic
