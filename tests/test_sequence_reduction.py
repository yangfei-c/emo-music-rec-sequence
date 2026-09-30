from copy import deepcopy
import json

import pytest
import torch
from emomm_sequence.models.sequence_reduction import DynamicReductionSelector, build_dynamic_selector
from emomm_sequence.losses.sequence_policy import policy_loss
from emomm_sequence.training.records import sha256
from emomm_sequence.training.sequence_selector import load_model


def test_dynamic_reduction_uses_small_local_attention_and_records_all_proposals():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    model = DynamicReductionSelector(8, 16, 1, 2)
    sizes = []
    handle = model.local_encoder.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    x, text = torch.randn(20, 8), torch.randn(3, 8)
    out = model.select(x, text, 5, 8, 0, 19, dynamic_k=4, sample=True)
    handle.remove()
    assert sizes == [5, 5, 5]  # four local candidates plus one condition token
    assert len(set(out.indices)) == 5 and out.indices[-1] == 19
    assert set(out.indices).issubset(set(out.proposed_indices))
    assert out.local_scored_occurrences == 12
    assert out.dynamic_k == 4
    policy_loss(out, .1, .3).backward()
    assert model.reduce_query.weight.grad.abs().sum() > 0
    assert next(model.local_encoder.parameters()).grad.abs().sum() > 0


def small_config(k=None):
    route = {"hidden_dim": 16, "attention_layers": 1, "attention_heads": 2}
    if k is not None:
        route["dynamic_k"] = k
    return {"assets": {"embedding_dim": 8}, "route_b": route}


def test_configured_k8_controls_local_attention_without_call_override():
    torch.manual_seed(9)
    torch.set_num_threads(2)
    model = build_dynamic_selector(small_config(8))
    sizes = []
    handle = model.local_encoder.register_forward_pre_hook(lambda module, args: sizes.append(args[0].shape[1]))
    out = model.select(torch.randn(20, 8), torch.randn(3, 8), 5, 16, 0, 19, sample=True)
    handle.remove()
    assert sizes == [9, 9, 9]
    assert out.dynamic_k == 8 and out.local_scored_occurrences == 24
    assert len(set(out.indices)) == 5 and out.indices[0] == 0 and out.indices[-1] == 19
    assert set(out.indices) <= set(out.proposed_indices)
    policy_loss(out, .1, .3).backward()
    assert model.reduce_query.weight.grad.abs().sum() > 0
    assert next(model.local_encoder.parameters()).grad.abs().sum() > 0


@pytest.mark.parametrize("k", [0, -1, 1.5, True])
def test_invalid_configured_width_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        build_dynamic_selector(small_config(k))


def test_invalid_call_width_is_rejected_before_selection():
    model = build_dynamic_selector(small_config(8))
    with pytest.raises(ValueError, match="positive integer"):
        model.select(torch.randn(20, 8), torch.randn(3, 8), 5, 16, 0, 19, dynamic_k=0)


@pytest.mark.parametrize("k", [None, 8])
def test_checkpoint_width_is_preserved_and_mismatched_requests_are_rejected(tmp_path, k):
    config = small_config(k)
    model = build_dynamic_selector(config)
    directory = tmp_path / "runs/pilot/dynamic_reduce/seed_42/width_check"
    directory.mkdir(parents=True)
    checkpoint = directory / "best.pt"
    torch.save({"manifest": {"config": config, "objective": "frozen"}, "model": model.state_dict()}, checkpoint)
    (directory / "selection.json").write_text(json.dumps({"checkpoint_sha256": sha256(checkpoint)}), encoding="utf-8")
    restored, _ = load_model(config, tmp_path, "pilot", "dynamic_reduce", 42, torch.device("cpu"), "width_check")
    assert restored.dynamic_k == (16 if k is None else k)
    mismatched = deepcopy(config)
    mismatched["route_b"]["dynamic_k"] = 8 if k is None else 16
    with pytest.raises(ValueError, match="differs from the selected checkpoint"):
        load_model(mismatched, tmp_path, "pilot", "dynamic_reduce", 42, torch.device("cpu"), "width_check")
