from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from emomm_sequence.training.records import sha256
from emomm_sequence.models.music_va import VAQuantileRegressor
from emomm_sequence.losses.va import pinball_loss
from emomm_sequence.calibration.cqr import (
    VAUncertaintyModel,
    apply_joint_cqr,
    cqr_rank,
    fit_joint_cqr,
    joint_nonconformity,
)


def test_quantile_head_is_ordered_bounded_and_trainable() -> None:
    model = VAQuantileRegressor(embedding_dim=4, hidden_dim=8, dropout=0)
    features = torch.randn(7, 4)
    quantiles = model(features)
    assert quantiles.shape == (7, 2, 3)
    assert torch.all((quantiles >= 0) & (quantiles <= 1))
    assert torch.all(quantiles[..., 0] <= quantiles[..., 1])
    assert torch.all(quantiles[..., 1] <= quantiles[..., 2])
    loss = pinball_loss(quantiles, torch.rand(7, 2))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_joint_cqr_uses_one_song_score_and_finite_sample_rank() -> None:
    assert cqr_rank(81, 0.1) == 74
    assert cqr_rank(35, 0.1) == 33
    with pytest.raises(ValueError, match="unavailable"):
        cqr_rank(4, 0.1)
    quantiles = np.tile(np.array([[[0.2, 0.5, 0.8], [0.2, 0.5, 0.8]]]), (10, 1, 1))
    targets = np.tile(np.array([[0.5, 0.5]]), (10, 1))
    targets[:2] = [[0.1, 0.5], [0.5, 0.9]]
    targets[5] = [0.02, 0.95]
    scores = joint_nonconformity(quantiles, targets)
    assert scores.shape == (10,)
    assert np.allclose(scores[:2], [0.1, 0.1])
    assert np.isclose(scores[5], 0.18)
    calibration = fit_joint_cqr(quantiles, targets, ["deam"] * 5 + ["pmemo"] * 5, alpha=0.2)
    assert calibration["source_corrections"]["deam"]["rank_one_based"] == 5
    assert calibration["source_corrections"]["pmemo"]["rank_one_based"] == 5
    assert np.isclose(calibration["source_corrections"]["deam"]["correction"], 0.1)
    assert np.isclose(calibration["source_corrections"]["pmemo"]["correction"], 0.18)
    median, lower, upper = apply_joint_cqr(quantiles, calibration, ["deam"] * 5 + ["pmemo"] * 5)
    assert median.shape == lower.shape == upper.shape == (10, 2)
    assert np.all(lower <= median)
    assert np.all(median <= upper)
    assert ((targets >= lower) & (targets <= upper)).all()


def test_runtime_loads_matching_artifacts_and_rejects_mismatch(tmp_path) -> None:
    torch.manual_seed(1)
    model = VAQuantileRegressor(embedding_dim=4, hidden_dim=8, dropout=0)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "model_config": {"embedding_dim": 4, "hidden_dim": 8, "dropout": 0.0}}, checkpoint)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "checkpoint_sha256": sha256(checkpoint),
        "source_corrections": {"deam": {"correction": 0.03}},
        "default_correction": 0.03,
    }), encoding="utf-8")
    runtime = VAUncertaintyModel.from_artifacts(checkpoint, calibration)
    median, lower, upper = runtime.predict(np.zeros((3, 4), dtype=np.float32))
    assert median.shape == lower.shape == upper.shape == (3, 2)
    assert np.all(lower <= median) and np.all(median <= upper)
    with pytest.raises(ValueError, match="does not match"):
        calibration.write_text(json.dumps({"checkpoint_sha256": "wrong"}), encoding="utf-8")
        VAUncertaintyModel.from_artifacts(checkpoint, calibration)
