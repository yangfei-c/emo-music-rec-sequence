from __future__ import annotations

import torch

from emomm_sequence.retrieval.factorization import (
    counterfactual_masks,
    mean_positive_label_embedding,
)


def test_mean_positive_label_embedding_handles_multilabel_and_empty_rows() -> None:
    targets = torch.tensor([[1, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=torch.bool)
    labels = torch.eye(3)
    pooled, valid = mean_positive_label_embedding(targets, labels)
    assert valid.tolist() == [True, True, False]
    assert torch.allclose(pooled[0], torch.tensor([2.0**-0.5, 2.0**-0.5, 0.0]))
    assert torch.allclose(pooled[1], torch.tensor([0.0, 0.0, 1.0]))


def test_counterfactual_masks_control_one_factor_at_a_time() -> None:
    preference = torch.tensor([[1, 0], [1, 0], [0, 1]], dtype=torch.bool)
    affect = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.bool)
    affect_negative, preference_negative = counterfactual_masks(preference, affect)
    assert affect_negative[0, 1]
    assert preference_negative[0, 2]
    assert not affect_negative.diagonal().any()
    assert not preference_negative.diagonal().any()
