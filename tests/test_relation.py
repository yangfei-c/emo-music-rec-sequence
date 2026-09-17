from __future__ import annotations

import numpy as np
import torch

from emomm_sequence.relation import relation_log_probabilities
from emomm_sequence.search import direction_from_va, directional_scores


def test_relation_probabilities_are_normalized() -> None:
    delta = torch.tensor([[[-0.4, 0.2], [0.3, -0.1]]])
    log_probabilities = relation_log_probabilities(
        delta,
        torch.tensor([0.10, 0.10]),
        torch.tensor([0.30, 0.30]),
        torch.tensor([0.20, 0.20]),
    )
    probabilities = log_probabilities.exp()
    assert probabilities.shape == (1, 2, 2, 5)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0]))


def test_direction_and_conservative_reduction() -> None:
    direction = direction_from_va(np.array([-0.5, 0.5]), np.array([0.5, 0.0]), 0.10)
    assert direction == (1, -1)
    probabilities = np.zeros((3, 2, 2, 2, 5), dtype=np.float64)
    probabilities[..., 2] = 0.2
    probabilities[..., 3] = 0.3
    probabilities[..., 1] = 0.1
    probabilities[..., 0] = 0.2
    probabilities[..., 4] = 0.2
    scores = directional_scores(probabilities, direction, endpoint=False)
    assert scores.shape == (2, 2)
    assert np.allclose(scores, 0.3)


def test_maintain_uses_no_large_change_probability() -> None:
    p = relation_log_probabilities(torch.zeros(2, 2, 2), torch.tensor([.025, .025]), torch.tensor([.20, .20]), torch.tensor([.10, .10])).exp().numpy()[None]
    scores = directional_scores(p, (0, 0), endpoint=True)
    assert np.all(scores > .50)
