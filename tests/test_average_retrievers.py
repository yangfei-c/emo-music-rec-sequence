from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/training"))

from average_retrievers import average_states


def test_average_states_uniformly_averages_floating_tensors() -> None:
    first = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
    second = {"weight": torch.tensor([3.0, 7.0]), "count": torch.tensor(2)}
    averaged = average_states([first, second])
    torch.testing.assert_close(averaged["weight"], torch.tensor([2.0, 5.0]))
    assert averaged["weight"].dtype == first["weight"].dtype
    assert averaged["count"].item() == 2


def test_average_states_normalizes_weights() -> None:
    first = {"weight": torch.tensor([0.0])}
    second = {"weight": torch.tensor([10.0])}
    averaged = average_states([first, second], [1.0, 3.0])
    torch.testing.assert_close(averaged["weight"], torch.tensor([7.5]))


def test_average_states_rejects_incompatible_non_floating_state() -> None:
    first = {"count": torch.tensor(1)}
    second = {"count": torch.tensor(2)}
    with pytest.raises(ValueError, match="non-floating"):
        average_states([first, second])
