from itertools import permutations
import numpy as np
import pytest
from emomm_sequence.evaluation.sequence_label_oracle import positional_va_oracles


def test_label_assignment_matches_exhaustive_unique_paths():
    rng = np.random.default_rng(71)
    va = rng.random((6, 2))
    targets = np.linspace([.25, .75], [.75, .25], 4)
    reference = positional_va_oracles(va, targets, 0, 5)
    cost = lambda path: float(np.abs(va[list(path)] - targets).mean())
    assert reference["free_endpoint_min_va_mae"] == pytest.approx(min(cost(p) for p in permutations(range(6), 4)))
    assert reference["fixed_endpoint_min_va_mae"] == pytest.approx(min(cost((0, *p, 5)) for p in permutations(range(1, 5), 2)))
    assert len(set(reference["free_endpoint_path_indices"])) == 4


def test_two_song_path_and_invalid_endpoints():
    va = np.array([[0, 0], [1, 1], [.5, .5]])
    targets = np.array([[1, 1], [0, 0]])
    result = positional_va_oracles(va, targets, 0, 1)
    assert result["free_endpoint_min_va_mae"] == 0
    assert result["fixed_endpoint_va_penalty"] == 1
    with pytest.raises(ValueError):
        positional_va_oracles(va, targets, 0, 0)
