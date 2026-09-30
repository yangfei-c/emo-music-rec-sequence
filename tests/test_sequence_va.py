import itertools
import numpy as np
import pytest
from emomm_sequence.inference.sequence_va import VAProblem, va_beam_search


def fixture():
    median = np.random.default_rng(44).uniform(.2, .8, (6, 2))
    return VAProblem(np.arange(6).astype(str), median, median - .1, median + .1,
                     np.linspace([.2, .3], [.8, .6], 4), np.linspace(0, 1, 6), 0, 5)


def test_shared_interval_adversary_matches_corner_enumeration():
    p = fixture()
    path = [0, 1, 2, 5]
    corners = []
    for mask in itertools.product([0, 1], repeat=8):
        y = np.where(np.array(mask).reshape(4, 2), p.upper[path], p.lower[path])
        value = .5 * np.abs(y - p.target_path).mean() + .25 * p.preference[path].mean() + .25 * np.abs(np.diff(y, axis=0) - p.step).mean() / 2
        corners.append(value)
    assert p.cost(path) == pytest.approx(max(corners), abs=1e-10)


def test_wide_va_beam_matches_enumerated_unique_paths():
    p = fixture()
    paths = [(0, *x, 5) for x in itertools.permutations(range(1, 5), 2)]
    best = min(p.cost(path) for path in paths)
    result = va_beam_search(p, range(6), width=100)
    assert result.objective == pytest.approx(best, abs=1e-10)
    assert result.status == "ok"


def test_zero_width_intervals_reduce_to_point_objective():
    p = fixture()
    p.lower = p.median.copy()
    p.upper = p.median.copy()
    assert p.cost([0, 1, 2, 5]) == pytest.approx(p.point_cost([0, 1, 2, 5]), abs=1e-10)
