from __future__ import annotations

import importlib.util
from itertools import product, permutations
from pathlib import Path
import sys

import numpy as np
import pytest

# The package initializer imports the production torch stack. This numerical
# unit test loads the standalone search module directly so it runs in a small
# NumPy-only environment as well.
_module_path = Path(__file__).resolve().parents[1] / "src" / "emomm_sequence" / "inference" / "search.py"
_spec = importlib.util.spec_from_file_location("robust_search_under_test", _module_path)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
point_path_cost = _module.point_path_cost
rank_relevance = _module.rank_relevance
robust_path_cost = _module.robust_path_cost
select_robust_sequence = _module.select_robust_sequence


def _brute_interval_cost(lower, upper, step, active):
    best = 0.0
    for vertex in product((0, 1), repeat=lower.size):
        selected = np.where(np.asarray(vertex).reshape(lower.shape), upper, lower)
        best = max(best, point_path_cost(selected, step, active))
    return best


@pytest.mark.parametrize("length", [2, 3, 5])
def test_full_path_dp_matches_interval_vertex_exhaustion(length):
    rng = np.random.default_rng(20260926 + length)
    center = rng.uniform(0.2, 0.8, size=(length, 2))
    radius = rng.uniform(0.0, 0.15, size=(length, 2))
    lower, upper = center - radius, center + radius
    step = np.array([0.075, -0.05])
    for active in ((True, True), (True, False), (False, True)):
        assert robust_path_cost(lower, upper, step, active) == pytest.approx(
            _brute_interval_cost(lower, upper, step, active), abs=1e-12
        )


def test_interval_collapse_widening_and_consistent_middle_song():
    median = np.array([[0.3, 0.4], [0.5, 0.4], [0.7, 0.4]])
    step = np.array([0.2, 0.0])
    point = point_path_cost(median, step, (True, False))
    assert robust_path_cost(median, median, step, (True, False)) == pytest.approx(point)

    tight_lower = median - np.array([[0.02, 0], [0.04, 0], [0.02, 0]])
    tight_upper = median + np.array([[0.02, 0], [0.04, 0], [0.02, 0]])
    wide_lower = tight_lower - np.array([[0.01, 0], [0.10, 0], [0.01, 0]])
    wide_upper = tight_upper + np.array([[0.01, 0], [0.10, 0], [0.01, 0]])
    tight = robust_path_cost(tight_lower, tight_upper, step, (True, False))
    wide = robust_path_cost(wide_lower, wide_upper, step, (True, False))
    assert wide >= tight >= point

    # Maximizing each edge independently may give the middle song two
    # incompatible VA values; the full-path DP cannot exceed that bound.
    independent_edges = sum(
        robust_path_cost(tight_lower[k : k + 2], tight_upper[k : k + 2], step, (True, False))
        for k in range(2)
    )
    assert tight <= independent_edges + 1e-12


def test_robust_and_point_planners_disagree_on_uncertain_middle_song():
    median = np.array([[0.30, 0.5], [0.50, 0.5], [0.48, 0.5], [0.70, 0.5]])
    lower = np.array([[0.28, 0.5], [0.35, 0.5], [0.46, 0.5], [0.68, 0.5]])
    upper = np.array([[0.32, 0.5], [0.65, 0.5], [0.50, 0.5], [0.72, 0.5]])
    relevance = np.ones(4)
    common = dict(
        median=median, lower=lower, upper=upper,
        start_relevance=relevance, target_relevance=relevance,
        num_tracks=3, start_index=0, target_index=3,
        direction=(1, None), lambda_retrieval=0.0, beam_width=4,
    )
    point = select_robust_sequence(**common, mode="point")
    robust = select_robust_sequence(**common, mode="robust")
    assert point.indices == (0, 1, 3)
    assert robust.indices == (0, 2, 3)
    assert point.point_path_cost == pytest.approx(0)
    assert robust.robust_path_cost < point.robust_path_cost
    assert robust.reference_step == pytest.approx((0.2, 0.0))


@pytest.mark.parametrize("mode", ["robust", "point"])
def test_wide_beam_equals_exhaustive_small_candidate_search(mode):
    median = np.array([
        [0.20, 0.85], [0.30, 0.70], [0.47, 0.55],
        [0.58, 0.48], [0.72, 0.30], [0.80, 0.20],
    ])
    radii = np.array([
        [0.02, 0.02], [0.04, 0.06], [0.09, 0.03],
        [0.02, 0.07], [0.05, 0.02], [0.02, 0.02],
    ])
    lower, upper = median - radii, median + radii
    start = np.array([1.0, 0.9, 0.5, 0.6, 0.1, 0.0])
    target = np.array([0.0, 0.1, 0.5, 0.4, 0.9, 1.0])
    n, lam = 4, 0.7
    step = (median[5] - median[0]) / (n - 1)
    scored = []
    for middle in permutations(range(1, 5), n - 2):
        path = (0, *middle, 5)
        robust = robust_path_cost(lower[list(path)], upper[list(path)], step)
        point = point_path_cost(median[list(path)], step)
        text = np.mean([
            1.0 - ((1.0 - k / (n - 1)) * start[i] + k / (n - 1) * target[i])
            for k, i in enumerate(path)
        ])
        cost = robust if mode == "robust" else point
        scored.append((cost / (2 * (n - 1)) + lam * text, path))
    expected_objective, expected_path = min(scored)
    result = select_robust_sequence(
        median, lower, upper, start, target, num_tracks=n,
        start_index=0, target_index=5, direction=(1, -1),
        lambda_retrieval=lam, beam_width=100, mode=mode,
    )
    assert result.status == "sequence_generated"
    assert result.failure_reason is None
    assert result.indices == expected_path
    assert result.objective == pytest.approx(expected_objective, abs=1e-12)


def test_variable_n_unique_anchors_and_position_relevance():
    median = np.full((6, 2), 0.5)
    start = np.array([1.0, 0.95, 0.25, 0.55, 0.10, 0.0])
    target = np.array([0.0, 0.10, 0.25, 0.55, 0.95, 1.0])
    result = select_robust_sequence(
        median, median, median, start, target,
        num_tracks=4, start_index=0, target_index=5,
        direction=(None, None), lambda_retrieval=1.0, beam_width=50,
    )
    assert result.indices == (0, 1, 4, 5)
    assert len(set(result.indices)) == 4
    assert result.robust_path_cost == 0.0
    assert result.point_path_cost == 0.0
    assert result.objective == pytest.approx(result.retrieval_match_cost)


def test_single_track_two_tracks_insufficient_and_anchor_conflict():
    median = np.array([[0.2, 0.8], [0.8, 0.2]])
    lower, upper = median - 0.05, median + 0.05
    start, target = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    kwargs = dict(median=median, lower=lower, upper=upper,
                  start_relevance=start, target_relevance=target,
                  start_index=0, target_index=1, direction=(1, -1))
    single = select_robust_sequence(**kwargs, num_tracks=1)
    assert single.status == "single_track_retrieved"
    assert single.indices == (1,)
    assert single.robust_path_cost == single.point_path_cost == 0.0
    two = select_robust_sequence(**kwargs, num_tracks=2)
    assert two.indices == (0, 1)
    assert two.point_path_cost == pytest.approx(0.0)
    assert two.robust_path_cost > 0
    too_many = select_robust_sequence(**kwargs, num_tracks=3)
    assert too_many.status == "insufficient_candidates"
    assert too_many.indices == ()
    assert too_many.failure_reason == "candidate_pool_lt_requested_n"
    conflict = select_robust_sequence(**{**kwargs, "target_index": 0}, num_tracks=2)
    assert conflict.status == "anchor_direction_conflict"
    assert conflict.failure_reason == "start_and_target_are_same_song"


def test_rank_relevance_ties_are_deterministic():
    assert rank_relevance(np.array([2.0, 2.0, 1.0]), ["z", "a", "m"]).tolist() == [0.5, 1.0, 0.0]
    assert rank_relevance(np.array([1.0]), ["only"]).tolist() == [1.0]
    with pytest.raises(ValueError, match="unique"):
        rank_relevance(np.array([1.0, 2.0]), ["same", "same"])
