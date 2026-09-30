import itertools
import numpy as np
import pytest
from emomm_sequence.losses.sequence_cost import make_problem
from emomm_sequence.inference.sequence_planner import exact_search, beam_search
from emomm_sequence.inference.trajectory_candidates import select_candidates, refine


def fixture_problem(n=4):
    rng = np.random.default_rng(5)
    return make_problem(rng.normal(size=(8, 6)), np.array([str(i) for i in range(8)]),
                        rng.normal(size=6), rng.normal(size=6), None, n, endpoints=(0, 7))


def test_exact_matches_enumeration_including_terminal_edge():
    p = fixture_problem()
    paths = [(0, *mid, 7) for mid in itertools.permutations(range(1, 7), 2)]
    expected = min(p.cost(path) for path in paths)
    actual = exact_search(p)
    beam = beam_search(p, np.arange(8), width=100)
    assert actual.certified
    assert actual.objective == pytest.approx(expected, abs=2e-7)
    assert beam.objective == pytest.approx(expected, abs=2e-7)


def test_pool_does_not_change_node_cost_and_gap_nonnegative():
    p = fixture_problem()
    c = select_candidates(p, 5)
    sub = make_problem(p.x[c], p.ids[c], p.query[0], p.query[1], None, 4,
                       endpoints=(int(np.flatnonzero(c == p.start)[0]), int(np.flatnonzero(c == p.target)[0])))
    assert np.allclose(p.nodes[:, c], sub.nodes, atol=1e-7)
    assert exact_search(p, c).objective >= exact_search(p).objective - 1e-7


def test_refinement_retains_legality_and_does_not_increase_cost():
    p = fixture_problem(5)
    c = select_candidates(p, 6)
    before = beam_search(p, c)
    after_c, after, stats = refine(p, c)
    assert after.objective <= before.objective + 1e-8
    assert len(set(after.path)) == 5 and len(after_c) <= 6
    assert stats["solver_calls"] == 3


@pytest.mark.parametrize("n", [1, 2])
def test_short_sequences(n):
    p = fixture_problem(n)
    result = exact_search(p)
    assert len(result.path) == n
