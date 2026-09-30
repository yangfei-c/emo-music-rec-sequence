import numpy as np
from emomm_sequence.data.sequence_tasks import Bank
from emomm_sequence.inference.sequence_runtime import encoded_recommend


def test_runtime_uses_shared_cost_and_returns_fixed_unique_length():
    from emomm_sequence.losses.sequence_cost import make_problem
    from emomm_sequence.inference.sequence_planner import exact_search
    rng = np.random.default_rng(23)
    bank = Bank("synthetic", "test", np.array([f"t{i}" for i in range(8)]),
                np.array([f"g{i}" for i in range(8)]), rng.normal(size=(8, 12)).astype("float32"))
    text = rng.normal(size=(3, 12)).astype("float32")
    result = encoded_recommend(bank, text, 5, 8, "a_grid", width=1000)
    problem = make_problem(bank.x, bank.ids, *text, 5)
    oracle = exact_search(problem)
    assert len(set(result["content_group_ids"])) == 5
    assert abs(result["parts"]["objective"] - oracle.objective) < 1e-7
    for n in (1, 2):
        short = encoded_recommend(bank, text, n, 8, "a_refine", has_preference=False)
        assert len(short["track_ids"]) == n
