"""Instrumentation must preserve the selected songs and restore wrapped calls."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from emomm_sequence.data.sequence_tasks import Bank
from emomm_sequence.data import sequence_va_tasks as va_tasks
from emomm_sequence.models.sequence_selector import SequenceSelector

spec = importlib.util.spec_from_file_location("va_latency", Path(__file__).parents[1] / "scripts/evaluation/evaluate_sequence_va_latency.py")
latency = importlib.util.module_from_spec(spec)
spec.loader.exec_module(latency)


def example(mode):
    random = np.random.default_rng(42)
    ids = np.array([f"song_{i}" for i in range(8)])
    median = random.uniform(.2, .8, (8, 2))
    radius = {"point": 0., "raw_interval": .05, "cqr_interval": .12}[mode]
    problem = va_tasks.VASequenceTask(ids, median, median-radius, median+radius,
                                     np.linspace([.2, .3], [.7, .6], 4), random.uniform(0, 1, 8), 0, 7)
    problem.x = random.normal(size=(8, 9)).astype("float32")
    problem.query = random.normal(size=(3, 9)).astype("float32")
    problem.nodes = .5 * np.maximum(np.abs(problem.lower[None]-problem.target_path[:, None]),
                                     np.abs(problem.upper[None]-problem.target_path[:, None])).mean(axis=-1) / 4
    bank = Bank("fixture", "validation", ids, ids.copy(), problem.x)
    task = {"query_id": "fixture_1", "query_family_id": "fixture", "source": "fixture", "suite": "e2",
            "split": "validation", "budget": 6}
    inputs = SimpleNamespace(mode=mode, make=lambda *_: (bank, problem, task))
    return inputs, bank, task, problem


@pytest.mark.parametrize("mode", latency.MODES)
@pytest.mark.parametrize("method", latency.METHODS)
def test_instrumentation_preserves_production_output(mode, method):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    inputs, bank, task, problem = example(mode)
    model = SequenceSelector(embedding_dim=9, hidden_dim=8, layers=1, heads=2).eval()
    if method == "b_set":
        with torch.inference_mode():
            chosen = model.select(torch.from_numpy(problem.x), torch.from_numpy(problem.query), 4, 6, 0, 7)
        candidates = np.asarray(chosen.indices)
    else:
        candidates = va_tasks.va_candidates(problem, 6)
    original_search = va_tasks.va_beam_search
    if method == "a_refine":
        candidates, expected = va_tasks.va_refine(problem, candidates, width=10)
    else:
        expected = original_search(problem, candidates, width=10)
    actual = latency.measure_request(inputs, bank, task, {}, method, model, "cpu", 10)
    assert va_tasks.va_beam_search is original_search
    assert actual["track_ids"] == bank.ids[list(expected.path)].tolist()
    assert actual["candidate_ids"] == bank.ids[candidates].tolist()
    assert actual["objective"] == expected.objective
    assert actual["solver_calls"] == (3 if method == "a_refine" else 1)
    assert actual["search_expansions"] >= expected.expansions
    assert actual["search_seconds"] > 0
    assert actual["total_seconds"] >= actual["search_seconds"] + actual["candidate_seconds"]
    assert actual["peak_cuda_mib"] == 0


def test_failed_refinement_restores_production_search(monkeypatch):
    inputs, bank, task, _ = example("point")
    original = va_tasks.va_beam_search
    def fail(*args):
        raise RuntimeError("injected failure")
    monkeypatch.setattr(va_tasks, "va_refine", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        latency.measure_request(inputs, bank, task, {}, "a_refine", None, "cpu", 10)
    assert va_tasks.va_beam_search is original


def test_parity_rejects_changed_anchor_and_objective():
    inputs, bank, task, _ = example("point")
    row = latency.measure_request(inputs, bank, task, {}, "a_grid", None, "cpu", 10)
    totals = {"max_absolute_objective_difference": 0., "different_paths": 0,
              "different_candidate_sets": 0, "compared_outputs": 0}
    latency.verify_parity(row, row.copy(), totals)
    assert totals["compared_outputs"] == 1
    with pytest.raises(ValueError, match="contract"):
        latency.verify_parity(row, {**row, "start_id": "wrong"}, totals)
    with pytest.raises(ValueError, match="objective changed"):
        latency.verify_parity(row, {**row, "objective": row["objective"] + .1}, totals)
