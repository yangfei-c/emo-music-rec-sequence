"""Budget saturation must not collapse distinct requested-budget comparisons."""
import importlib.util
from pathlib import Path

import pytest


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


prepare = module("budget_prepare", "scripts/data/prepare_sequence_budget_curve.py")
report = module("budget_report", "scripts/evaluation/report_sequence_budget_curve.py")


def test_curve_preserves_requested_budgets_and_family_identity():
    base = {"suite": "e2", "source": "pmemo", "split": "validation", "phrase_index": 2,
            "query_family_id": "f1", "query_id": "old_1", "num_tracks": 3, "budget": 32,
            "start_text_id": 0, "target_text_id": 1, "preference_text_id": None, "indices": None}
    rows = prepare.build_tasks([base, {**base, "query_id": "old_2", "num_tracks": 10, "budget": 64}], "pilot", "validation")
    assert len(rows) == 8 and len({r["query_id"] for r in rows}) == 8
    assert {r["budget"] for r in rows} == {32, 64, 128, 256}
    assert {r["query_family_id"] for r in rows} == {"f1"}
    assert {r["origin_query_id"] for r in rows} == {"old_1"}
    with pytest.raises(ValueError, match="changes more"):
        prepare.build_tasks([base, {**base, "target_text_id": 2}], "pilot", "validation")


def observed():
    return [{"method": method, "suite": "e2", "source": "pmemo", "num_tracks": 5,
             "requested_budget": budget, "budget": 116, "library_size": 116, "budget_saturated": True,
             "query_id": f"f1_{budget}", "query_family_id": "f1", "split": "validation", "seed": 42,
             "start_id": "s", "target_id": "t", "phrase_index": 2, "status": "ok", "total_seconds": .1,
             "objective": (.2 if budget == 128 else .3) if method == "b_set" else .1,
             "true_va_mae": .2} for budget in (128, 256) for method in report.METHODS]


def test_saturated_budgets_stay_separate_and_missing_methods_fail():
    rows = observed()
    table, comparisons = report.aggregate(rows)
    assert len(table) == 10
    assert {r["requested_budget"] for r in table} == {128, 256}
    for row in comparisons:
        if row["metric"] == "objective":
            assert row["mean"] == pytest.approx(.1 if row["requested_budget"] == 128 else .2)
            assert row["paired_requests"] == 1
    with pytest.raises(ValueError, match="all five methods"):
        report.aggregate(rows[:-1])


def test_failure_penalty_retained_without_inventing_label_quality():
    rows = observed()
    rows = [{**r, "status": "timeout", "objective": 1., "true_va_mae": None}
            if r["method"] == "b_set" and r["requested_budget"] == 256 else r for r in rows]
    table, comparisons = report.aggregate(rows)
    failed = next(r for r in table if r["method"] == "b_set" and r["requested_budget"] == 256)
    assert failed["failed"] == 1 and failed["objective"] == 1. and failed["true_va_mae"] is None
    for row in comparisons:
        if row["requested_budget"] == 256:
            assert row["metric"] == "objective" and row["mean"] == pytest.approx(.9)
