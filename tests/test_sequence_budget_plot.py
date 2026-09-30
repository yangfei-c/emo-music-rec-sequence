"""Synthetic in-memory fixtures test figure input guards; no figures are made."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("budget_plot", Path(__file__).parents[1]/"scripts/evaluation/plot_sequence_budget_curve.py")
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def inputs():
    tasks = [{"query_id": f"{s}_{n}_{b}_{i}", "query_family_id": f"family_{i}", "suite": suite,
              "source": s, "num_tracks": n, "budget": b}
             for suite, sources in plot.SOURCES.items() for s in sources
             for n in plot.LENGTHS for b in plot.BUDGETS for i in range(4)]
    measurements = [{**t, "method": m, "requested_budget": t["budget"], "timing_repeat": repeat,
                     "status": "ok", "total_seconds": .001*t["budget"]}
                    for t in tasks for m in plot.METHODS for repeat in range(1, 6)]
    table = []
    for suite, sources in plot.SOURCES.items():
        for source in sources:
            size = {"deam": 271, "pmemo": 116, "merge": 332, "mtg": 9856}[source]
            for m in plot.METHODS:
                for n in plot.LENGTHS:
                    for b in plot.BUDGETS:
                        table.append({"method": m, "suite": suite, "source": source, "num_tracks": n,
                                      "requested_budget": b, "budget": min(b, size), "failed": 0, "requests": 8,
                                      "saturated": b >= size, "true_va_mae": .2, "true_va_mae_observed_requests": 8,
                                      "observed_preference_tag_match": .3, "observed_preference_tag_match_observed_requests": 8})
    comparisons = [{"suite": "e2", "metric": "objective", "contrast": f"b_set_minus_{m}",
                    "num_tracks": n, "requested_budget": b, "mean": -.001, "ci95": [-.002, .0001]}
                   for m in plot.METHODS if m != "b_set" for n in plot.LENGTHS for b in plot.BUDGETS]
    summary = {"status": "completed", "failed": 0, "run_id": "fixture", "task_sha256": "fixture_hash",
               "tables": table, "paired_comparisons": comparisons}
    timing = {"status": "completed", "overlap_samples": [], "measurements": len(measurements),
              "fingerprint": {"methods": list(plot.METHODS), "fitted_seed": 42, "repeats": 5, "warmup_passes": 1,
                              "per_stratum": 4, "main_run_id": "fixture", "source_tasks_sha256": "fixture_hash"}}
    return summary, timing, measurements, tasks


def test_complete_curves_preserve_budget_and_measurement_units():
    values = inputs()
    points, comparisons = plot.prepare_points(*values)
    assert len(points) == 80 and len(comparisons) == 32
    selected = [r for r in points if r["method"] == "b_set" and r["suite"] == "e2" and r["num_tracks"] == 5]
    assert [r["requested_budget"] for r in selected] == [32, 64, 128, 256]
    for row in selected:
        assert row["quality_requests"] == 24
        assert row["timing_requests"] == 12 and row["measurements"] == 60
        assert row["timing_request_family_clusters"] == 4
    assert selected[-1]["effective_budget_by_source"]["pmemo"] == 116
    assert selected[-1]["saturated_sources"] == ["pmemo"]


@pytest.mark.parametrize("problem", ["missing_measurement", "duplicate_quality", "nonpositive_time", "overlap", "failure", "missing_stratum"])
def test_incomplete_or_incompatible_evidence_cannot_be_plotted(problem):
    summary, timing, measurements, tasks = inputs()
    if problem == "missing_measurement":
        measurements.pop()
    elif problem == "duplicate_quality":
        summary["tables"].append(copy.deepcopy(summary["tables"][0]))
    elif problem == "nonpositive_time":
        measurements[0]["total_seconds"] = 0.
    elif problem == "overlap":
        timing["overlap_samples"].append({"pid": 999})
    elif problem == "failure":
        summary["failed"] = 1
    else:
        tasks.pop()
    with pytest.raises(ValueError):
        plot.prepare_points(summary, timing, measurements, tasks)
