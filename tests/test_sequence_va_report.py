import importlib.util
from pathlib import Path
import sys

import pytest


scripts = Path(__file__).resolve().parents[1] / "scripts" / "evaluation"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("va_extensions", scripts / "report_sequence_extensions.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def row(method, value, seed=42):
    return {"split": "validation", "mode": "point", "method": method, "query_id": "q1",
            "query_family_id": "f1", "source": "deam", "suite": "e2", "seed": seed,
            "status": "ok", "start_id": "a", "target_id": "b", "library_size": 30,
            "va_endpoint_protocol": report.VA_ENDPOINT_PROTOCOL, "objective": value,
            "true_va_mae": value, "selected_song_joint_coverage": 0.0}


def test_va_report_averages_seeds_and_excludes_point_coverage():
    result = report.paired_va_comparisons([row("a_grid", .4), row("b_set", .1), row("b_set", .3, 43)])
    assert {r["metric"] for r in result} == {"objective", "true_va_mae"}
    assert all(r["mean"] == pytest.approx(-.2) for r in result)
    assert all(r["paired_requests"] == 1 for r in result)


def test_va_report_rejects_incompatible_anchors():
    other = row("b_set", .1)
    other["start_id"] = "different"
    with pytest.raises(ValueError, match="endpoints"):
        report.paired_va_comparisons([row("a_grid", .4), other])


def test_va_report_keeps_failure_penalty_and_count():
    failed = row("b_set", None)
    failed["status"] = "failed"
    result = report.paired_va_comparisons([row("a_grid", .4), failed])
    assert len(result) == 1
    assert result[0]["metric"] == "objective"
    assert result[0]["mean"] == pytest.approx(.6)
    assert result[0]["requests_with_failure"] == 1
