import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("sequence_report", Path(__file__).resolve().parents[1] / "scripts/evaluation/report_sequence_routes.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_repeating_same_request_on_three_sources_does_not_triple_sample_size():
    values = {"sad_happy": -.1, "happy_calm": .03, "calm_tense": .02}
    one = report.family_bootstrap({"a": values}, repeats=500)
    three = report.family_bootstrap({"a": values, "b": values, "c": values}, repeats=500)
    assert three["query_family_clusters"] == 3
    import numpy as np
    assert np.allclose(one["ci95"], three["ci95"])


def test_incomplete_shared_families_do_not_produce_false_ci():
    result = report.family_bootstrap({"a": {"x": 1, "y": 2}, "b": {"x": 1}})
    assert result["ci95"] is None
