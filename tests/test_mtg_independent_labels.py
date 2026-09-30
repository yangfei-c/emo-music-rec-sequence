"""Guard the independent categorical-label interpretation."""

from __future__ import annotations

import pytest

from scripts.evaluation.evaluate_mtg_independent_labels import (
    label_state,
    load_consensus_moods,
    summarize_cases,
)


HEADER = "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tANNOTATIONS\n"


def test_consensus_parser_keeps_negative_distinct_from_unobserved(tmp_path):
    path = tmp_path / "labels.tsv"
    path.write_text(
        HEADER
        + "track_1\tartist_1\talbum_1\t01/1.mp3\t120\t"
        + "mood_sad---sad,sad,sad\t"
        + "mood_happy---not_happy,not_happy,not_happy\n",
        encoding="utf-8",
    )
    labels = load_consensus_moods(path)
    assert label_state(labels, "track_1", "sad") == "positive"
    assert label_state(labels, "track_1", "happy") == "negative"
    assert label_state(labels, "track_1", "relaxed") == "unobserved"
    assert label_state(labels, "unknown", "sad") == "unobserved"


def test_parser_rejects_disagreement_even_if_input_is_named_clean(tmp_path):
    path = tmp_path / "labels.tsv"
    path.write_text(
        HEADER
        + "track_1\tartist_1\talbum_1\t01/1.mp3\t120\t"
        + "mood_sad---sad,not_sad,sad\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-consensus"):
        load_consensus_moods(path)


def test_summary_uses_all_and_observed_denominators_separately():
    cases = [
        {"first_expected_label": "positive", "last_expected_label": "positive",
         "both_endpoints_observed": True, "both_endpoints_positive": True},
        {"first_expected_label": "unobserved", "last_expected_label": "negative",
         "both_endpoints_observed": False, "both_endpoints_positive": False},
    ]
    summary = summarize_cases(cases)
    assert summary["endpoint_label_observed_rate"] == 0.75
    assert summary["both_endpoints_positive_rate_all_requests"] == 0.5
    assert summary["both_endpoints_positive_rate_both_observed"] == 1.0
