from __future__ import annotations

import pytest

from emomm_sequence.training.semantic_views import (
    lyric_window_candidates,
    normalize_text,
    rank_view_candidates,
)


def test_lyric_views_are_distinct_deterministic_windows() -> None:
    text = "  " + " \n ".join(f"word{index}" for index in range(20)) + "  "
    views = lyric_window_candidates(
        text, window_words=8, candidate_count=3, min_lyrics_words=10
    )
    assert views == [
        "word0 word1 word2 word3 word4 word5 word6 word7",
        "word6 word7 word8 word9 word10 word11 word12 word13",
        "word12 word13 word14 word15 word16 word17 word18 word19",
    ]
    assert all(view != normalize_text(text) for view in views)


def test_short_lyrics_are_skipped_and_near_window_text_is_fragmented() -> None:
    assert lyric_window_candidates(
        "one two three", window_words=8, candidate_count=3, min_lyrics_words=4
    ) == []
    views = lyric_window_candidates(
        "one two three four five six seven eight",
        window_words=8,
        candidate_count=2,
        min_lyrics_words=4,
    )
    assert views == [
        "one two three four five six",
        "three four five six seven eight",
    ]


def test_view_ranking_combines_audio_and_text_with_stable_ties() -> None:
    ranking = rank_view_candidates([0.8, 0.4, 0.6], [0.2, 0.6, 0.4])
    assert ranking == [0, 1, 2]
    assert rank_view_candidates([0.1, 0.9], [0.9, 0.1], audio_weight=0.8) == [1, 0]
    with pytest.raises(ValueError):
        rank_view_candidates([0.1], [0.2, 0.3])
