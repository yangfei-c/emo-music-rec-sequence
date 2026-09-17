from __future__ import annotations

import numpy as np

from emomm_sequence.search import Candidate, fallback_sequence, select_music_sequence


def candidates(count: int = 6) -> list[Candidate]:
    result = []
    for index in range(count):
        embedding = np.zeros(count, dtype=np.float32)
        embedding[index] = 1.0
        result.append(Candidate(
            track_id=f"track-{index}",
            artist_id=f"artist-{index}",
            audio_path=f"audio-{index}.mp3",
            start_relevance=1.0 - index / 10,
            target_relevance=index / 10,
            preference_percentile=0.90,
            embedding=embedding,
            start_anchor=index == 0,
            target_anchor=index == 4,
        ))
    return result


def test_sequence_selection_returns_five_unique_ordered_tracks() -> None:
    items = candidates()
    edges = np.full((6, 6), 0.10)
    for left, right in zip(range(4), range(1, 5)):
        edges[left, right] = 0.90
    endpoints = np.zeros((6, 6))
    endpoints[0, 4] = 0.90
    result = select_music_sequence(items, edges, endpoints)
    assert result.status == "transition_found"
    assert result.indices == (0, 1, 2, 3, 4)
    assert len(set(result.indices)) == 5
    assert result.mean_edge == 0.90


def test_low_edge_is_explicit_failure_and_fallback_is_deterministic() -> None:
    items = candidates()
    edges = np.full((6, 6), 0.20)
    endpoints = np.zeros((6, 6))
    endpoints[0, 4] = 0.90
    result = select_music_sequence(items, edges, endpoints)
    assert result.status == "fallback_preference"
    assert result.failure_reason == "low_edge_score"
    assert fallback_sequence(items, True) == (5, 4, 3, 2, 1)


def test_preference_filter_failure_is_reported() -> None:
    items = candidates()
    items = [
        Candidate(**{**item.__dict__, "preference_percentile": 0.1 if index else 0.9})
        for index, item in enumerate(items)
    ]
    result = select_music_sequence(items, np.ones((6, 6)), np.ones((6, 6)))
    assert result.status == "fallback_preference"
    assert result.failure_reason == "preference_filter_lt_5"


def test_invalid_high_mean_path_does_not_hide_a_valid_path() -> None:
    items = candidates(7)
    edges = np.zeros((7, 7))
    for left, right in zip(range(4), range(1, 5)):
        edges[left, right] = 0.60
    edges[0, 5] = edges[5, 6] = 0.99
    edges[6, 3] = 0.10
    endpoints = np.zeros((7, 7))
    endpoints[0, 4] = 0.90
    result = select_music_sequence(items, edges, endpoints)
    assert result.status == "transition_found"
    assert result.indices == (0, 1, 2, 3, 4)
    assert result.weakest_edge >= 0.40
