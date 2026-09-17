"""Deterministic preference-constrained mean-edge sequence selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Direction = tuple[int | None, int | None]


@dataclass(frozen=True)
class Candidate:
    track_id: str
    artist_id: str
    audio_path: str
    start_relevance: float
    target_relevance: float
    preference_percentile: float
    embedding: np.ndarray
    start_anchor: bool
    target_anchor: bool


@dataclass(frozen=True)
class SearchResult:
    status: str
    indices: tuple[int, ...]
    mean_edge: float | None
    weakest_edge: float | None
    endpoint_score: float | None
    mean_preference: float | None
    failure_reason: str | None


def normalize_direction(values: Sequence[int | None]) -> Direction:
    if len(values) != 2:
        raise ValueError("direction must contain valence and arousal")
    result = tuple(None if value is None else int(value) for value in values)
    if any(value not in {-1, 0, 1, None} for value in result):
        raise ValueError("direction values must be -1, 0, 1, or null")
    return result  # type: ignore[return-value]


def direction_from_va(current: np.ndarray, target: np.ndarray, tolerance: float) -> Direction:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)
    if delta.shape != (2,):
        raise ValueError("affect coordinates must be [valence, arousal]")
    return normalize_direction(tuple(0 if abs(value) <= tolerance else (1 if value > 0 else -1) for value in delta))


def _allowed_classes(value: int, endpoint: bool) -> tuple[int, ...]:
    if value == 1:
        return (3, 4) if endpoint else (2, 3)
    if value == -1:
        return (0, 1) if endpoint else (1, 2)
    if value == 0:
        # "Maintain" permits small fluctuations, while excluding large moves.
        # The narrow exact-equality class alone cannot reach the configured
        # confidence thresholds with a calibrated temperature >= 0.1.
        return (1, 2, 3)
    raise ValueError(value)


def directional_scores(probabilities: np.ndarray, direction: Direction, endpoint: bool) -> np.ndarray:
    """Reduce seed and active-axis probabilities conservatively."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 5 or values.shape[-2:] != (2, 5):
        raise ValueError("probabilities must be [seed,left,right,2,5]")
    axes = []
    for axis, requested in enumerate(normalize_direction(direction)):
        if requested is not None:
            axes.append(values[..., axis, list(_allowed_classes(requested, endpoint))].sum(axis=-1))
    if not axes:
        return np.ones(values.shape[1:3], dtype=np.float64)
    return np.stack(axes, axis=-1).min(axis=(0, 3))


def _secondary_key(indices: tuple[int, ...], candidates: Sequence[Candidate]) -> tuple:
    preference = float(np.mean([candidates[index].preference_percentile for index in indices]))
    endpoint = (candidates[indices[0]].start_relevance + candidates[indices[-1]].target_relevance) / 2.0
    counts: dict[str, int] = {}
    for index in indices:
        artist = candidates[index].artist_id.strip()
        if artist:
            counts[artist] = counts.get(artist, 0) + 1
    repetitions = sum(max(0, count - 1) for count in counts.values())
    distance = 0.0
    if len(indices) > 1:
        distance = float(np.mean([
            1.0 - float(np.dot(candidates[left].embedding, candidates[right].embedding))
            for left, right in zip(indices[:-1], indices[1:])
        ]))
    return (
        preference,
        float(endpoint),
        -repetitions,
        -distance,
        tuple(candidates[index].track_id for index in indices),
    )


def _failure(reason: str, indices: tuple[int, ...] = ()) -> SearchResult:
    return SearchResult("fallback_preference", indices, None, None, None, None, reason)


def select_music_sequence(
    candidates: Sequence[Candidate],
    edge_scores: np.ndarray,
    endpoint_scores: np.ndarray,
    *,
    preference_threshold: float = 0.80,
    endpoint_threshold: float = 0.50,
    edge_threshold: float = 0.40,
    search_width: int = 50,
    has_preference: bool = True,
    maintain: bool = False,
) -> SearchResult:
    if len(candidates) < 5:
        return SearchResult("insufficient_catalog", (), None, None, None, None, "candidate_pool_lt_5")
    eligible = [
        index for index, item in enumerate(candidates)
        if not has_preference or item.preference_percentile >= preference_threshold
    ]
    if len(eligible) < 5:
        return _failure("preference_filter_lt_5")
    start = [index for index in eligible if candidates[index].start_anchor]
    target = [index for index in eligible if candidates[index].target_anchor]
    if not start or not target:
        return _failure("endpoint_anchor_empty")
    if not any(
        left != right and endpoint_scores[left, right] >= endpoint_threshold
        for left in start for right in target
    ):
        return _failure("no_endpoint_pair")

    frontier: list[tuple[tuple[int, ...], float, float]] = [((index,), 0.0, 1.0) for index in start]
    for position in range(1, 5):
        expanded: list[tuple[tuple[int, ...], float, float]] = []
        options = target if position == 4 else eligible
        for indices, edge_sum, weakest in frontier:
            for index in options:
                if index in indices:
                    continue
                if position == 4 and endpoint_scores[indices[0], index] < endpoint_threshold:
                    continue
                edge = float(edge_scores[indices[-1], index])
                if edge < edge_threshold:
                    continue
                expanded.append(((*indices, index), edge_sum + edge, min(weakest, edge)))
        if not expanded:
            return _failure("low_edge_score")
        expanded.sort(
            key=lambda state: (
                -state[1] / (len(state[0]) - 1),
                -state[2],
                tuple(candidates[index].track_id for index in state[0]),
            )
        )
        frontier = expanded[:search_width]

    indices, edge_sum, weakest = max(
        frontier,
        key=lambda state: (state[1] / 4.0, state[2], _secondary_key(state[0], candidates)),
    )
    endpoint = float(endpoint_scores[indices[0], indices[-1]])
    mean_preference = float(np.mean([candidates[index].preference_percentile for index in indices]))
    status = "maintain_found" if maintain else "transition_found"
    if weakest < edge_threshold:
        return SearchResult(
            "fallback_preference", indices, edge_sum / 4.0, weakest,
            endpoint, mean_preference, "low_edge_score",
        )
    return SearchResult(status, indices, edge_sum / 4.0, weakest, endpoint, mean_preference, None)


def fallback_sequence(candidates: Sequence[Candidate], has_preference: bool) -> tuple[int, ...]:
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -candidates[index].preference_percentile if has_preference else -candidates[index].target_relevance,
            -candidates[index].target_relevance,
            candidates[index].track_id,
        ),
    )
    return tuple(order[:5])
