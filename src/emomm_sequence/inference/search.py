"""Length-conditioned playlist search with a consistent interval adversary.

The two-state dynamic program scores a *fixed* playlist exactly. Beam search
over playlists is approximate; its width is a computation budget, not a claim
of global optimality. All arrays passed to the planner refer to the same
candidate pool and use the music-side [0, 1] VA scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Direction = Sequence[int | None]


@dataclass(frozen=True)
class RobustSearchResult:
    status: str
    indices: tuple[int, ...]
    robust_path_cost: float | None
    point_path_cost: float | None
    retrieval_match_cost: float | None
    objective: float | None
    reference_step: tuple[float, float]
    failure_reason: str | None = None


@dataclass(frozen=True)
class _State:
    indices: tuple[int, ...]
    dp: np.ndarray  # [axis, endpoint state]
    point_cost: float
    text_cost: float
    beam_score: float


def rank_relevance(scores: np.ndarray, track_ids: Sequence[str]) -> np.ndarray:
    """Return candidate-pool rank scores in [0, 1], breaking ties by ID."""
    values = np.asarray(scores, dtype=np.float64)
    ids = np.asarray(track_ids, dtype=str)
    if values.ndim != 1 or ids.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("scores and track_ids must be aligned finite vectors")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("track_ids must be unique")
    if len(values) == 0:
        return np.empty(0, dtype=np.float64)
    if len(values) == 1:
        return np.ones(1, dtype=np.float64)
    order = np.lexsort((ids, -values))
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values))
    return 1.0 - ranks / (len(values) - 1)


def _path_inputs(
    values: np.ndarray, reference_step: np.ndarray, active_axes: Sequence[bool]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = np.asarray(values, dtype=np.float64)
    step = np.asarray(reference_step, dtype=np.float64)
    active = np.asarray(active_axes, dtype=bool)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 1:
        raise ValueError("path must have shape [num_tracks, 2]")
    if step.shape != (2,) or active.shape != (2,):
        raise ValueError("reference_step and active_axes must have two axes")
    if not np.isfinite(path).all() or not np.isfinite(step).all():
        raise ValueError("path and reference_step must be finite")
    return path, step, active


def point_path_cost(
    path_median: np.ndarray,
    reference_step: np.ndarray,
    active_axes: Sequence[bool] = (True, True),
) -> float:
    """Total absolute point-VA deviation from a fixed per-step target."""
    median, step, active = _path_inputs(path_median, reference_step, active_axes)
    if len(median) == 1 or not active.any():
        return 0.0
    return float(np.abs(np.diff(median, axis=0) - step)[..., active].sum())


def _extend_dp(
    previous_dp: np.ndarray,
    previous_ends: np.ndarray,
    new_ends: np.ndarray,
    reference_step: np.ndarray,
) -> np.ndarray:
    """Update both endpoint states for one or more candidate songs at once."""
    # new_ends: [option, axis, endpoint]; previous_ends: [axis, endpoint].
    deviations = np.abs(
        new_ends[:, :, None, :]
        - previous_ends[None, :, :, None]
        - reference_step[None, :, None, None]
    )
    return np.max(previous_dp[None, :, :, None] + deviations, axis=2)


def robust_path_cost(
    path_lower: np.ndarray,
    path_upper: np.ndarray,
    reference_step: np.ndarray,
    active_axes: Sequence[bool] = (True, True),
) -> float:
    """Exact worst-case full-path deviation over per-song VA rectangles.

    Each song's VA value is shared by its incoming and outgoing edges. The
    objective is convex in every song coordinate, so a maximizer exists at a
    rectangle vertex; the two-state DP enumerates those vertices implicitly.
    """
    lower, step, active = _path_inputs(path_lower, reference_step, active_axes)
    upper, other_step, other_active = _path_inputs(path_upper, reference_step, active_axes)
    if upper.shape != lower.shape or np.any(lower > upper):
        raise ValueError("lower and upper must be aligned ordered intervals")
    if not np.array_equal(step, other_step) or not np.array_equal(active, other_active):
        raise AssertionError("inconsistent path inputs")
    if len(lower) == 1 or not active.any():
        return 0.0
    dp = np.zeros((2, 2), dtype=np.float64)
    for k in range(1, len(lower)):
        previous = np.stack((lower[k - 1], upper[k - 1]), axis=-1)
        current = np.stack((lower[k], upper[k]), axis=-1)[None]
        dp = _extend_dp(dp, previous, current, step)[0]
    return float(np.max(dp, axis=1)[active].sum())


def _failed(status: str, reason: str, step: np.ndarray | None = None) -> RobustSearchResult:
    values = (0.0, 0.0) if step is None else (float(step[0]), float(step[1]))
    return RobustSearchResult(status, (), None, None, None, None, values, reason)


def select_robust_sequence(
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    start_relevance: np.ndarray,
    target_relevance: np.ndarray,
    *,
    num_tracks: int,
    start_index: int,
    target_index: int,
    direction: Direction = (1, 1),
    lambda_retrieval: float = 1.0,
    beam_width: int = 50,
    track_ids: Sequence[str] | None = None,
    mode: str = "robust",
) -> RobustSearchResult:
    """Search unique-song paths with fixed music-side anchors.

    ``direction`` has one entry per axis: -1 or +1 uses the anchored median
    difference divided by N-1, 0 targets zero change, and None omits that
    axis. ``mode='point'`` uses the identical beam and objective with the
    point-VA loss, for an apples-to-apples baseline. Returned path costs are
    *totals*; ``retrieval_match_cost`` is the mean positional text cost.
    """
    if isinstance(num_tracks, (bool, np.bool_)) or not isinstance(num_tracks, (int, np.integer)) or num_tracks < 1:
        raise ValueError("num_tracks must be a positive integer")
    if isinstance(beam_width, (bool, np.bool_)) or not isinstance(beam_width, (int, np.integer)) or beam_width < 1:
        raise ValueError("beam_width must be a positive integer")
    if not np.isfinite(lambda_retrieval) or lambda_retrieval < 0:
        raise ValueError("lambda_retrieval must be finite and nonnegative")
    if mode not in {"robust", "point"}:
        raise ValueError("mode must be 'robust' or 'point'")
    if len(direction) != 2 or any(value not in {-1, 0, 1, None} for value in direction):
        raise ValueError("direction must contain two entries from {-1, 0, 1, None}")

    m = np.asarray(median, dtype=np.float64)
    lo = np.asarray(lower, dtype=np.float64)
    hi = np.asarray(upper, dtype=np.float64)
    start = np.asarray(start_relevance, dtype=np.float64)
    target = np.asarray(target_relevance, dtype=np.float64)
    if m.ndim != 2 or m.shape[1] != 2 or lo.shape != m.shape or hi.shape != m.shape:
        raise ValueError("median, lower, upper must have shape [candidate, 2]")
    count = len(m)
    if start.shape != (count,) or target.shape != (count,):
        raise ValueError("relevance vectors must align with candidates")
    if not all(np.isfinite(array).all() for array in (m, lo, hi, start, target)):
        raise ValueError("candidate values must be finite")
    if np.any(lo < 0) or np.any(hi > 1) or np.any(lo > m) or np.any(m > hi):
        raise ValueError("VA bounds must satisfy 0 <= lower <= median <= upper <= 1")
    if np.any(start < 0) or np.any(start > 1) or np.any(target < 0) or np.any(target > 1):
        raise ValueError("relevance scores must be in [0, 1]")
    if track_ids is None:
        ids = np.asarray([str(i) for i in range(count)], dtype=str)
    else:
        ids = np.asarray(track_ids, dtype=str)
        if ids.shape != (count,) or len(set(ids.tolist())) != count:
            raise ValueError("track_ids must be unique and aligned with candidates")
    if num_tracks > count:
        return _failed("insufficient_candidates", "candidate_pool_lt_requested_n")
    if not 0 <= start_index < count or not 0 <= target_index < count:
        raise ValueError("anchor indices must refer to candidates")
    if num_tracks > 1 and start_index == target_index:
        return _failed("anchor_direction_conflict", "start_and_target_are_same_song")

    active = np.asarray([value is not None for value in direction], dtype=bool)
    step = np.zeros(2, dtype=np.float64)
    if num_tracks > 1:
        anchored_delta = m[target_index] - m[start_index]
        if any(value in (-1, 1) and value * anchored_delta[axis] < -1.0e-12 for axis, value in enumerate(direction)):
            return _failed("anchor_direction_conflict", "anchors_oppose_requested_direction")
        for axis, value in enumerate(direction):
            if value in (-1, 1):
                step[axis] = anchored_delta[axis] / (num_tracks - 1)

    if num_tracks == 1:
        index = target_index
        text_cost = 1.0 - float(target[index])
        return RobustSearchResult(
            "single_track_retrieved", (index,), 0.0, 0.0, text_cost,
            float(lambda_retrieval * text_cost), (0.0, 0.0), None,
        )

    # Lexicographic track IDs provide stable ties independent of input order.
    id_order = np.argsort(ids, kind="stable")
    id_rank = np.empty(count, dtype=np.int64)
    id_rank[id_order] = np.arange(count)
    bounds = np.stack((lo, hi), axis=-1)  # [candidate, axis, endpoint]
    normalization = int(active.sum()) * (num_tracks - 1)

    def path_score(dp: np.ndarray, point_cost: float | np.ndarray) -> float | np.ndarray:
        if mode == "point":
            return point_cost
        return np.max(dp, axis=-1)[..., active].sum(axis=-1)

    def normalized_cost(cost: float | np.ndarray) -> float | np.ndarray:
        return cost / normalization if normalization else np.zeros_like(cost, dtype=np.float64)

    def text_step(indices: np.ndarray, position: int) -> np.ndarray:
        u = position / (num_tracks - 1)
        return 1.0 - ((1.0 - u) * start[indices] + u * target[indices])

    first = _State(
        (start_index,), np.zeros((2, 2), dtype=np.float64), 0.0,
        float(1.0 - start[start_index]), 0.0,
    )
    frontier = [first]
    for position in range(1, num_tracks):
        final = position == num_tracks - 1
        next_frontier: list[_State] = []
        for state in frontier:
            if final:
                if target_index in state.indices:
                    continue
                options = np.asarray([target_index], dtype=np.int64)
            else:
                used = set(state.indices)
                options = np.asarray(
                    [i for i in range(count) if i != target_index and i not in used],
                    dtype=np.int64,
                )
            if len(options) == 0:
                continue
            previous = state.indices[-1]
            updated = _extend_dp(state.dp, bounds[previous], bounds[options], step)
            point_increment = np.abs(m[options] - m[previous] - step)[:, active].sum(axis=1)
            point_totals = state.point_cost + point_increment
            text_totals = state.text_cost + text_step(options, position)
            selected_costs = path_score(updated, point_totals)
            partial = normalized_cost(selected_costs) + lambda_retrieval * text_totals / num_tracks
            remaining_edges = num_tracks - position - 1
            if remaining_edges and active.any():
                # Triangle-inequality slack is used only for beam ordering.
                # This is not asserted to be an admissible robust-cost bound.
                slack = np.abs(m[target_index] - m[options] - remaining_edges * step)[:, active].sum(axis=1)
                partial = partial + slack / normalization
            local_order = np.lexsort((id_rank[options], partial))[:beam_width]
            for local in local_order:
                index = int(options[local])
                next_frontier.append(_State(
                    (*state.indices, index), updated[local].copy(),
                    float(point_totals[local]), float(text_totals[local]), float(partial[local]),
                ))
        if not next_frontier:
            return _failed("insufficient_candidates", "no_unique_path_to_target", step)
        next_frontier.sort(key=lambda state: (
            state.beam_score, tuple(ids[list(state.indices)].tolist()),
        ))
        frontier = next_frontier[:beam_width]

    # The final beam score is the complete J, with no look-ahead heuristic.
    best = min(frontier, key=lambda state: (
        state.beam_score, tuple(ids[list(state.indices)].tolist()),
    ))
    chosen = np.asarray(best.indices, dtype=np.int64)
    robust = float(np.max(best.dp, axis=1)[active].sum()) if active.any() else 0.0
    point = float(best.point_cost)
    retrieval = float(best.text_cost / num_tracks)
    chosen_cost = robust if mode == "robust" else point
    objective = float(normalized_cost(chosen_cost) + lambda_retrieval * retrieval)
    return RobustSearchResult(
        "sequence_generated", best.indices, robust, point, retrieval,
        objective, (float(step[0]), float(step[1])), None,
    )
