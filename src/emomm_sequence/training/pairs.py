"""Leakage-safe within-source pair sampling and ordinal labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PairBatch:
    left: np.ndarray
    right: np.ndarray
    source: np.ndarray
    mode: np.ndarray
    labels5: np.ndarray
    labels3: np.ndarray


def training_thresholds(
    targets: np.ndarray, sources: np.ndarray, splits: np.ndarray
) -> dict[str, dict[str, list[float]]]:
    """Compute source/axis thresholds from train songs only."""

    result: dict[str, dict[str, list[float]]] = {}
    for source in sorted(set(sources.astype(str))):
        mask = (sources == source) & (splits == "train")
        if int(mask.sum()) < 4:
            raise ValueError(f"source {source!r} has too few train songs")
        q75, q25 = np.percentile(targets[mask], [75, 25], axis=0)
        iqr = q75 - q25
        if np.any(iqr <= 1.0e-8):
            raise ValueError(f"source {source!r} has a degenerate VA IQR")
        result[source] = {
            "iqr": iqr.astype(float).tolist(),
            "small": (0.10 * iqr).astype(float).tolist(),
            "big": (0.75 * iqr).astype(float).tolist(),
        }
    return result


def label_pair_differences(
    differences: np.ndarray,
    source_names: np.ndarray,
    thresholds: dict[str, dict[str, list[float]]],
) -> tuple[np.ndarray, np.ndarray]:
    labels5 = np.empty(differences.shape, dtype=np.int64)
    labels3 = np.empty(differences.shape, dtype=np.int64)
    for source in sorted(set(source_names.astype(str))):
        mask = source_names == source
        small = np.asarray(thresholds[source]["small"], dtype=np.float64)
        big = np.asarray(thresholds[source]["big"], dtype=np.float64)
        values = differences[mask]
        labels5[mask] = np.where(
            values < -big,
            0,
            np.where(values < -small, 1, np.where(values <= small, 2, np.where(values <= big, 3, 4))),
        )
        labels3[mask] = np.where(values < -small, 0, np.where(values <= small, 1, 2))
    return labels5, labels3


def fixed_evaluation_labels(
    differences: np.ndarray, small: float = 0.05, big: float = 0.25
) -> np.ndarray:
    """Dataset-independent labels used only for held-out external evaluation."""

    return np.where(
        differences < -big,
        0,
        np.where(
            differences < -small,
            1,
            np.where(differences <= small, 2, np.where(differences <= big, 3, 4)),
        ),
    ).astype(np.int64)


def _add_pair(
    pair: tuple[int, int], selected: set[tuple[int, int]], degree: np.ndarray, cap: int
) -> bool:
    left, right = sorted(pair)
    if left == right or (left, right) in selected or degree[left] >= cap or degree[right] >= cap:
        return False
    selected.add((left, right))
    degree[left] += 1
    degree[right] += 1
    return True


def sample_unordered_pairs(
    indices: np.ndarray,
    preference_embeddings: np.ndarray,
    *,
    seed: int,
    max_pairs: int = 100_000,
    max_pairs_per_song: int = 32,
    neighbor_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample random and preference-neighborhood pairs without using VA labels."""

    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) < 2:
        raise ValueError("pair sampling requires at least two songs")
    if not 0.0 <= neighbor_fraction <= 1.0:
        raise ValueError("neighbor_fraction must be in [0,1]")
    rng = np.random.default_rng(seed)
    local_pref = preference_embeddings[indices].astype(np.float64, copy=True)
    local_pref /= np.maximum(np.linalg.norm(local_pref, axis=1, keepdims=True), 1.0e-12)
    # Use half of the theoretical degree-saturated maximum.  The protocol sets
    # an upper bound (32), not a target; leaving degree slack avoids rejection
    # sampling pathologies while retaining thousands of pairs per source.
    budget = min(int(max_pairs), len(indices) * int(max_pairs_per_song) // 4)
    neighbor_budget = int(round(budget * neighbor_fraction))
    selected: set[tuple[int, int]] = set()
    neighbor_pairs: set[tuple[int, int]] = set()
    degree = np.zeros(len(indices), dtype=np.int64)

    if neighbor_budget:
        similarities = local_pref @ local_pref.T
        np.fill_diagonal(similarities, -np.inf)
        nearest = np.argsort(-similarities, axis=1, kind="stable")
        anchors = rng.permutation(len(indices))
        depth = 0
        while len(neighbor_pairs) < neighbor_budget and depth < len(indices) - 1:
            for left in anchors:
                right = int(nearest[left, depth])
                pair = tuple(sorted((int(left), right)))
                if _add_pair(pair, selected, degree, max_pairs_per_song):
                    neighbor_pairs.add(pair)
                    if len(neighbor_pairs) >= neighbor_budget:
                        break
            depth += 1

    attempts = 0
    max_attempts = max(10_000, 100 * budget)
    while len(selected) < budget and attempts < max_attempts:
        left, right = rng.integers(0, len(indices), size=2)
        _add_pair((int(left), int(right)), selected, degree, max_pairs_per_song)
        attempts += 1

    pairs = sorted(selected)
    modes = np.asarray(
        ["neighbor" if pair in neighbor_pairs else "random" for pair in pairs], dtype=np.str_
    )
    oriented = []
    for left, right in pairs:
        oriented.append((right, left) if rng.random() < 0.5 else (left, right))
    global_pairs = np.asarray([(indices[l], indices[r]) for l, r in oriented], dtype=np.int64)
    return global_pairs, modes


def build_pair_batch(
    targets: np.ndarray,
    sources: np.ndarray,
    splits: np.ndarray,
    preference_embeddings: np.ndarray,
    thresholds: dict[str, dict[str, list[float]]],
    *,
    partition: str,
    seed: int,
    max_pairs: int = 100_000,
    max_pairs_per_song: int = 32,
    neighbor_fraction: float = 0.5,
) -> PairBatch:
    all_pairs, all_modes = [], []
    for source_index, source in enumerate(sorted(set(sources.astype(str)))):
        indices = np.flatnonzero((sources == source) & (splits == partition))
        if len(indices) < 2:
            continue
        pairs, modes = sample_unordered_pairs(
            indices,
            preference_embeddings,
            seed=seed + 1009 * source_index,
            max_pairs=max_pairs,
            max_pairs_per_song=max_pairs_per_song,
            neighbor_fraction=neighbor_fraction,
        )
        all_pairs.append(pairs)
        all_modes.append(modes)
    if not all_pairs:
        raise ValueError(f"no relation pairs for partition {partition!r}")
    pairs = np.concatenate(all_pairs)
    modes = np.concatenate(all_modes)
    pair_sources = sources[pairs[:, 0]].astype(str)
    if not np.array_equal(pair_sources, sources[pairs[:, 1]].astype(str)):
        raise AssertionError("cross-source pair generated")
    differences = targets[pairs[:, 1]] - targets[pairs[:, 0]]
    labels5, labels3 = label_pair_differences(differences, pair_sources, thresholds)
    return PairBatch(pairs[:, 0], pairs[:, 1], pair_sources, modes, labels5, labels3)
