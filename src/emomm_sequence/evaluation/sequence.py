"""Evaluate Route 2 on held-out aggregate music-expression VA labels.

The policy partition alone chooses lambda and beam width. The historically
observed test outcomes are evaluated only after this choice is frozen. Query pools,
anchors and a VA-derived relevance *proxy* are shared by every method; this
mechanism experiment does not measure genuine text/preference satisfaction.

Example:
    python scripts/evaluation/evaluate_route2.py \
      --run artifacts/runs/route2_uncertainty_20260926/music_va_quantiles \
      --output artifacts/runs/route2_uncertainty_20260926/evaluation/route2_evaluation
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]

from emomm_sequence.inference.search import (
    point_path_cost,
    rank_relevance,
    robust_path_cost,
    select_robust_sequence,
)
from emomm_sequence.calibration.cqr import (
    VAUncertaintyModel,
    apply_joint_cqr,
    cqr_rank,
)

PARTITIONS = ("train", "validation_model", "calibration", "policy", "test")
DIRECTIONS = ((1, 1), (1, -1), (-1, 1), (-1, -1), (0, 0))
METHODS = ("robust", "point", "uncalibrated", "constant_width", "target_topn")


@dataclass(frozen=True)
class Data:
    embeddings: np.ndarray
    true_va: np.ndarray
    identities: np.ndarray
    groups: np.ndarray
    sources: np.ndarray
    partitions: np.ndarray


@dataclass(frozen=True)
class Prediction:
    indices: np.ndarray  # sorted global row indices for exactly one partition
    median: np.ndarray
    calibrated_lower: np.ndarray
    calibrated_upper: np.ndarray
    uncalibrated_lower: np.ndarray
    uncalibrated_upper: np.ndarray
    constant_lower: np.ndarray
    constant_upper: np.ndarray


@dataclass(frozen=True)
class Query:
    query_id: str
    source: str
    direction: tuple[int, int]
    pool: np.ndarray  # global row indices, fixed across methods/seeds/lengths
    start: int  # pool-local index
    target: int  # pool-local index
    start_relevance: np.ndarray  # VA-derived proxy, not measured text relevance
    target_relevance: np.ndarray


def _hash_groups(groups: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(groups.astype(str))).encode()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_data(run: Path) -> tuple[Data, dict]:
    """Enforce the training run's exact content-group manifest and audit."""
    config, audit = _json(run / "effective_config.json"), _json(run / "data_audit.json")
    blocks = []
    for source, relative in config["data"]["sources"].items():
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as loaded:
            block = {
                "embeddings": loaded["audio_embeddings"].astype(np.float32),
                "true_va": (loaded["va_targets"].astype(np.float64) + 1.0) / 2.0,
                "identities": loaded["global_identities"].astype(str),
                "groups": loaded["content_group_ids"].astype(str),
                "sources": np.repeat(source, len(loaded["global_identities"])),
            }
        if len(block["embeddings"]) != len(block["true_va"]):
            raise ValueError(f"{source}: feature/label row mismatch")
        blocks.append(block)
    joined = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    with np.load(run / "partition_manifest.npz", allow_pickle=False) as saved:
        manifest = {key: saved[key].astype(str) for key in (
            "identities", "content_group_ids", "sources", "partitions"
        )}
    for key, manifest_key in (("identities", "identities"), ("groups", "content_group_ids"), ("sources", "sources")):
        if not np.array_equal(joined[key], manifest[manifest_key]):
            raise ValueError(f"partition manifest disagrees with clean data: {key}")
    partitions = manifest["partitions"]
    if not set(partitions).issubset(PARTITIONS):
        raise ValueError("partition manifest has unknown partitions")
    group_partition: dict[str, str] = {}
    for group, partition in zip(joined["groups"], partitions, strict=True):
        prior = group_partition.setdefault(group, partition)
        if prior != partition:
            raise ValueError(f"content group crosses partitions: {group}")
    for source in sorted(set(joined["sources"])):
        for partition in PARTITIONS:
            selected = (joined["sources"] == source) & (partitions == partition)
            expected = audit["counts"][source][partition]
            if int(selected.sum()) != expected:
                raise ValueError(f"{source}/{partition}: partition count drift")
            if _hash_groups(joined["groups"][selected]) != audit["partition_group_sha256"][source][partition]:
                raise ValueError(f"{source}/{partition}: content-group hash drift")
    if not np.isfinite(joined["true_va"]).all() or np.any(joined["true_va"] < 0) or np.any(joined["true_va"] > 1):
        raise ValueError("true VA must be finite and in [0,1]")
    return Data(**joined, partitions=partitions), audit


def _constant_widths(quantiles: np.ndarray, data: Data, indices: np.ndarray, alpha: float) -> dict[str, float]:
    """Joint-axis split-conformal radius around the same model median."""
    widths = {}
    for source in sorted(set(data.sources[indices])):
        rows = data.sources[indices] == source
        residual = np.max(np.abs(data.true_va[indices[rows]] - quantiles[rows, :, 1]), axis=1)
        ordered = np.sort(residual)
        widths[source] = float(ordered[cqr_rank(len(ordered), alpha) - 1])
    return widths


def predict_partition(model: VAUncertaintyModel, data: Data, partition: str, widths: dict[str, float]) -> Prediction:
    indices = np.flatnonzero(data.partitions == partition)
    raw = model.predict_quantiles(data.embeddings[indices])
    median, lower, upper = apply_joint_cqr(raw, model.calibration, data.sources[indices])
    constant = np.asarray([widths[source] for source in data.sources[indices]], dtype=np.float64)[:, None]
    return Prediction(
        indices, median, lower, upper, raw[:, :, 0], raw[:, :, 2],
        np.maximum(0.0, median - constant), np.minimum(1.0, median + constant),
    )


def load_models_and_widths(run: Path, data: Data, device: str) -> tuple[dict[int, VAUncertaintyModel], dict[int, dict[str, float]], dict]:
    aggregate = _json(run / "aggregate.json")
    seeds = [int(seed) for seed in aggregate["seeds"]]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("run must contain unique model seeds")
    calibration_indices = np.flatnonzero(data.partitions == "calibration")
    models, widths = {}, {}
    for seed in seeds:
        seed_dir = run / f"seed_{seed}"
        model = VAUncertaintyModel.from_artifacts(seed_dir / "best.pt", seed_dir / "calibration.json", device=device)
        raw = model.predict_quantiles(data.embeddings[calibration_indices])
        widths[seed] = _constant_widths(raw, data, calibration_indices, float(model.calibration["alpha"]))
        for source, details in model.calibration["source_corrections"].items():
            expected = int(np.sum(data.sources[calibration_indices] == source))
            if details["n_songs"] != expected:
                raise ValueError(f"seed {seed}: calibration song count mismatch for {source}")
        models[seed] = model
    return models, widths, aggregate


def _stable_rng_key(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _choose_anchors(median: np.ndarray, direction: tuple[int, int]) -> tuple[int, int] | None:
    """Use predicted VA only; labels are hidden during query construction."""
    delta = median[None, :, :] - median[:, None, :]
    eligible = np.ones(delta.shape[:2], dtype=bool)
    np.fill_diagonal(eligible, False)
    score = np.zeros(delta.shape[:2], dtype=np.float64)
    for axis, sign in enumerate(direction):
        if sign:
            eligible &= sign * delta[:, :, axis] > 0.0
            score += sign * delta[:, :, axis]
        else:
            score -= np.abs(delta[:, :, axis])
    if not eligible.any():
        return None
    score[~eligible] = -np.inf
    left, right = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(left), int(right)


def build_queries(data: Data, reference: Prediction, *, partition: str, pool_size: int, repeats: int) -> list[Query]:
    if repeats < 1 or pool_size < 2:
        raise ValueError("repeats >= 1 and pool_size >= 2 are required")
    queries = []
    for source in sorted(set(data.sources[reference.indices])):
        eligible = reference.indices[data.sources[reference.indices] == source]
        if len(eligible) < pool_size:
            raise ValueError(f"{source}/{partition} has {len(eligible)} songs, below candidate pool {pool_size}")
        for direction in DIRECTIONS:
            for repeat in range(repeats):
                rng = np.random.default_rng(_stable_rng_key("route2", partition, source, direction, repeat))
                pool = np.sort(rng.choice(eligible, size=pool_size, replace=False))
                local = np.searchsorted(reference.indices, pool)
                predicted = reference.median[local]
                anchors = _choose_anchors(predicted, direction)
                if anchors is None:
                    raise ValueError(f"no valid predicted anchor pair: {source}/{partition}/{direction}/{repeat}")
                start, target = anchors
                ids = data.identities[pool]
                start_relevance = rank_relevance(-np.linalg.norm(predicted - predicted[start], axis=1), ids)
                target_relevance = rank_relevance(-np.linalg.norm(predicted - predicted[target], axis=1), ids)
                queries.append(Query(
                    f"{partition}:{source}:{direction[0]},{direction[1]}:{repeat}", source,
                    direction, pool, start, target, start_relevance, target_relevance,
                ))
    return queries


def _true_path_metrics(truth: np.ndarray, median: np.ndarray, direction: tuple[int, int]) -> dict[str, float]:
    n = len(truth)
    delta = np.diff(truth, axis=0)
    predicted_h = np.zeros(2, dtype=np.float64)
    active = np.asarray([value is not None for value in direction], dtype=bool)
    directed = np.asarray([value != 0 and value is not None for value in direction], dtype=bool)
    for axis, sign in enumerate(direction):
        if sign:
            predicted_h[axis] = (median[-1, axis] - median[0, axis]) / (n - 1)
    deviation = np.abs(delta - predicted_h)[..., active]
    true_h = np.zeros(2, dtype=np.float64)
    true_h[active] = (truth[-1, active] - truth[0, active]) / (n - 1)
    backward = np.maximum(0.0, -delta[:, directed] * np.asarray(direction)[directed])
    return {
        "true_reference_deviation_mean": float(deviation.mean()),
        "true_reference_deviation_total": float(deviation.sum()),
        "true_reference_deviation_worst_step": float(deviation.sum(axis=1).max() / active.sum()),
        "true_endpoint_step_unevenness": float(np.abs(delta[:, active] - true_h[active]).mean()),
        "true_backward_magnitude_mean": float(backward.mean()) if backward.size else None,
        "true_backward_edge_fraction": float((backward > 0).mean()) if backward.size else None,
        "true_net_progress_mean": float((np.asarray(direction)[directed] * (truth[-1, directed] - truth[0, directed])).mean()) if directed.any() else None,
        "true_target_error": float(np.abs(truth[-1, active] - median[-1, active]).mean()),
        "true_maintain_total_change": float(np.abs(delta[:, ~directed]).sum()) if (~directed).any() else 0.0,
    }


def _proxy_cost(query: Query, path: Sequence[int]) -> float:
    n = len(path)
    return float(np.mean([
        1.0 - ((1.0 - k / (n - 1)) * query.start_relevance[index]
               + k / (n - 1) * query.target_relevance[index])
        for k, index in enumerate(path)
    ]))


def evaluate_query(
    data: Data, prediction: Prediction, query: Query, *, seed: int, n: int,
    method: str, lam: float, beam: int, partition: str,
) -> dict:
    """Return one paired result; all variants share pool, anchors and proxy."""
    if method not in METHODS or n < 2:
        raise ValueError("unknown method or N < 2")
    local = np.searchsorted(prediction.indices, query.pool)
    if not np.array_equal(prediction.indices[local], query.pool):
        raise ValueError("query pool is outside prediction partition")
    m = prediction.median[local]
    options = {
        "robust": (prediction.calibrated_lower[local], prediction.calibrated_upper[local], "robust"),
        "point": (m, m, "point"),
        "uncalibrated": (prediction.uncalibrated_lower[local], prediction.uncalibrated_upper[local], "robust"),
        "constant_width": (prediction.constant_lower[local], prediction.constant_upper[local], "robust"),
        "target_topn": (prediction.calibrated_lower[local], prediction.calibrated_upper[local], "robust"),
    }
    lower, upper, mode = options[method]
    base = {
        "partition": partition, "source": query.source, "seed": seed,
        "query_id": query.query_id, "direction_v": query.direction[0],
        "direction_a": query.direction[1],
        "task_type": "maintain" if query.direction == (0, 0) else "directional",
        "n": n, "method": method,
        "lambda_retrieval": lam, "beam_width": beam,
        "candidate_pool_size": len(query.pool),
        "start_id": str(data.identities[query.pool[query.start]]),
        "target_id": str(data.identities[query.pool[query.target]]),
    }
    began = time.perf_counter()
    if n > len(query.pool):
        return {**base, "status": "insufficient_candidates", "failure_reason": "candidate_pool_lt_requested_n",
                "elapsed_seconds": time.perf_counter() - began}
    anchored_delta = m[query.target] - m[query.start]
    if any(sign * anchored_delta[axis] < -1.0e-12 for axis, sign in enumerate(query.direction) if sign):
        return {**base, "status": "anchor_direction_conflict", "failure_reason": "anchors_oppose_requested_direction",
                "elapsed_seconds": time.perf_counter() - began}
    if method == "target_topn":
        middle = sorted(
            (i for i in range(len(query.pool)) if i not in (query.start, query.target)),
            key=lambda i: (-query.target_relevance[i], str(data.identities[query.pool[i]])),
        )[:n - 2]
        path = (query.start, *middle, query.target)
        step = np.asarray([anchored_delta[axis] / (n - 1) if sign else 0.0
                           for axis, sign in enumerate(query.direction)])
        active = np.ones(2, dtype=bool)
        robust_cost = robust_path_cost(lower[list(path)], upper[list(path)], step, active)
        point_cost = point_path_cost(m[list(path)], step, active)
        proxy = _proxy_cost(query, path)
    else:
        result = select_robust_sequence(
            m, lower, upper, query.start_relevance, query.target_relevance,
            num_tracks=n, start_index=query.start, target_index=query.target,
            direction=query.direction, lambda_retrieval=lam, beam_width=beam,
            track_ids=data.identities[query.pool], mode=mode,
        )
        if result.status != "sequence_generated":
            return {**base, "status": result.status, "failure_reason": result.failure_reason,
                    "elapsed_seconds": time.perf_counter() - began}
        path = result.indices
        robust_cost = result.robust_path_cost
        point_cost = result.point_path_cost
        proxy = result.retrieval_match_cost
    global_path = query.pool[list(path)]
    truth = data.true_va[global_path]
    predicted_path = m[list(path)]
    metrics = _true_path_metrics(truth, predicted_path, query.direction)
    calibrated_covered = (
        (truth >= prediction.calibrated_lower[local[list(path)]])
        & (truth <= prediction.calibrated_upper[local[list(path)]])
    ).all(axis=1)
    return {
        **base, "status": "sequence_generated", "failure_reason": "",
        "elapsed_seconds": time.perf_counter() - began,
        "song_ids_json": json.dumps(data.identities[global_path].tolist(), ensure_ascii=False),
        "robust_path_cost": robust_cost, "point_path_cost": point_cost,
        "proxy_relevance_cost": proxy,
        "selected_song_joint_coverage_calibrated": float(calibrated_covered.mean()),
        "entire_playlist_joint_coverage_calibrated": int(calibrated_covered.all()),
        **metrics,
    }


def _mean(rows: list[dict], field: str) -> float | None:
    valid = [float(row[field]) for row in rows if field in row and row[field] not in ("", None)]
    return float(np.mean(valid)) if valid else None


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_rows(data: Data, prediction: Prediction, rows: list[dict], *, seed: int, partition: str) -> list[dict]:
    output = []
    for source in sorted(set(data.sources[prediction.indices])):
        source_mask = data.sources[prediction.indices] == source
        global_source = prediction.indices[source_mask]
        truth = data.true_va[global_source]
        lower = prediction.calibrated_lower[source_mask]
        upper = prediction.calibrated_upper[source_mask]
        covered_axis = (truth >= lower) & (truth <= upper)
        covered = covered_axis.all(axis=1)
        lookup = {int(index): int(position) for position, index in enumerate(global_source)}
        for method in METHODS:
            for n in sorted({int(row["n"]) for row in rows}):
                selected_rows = [row for row in rows if row["seed"] == seed and row["source"] == source
                                 and row["method"] == method and row["n"] == n and row["status"] == "sequence_generated"]
                selected_ids = [song for row in selected_rows for song in json.loads(row["song_ids_json"])]
                identity_to_index = {str(data.identities[index]): index for index in global_source}
                appearances = np.asarray([lookup[identity_to_index[song]] for song in selected_ids], dtype=int)
                unique = np.unique(appearances)
                rng = np.random.default_rng(_stable_rng_key("matched_random_coverage", partition, seed, source, method, n))
                random_sample = rng.choice(len(covered), size=len(unique), replace=False) if len(unique) else np.asarray([], dtype=int)
                output.append({
                    "partition": partition, "source": source, "seed": seed, "method": method, "n": n,
                    "all_songs_n": len(covered), "all_song_v_coverage": float(covered_axis[:, 0].mean()),
                    "all_song_a_coverage": float(covered_axis[:, 1].mean()),
                    "all_song_joint_coverage": float(covered.mean()),
                    "all_song_mean_width_v": float((upper[:, 0] - lower[:, 0]).mean()),
                    "all_song_mean_width_a": float((upper[:, 1] - lower[:, 1]).mean()),
                    "selected_unique_n": len(unique), "selected_occurrences_n": len(appearances),
                    "selected_unique_joint_coverage": float(covered[unique].mean()) if len(unique) else None,
                    "selected_occurrence_joint_coverage": float(covered[appearances].mean()) if len(appearances) else None,
                    "matched_random_unique_joint_coverage": float(covered[random_sample].mean()) if len(random_sample) else None,
                    "whole_playlist_joint_coverage": _mean(selected_rows, "entire_playlist_joint_coverage_calibrated"),
                    "coverage_scope": "empirical selection-conditioned diagnostic; no conformal guarantee",
                })
    return output


def _interval_rows(data: Data, prediction: Prediction, *, seed: int, partition: str) -> list[dict]:
    result = []
    for source in sorted(set(data.sources[prediction.indices])):
        selected = data.sources[prediction.indices] == source
        truth = data.true_va[prediction.indices[selected]]
        for name, lower, upper in (
            ("calibrated", prediction.calibrated_lower[selected], prediction.calibrated_upper[selected]),
            ("uncalibrated", prediction.uncalibrated_lower[selected], prediction.uncalibrated_upper[selected]),
            ("constant_width", prediction.constant_lower[selected], prediction.constant_upper[selected]),
        ):
            covered = (truth >= lower) & (truth <= upper)
            result.append({
                "partition": partition, "seed": seed, "source": source, "interval": name,
                "n_songs": len(truth), "v_coverage": float(covered[:, 0].mean()),
                "a_coverage": float(covered[:, 1].mean()),
                "joint_coverage": float(covered.all(axis=1).mean()),
                "mean_width_v": float((upper[:, 0] - lower[:, 0]).mean()),
                "mean_width_a": float((upper[:, 1] - lower[:, 1]).mean()),
                "median_width_v": float(np.median(upper[:, 0] - lower[:, 0])),
                "median_width_a": float(np.median(upper[:, 1] - lower[:, 1])),
            })
    return result


def _summary_rows(rows: list[dict]) -> list[dict]:
    result = []
    keys = sorted({(row["source"], row["seed"], row["method"], row["n"], row["task_type"]) for row in rows})
    for source, seed, method, n, task_type in keys:
        selected = [row for row in rows if (row["source"], row["seed"], row["method"], row["n"], row["task_type"]) == (source, seed, method, n, task_type)]
        success = [row for row in selected if row["status"] == "sequence_generated"]
        result.append({
            "source": source, "seed": seed, "method": method, "n": n, "task_type": task_type,
            "requests": len(selected), "generated": len(success),
            "generation_rate": len(success) / len(selected),
            "status_counts_json": json.dumps({status: sum(row["status"] == status for row in selected)
                                         for status in sorted({row["status"] for row in selected})}),
            **{f"mean_{key}": _mean(success, key) for key in (
                "true_reference_deviation_mean", "true_reference_deviation_total",
                "true_reference_deviation_worst_step", "true_endpoint_step_unevenness",
                "true_backward_magnitude_mean", "true_backward_edge_fraction",
                "true_net_progress_mean", "true_target_error", "true_maintain_total_change",
                "proxy_relevance_cost", "elapsed_seconds", "entire_playlist_joint_coverage_calibrated",
            )},
        })
    return result


def _paired_rows(rows: list[dict]) -> list[dict]:
    """Per-query paired differences; overlapping playlists are not i.i.d. units."""
    by_key = {(row["source"], row["seed"], row["query_id"], row["n"], row["method"]): row for row in rows}
    paired = []
    fields = ("true_reference_deviation_mean", "true_backward_magnitude_mean",
              "true_backward_edge_fraction", "true_maintain_total_change",
              "proxy_relevance_cost")
    for row in rows:
        if row["method"] == "robust":
            continue
        ours = by_key[(row["source"], row["seed"], row["query_id"], row["n"], "robust")]
        both_generated = row["status"] == ours["status"] == "sequence_generated"
        paired.append({
            "source": row["source"], "seed": row["seed"], "query_id": row["query_id"],
            "n": row["n"], "task_type": row["task_type"],
            "baseline": row["method"], "both_generated": int(both_generated),
            **{f"robust_minus_baseline_{field}": float(ours[field] - row[field])
               if both_generated and ours[field] is not None and row[field] is not None else None
               for field in fields},
        })
    return paired


def _cluster_bootstrap_rows(paired: list[dict], repeats: int) -> list[dict]:
    """Exploratory query-cluster CIs, stratified by source.

    All seeds of one query remain in the same cluster. Different query pools
    can share songs, so these intervals do not establish song-independent
    inferential coverage.
    """
    if repeats < 100:
        raise ValueError("bootstrap-replicates must be at least 100")
    output = []
    for n in sorted({row["n"] for row in paired}):
        for task_type in ("directional", "maintain"):
            for seed_scope in ("all", *sorted({row["seed"] for row in paired})):
                selected = [row for row in paired if row["baseline"] == "point"
                            and row["n"] == n and row["task_type"] == task_type
                            and (seed_scope == "all" or row["seed"] == seed_scope)
                            and row["both_generated"]]
                for field in ("true_reference_deviation_mean", "true_backward_magnitude_mean",
                              "true_backward_edge_fraction", "true_maintain_total_change",
                              "proxy_relevance_cost"):
                    key = f"robust_minus_baseline_{field}"
                    clusters: dict[tuple[str, str], list[float]] = {}
                    for row in selected:
                        if row[key] is not None:
                            clusters.setdefault((row["source"], row["query_id"]), []).append(float(row[key]))
                    by_source: dict[str, np.ndarray] = {}
                    for source in sorted({name for name, _ in clusters}):
                        by_source[source] = np.asarray([
                            np.mean(values) for (name, _), values in sorted(clusters.items()) if name == source
                        ], dtype=np.float64)
                    count = sum(len(values) for values in by_source.values())
                    if count == 0:
                        continue
                    mean = float(sum(values.sum() for values in by_source.values()) / count)
                    rng = np.random.default_rng(_stable_rng_key("route2_query_bootstrap", n, task_type, seed_scope, field))
                    resampled_sum = np.zeros(repeats, dtype=np.float64)
                    for values in by_source.values():
                        draws = rng.integers(0, len(values), size=(repeats, len(values)))
                        resampled_sum += values[draws].sum(axis=1)
                    low, high = np.quantile(resampled_sum / count, [0.025, 0.975])
                    output.append({
                        "baseline": "point", "n": n, "task_type": task_type,
                        "seed_scope": seed_scope, "metric": field, "n_query_clusters": count,
                        "n_paired_rows": sum(len(values) for values in clusters.values()),
                        "mean_robust_minus_baseline": mean,
                        "cluster_bootstrap_95_low": float(low),
                        "cluster_bootstrap_95_high": float(high),
                        "bootstrap_replicates": repeats,
                        "scope": "exploratory query-cluster CI; query pools can share songs",
                    })
    return output


def _complete_seed_rows(paired: list[dict], seeds: Sequence[int]) -> list[dict]:
    """Keep query clusters that generated both methods in every model seed."""
    seed_set = set(seeds)
    complete: dict[tuple[int, str, str], set[int]] = {}
    for row in paired:
        if row["baseline"] == "point" and row["both_generated"]:
            complete.setdefault((row["n"], row["task_type"], row["query_id"]), set()).add(row["seed"])
    eligible = {key for key, available in complete.items() if available == seed_set}
    return [row for row in paired if (row["n"], row["task_type"], row["query_id"]) in eligible]


def _query_rows(data: Data, queries: list[Query]) -> list[dict]:
    return [{
        "query_id": query.query_id, "source": query.source,
        "direction_v": query.direction[0], "direction_a": query.direction[1],
        "candidate_ids_json": json.dumps(data.identities[query.pool].tolist(), ensure_ascii=False),
        "start_id": str(data.identities[query.pool[query.start]]),
        "target_id": str(data.identities[query.pool[query.target]]),
        "start_proxy_relevance_json": json.dumps(query.start_relevance.tolist()),
        "target_proxy_relevance_json": json.dumps(query.target_relevance.tolist()),
    } for query in queries]


def _policy_choice(rows: list[dict], lambdas: Sequence[float], beams: Sequence[int]) -> dict:
    def paired_key(row: dict) -> tuple:
        return row["seed"], row["query_id"], row["n"]

    common_keys = set.intersection(*(
        {paired_key(row) for row in rows if row["lambda_retrieval"] == lam
         and row["beam_width"] == beam and row["status"] == "sequence_generated"}
        for lam in lambdas for beam in beams
    ))
    if not common_keys:
        raise ValueError("no common successful policy queries across the parameter grid")
    options = []
    for lam in lambdas:
        for beam in beams:
            relevant = [row for row in rows if row["lambda_retrieval"] == lam and row["beam_width"] == beam]
            success = [row for row in relevant if paired_key(row) in common_keys]
            options.append({
                "lambda_retrieval": lam, "beam_width": beam, "requests": len(relevant),
                "generated": sum(row["status"] == "sequence_generated" for row in relevant),
                "paired_selection_queries": len(success),
                "true_reference_deviation_mean": _mean(success, "true_reference_deviation_mean"),
                "proxy_relevance_cost": _mean(success, "proxy_relevance_cost"),
                "elapsed_seconds_mean": _mean(success, "elapsed_seconds"),
            })
    def regret(field: str, value: float) -> float:
        values = np.asarray([option[field] for option in options])
        span = float(values.max() - values.min())
        return 0.0 if span == 0.0 else float((value - values.min()) / span)
    for option in options:
        option["balanced_max_regret"] = max(
            regret("true_reference_deviation_mean", option["true_reference_deviation_mean"]),
            regret("proxy_relevance_cost", option["proxy_relevance_cost"]),
        )
    chosen = min(options, key=lambda option: (
        option["balanced_max_regret"], option["beam_width"],
        abs(option["lambda_retrieval"] - 1.0),
    ))
    return {
        "selection_partition": "policy",
        "criterion": "minimum maximum normalized regret across true reference deviation and VA-derived relevance proxy; ties by smaller beam then lambda nearest 1",
        "selected_lambda_retrieval": chosen["lambda_retrieval"],
        "selected_beam_width": chosen["beam_width"],
        "common_successful_policy_requests_per_grid_setting": len(common_keys),
        "grid": options,
    }


def run(args: argparse.Namespace) -> dict:
    run_dir, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output}")
    if any(n < 2 for n in args.lengths) or len(set(args.lengths)) != len(args.lengths):
        raise ValueError("lengths must be distinct integers >= 2")
    if args.candidate_pool_size < max(args.lengths):
        raise ValueError("candidate pool must be at least the largest requested N")
    if not args.lambda_grid or any(not np.isfinite(v) or v < 0 for v in args.lambda_grid):
        raise ValueError("lambda grid must contain finite nonnegative values")
    if not args.beam_grid or any(value < 1 for value in args.beam_grid):
        raise ValueError("beam grid must contain positive widths")

    data, audit = load_data(run_dir)
    models, widths, aggregate = load_models_and_widths(run_dir, data, args.device)
    selected_seed = int(aggregate["selected_seed_by_validation_only"])
    if selected_seed not in models:
        raise ValueError("validation-selected seed not found")
    policy_predictions = {seed: predict_partition(model, data, "policy", widths[seed]) for seed, model in models.items()}
    policy_queries = build_queries(
        data, policy_predictions[selected_seed], partition="policy",
        pool_size=args.candidate_pool_size, repeats=args.queries_per_direction,
    )
    policy_rows = [
        evaluate_query(data, prediction, query, seed=seed, n=n, method="robust", lam=lam, beam=beam, partition="policy")
        for lam in args.lambda_grid for beam in args.beam_grid
        for seed, prediction in policy_predictions.items()
        for query in policy_queries for n in args.lengths
    ]
    policy = _policy_choice(policy_rows, args.lambda_grid, args.beam_grid)
    lam, beam = policy["selected_lambda_retrieval"], policy["selected_beam_width"]
    point_policy_rows: list[dict] = []
    point_policy = None
    if args.independent_point_policy:
        point_policy_rows = [
            evaluate_query(data, prediction, query, seed=seed, n=n, method="point", lam=point_lam,
                           beam=point_beam, partition="policy")
            for point_lam in args.lambda_grid for point_beam in args.beam_grid
            for seed, prediction in policy_predictions.items()
            for query in policy_queries for n in args.lengths
        ]
        point_policy = _policy_choice(point_policy_rows, args.lambda_grid, args.beam_grid)

    def test_parameters(method: str) -> tuple[float, int]:
        if method == "point" and point_policy is not None:
            return point_policy["selected_lambda_retrieval"], point_policy["selected_beam_width"]
        return lam, beam

    # Historical test outcomes are evaluated only below, after policy freeze.
    test_predictions = {seed: predict_partition(model, data, "test", widths[seed]) for seed, model in models.items()}
    test_queries = build_queries(
        data, test_predictions[selected_seed], partition="test",
        pool_size=args.candidate_pool_size, repeats=args.queries_per_direction,
    )
    test_rows = [
        evaluate_query(data, prediction, query, seed=seed, n=n, method=method,
                       lam=test_parameters(method)[0], beam=test_parameters(method)[1], partition="test")
        for seed, prediction in test_predictions.items()
        for query in test_queries for n in args.lengths for method in METHODS
    ]
    paired = _paired_rows(test_rows)
    complete_seed_paired = _complete_seed_rows(paired, list(models))
    coverage = [row for seed, prediction in test_predictions.items()
                for row in _coverage_rows(data, prediction, test_rows, seed=seed, partition="test")]
    intervals = [row for seed, prediction in test_predictions.items()
                 for row in _interval_rows(data, prediction, seed=seed, partition="test")]
    output.mkdir(parents=True)
    for name, rows in (
        ("policy_queries.csv", _query_rows(data, policy_queries)),
        ("test_queries.csv", _query_rows(data, test_queries)),
        ("policy_grid.csv", policy_rows), ("test_per_query.csv", test_rows),
        ("policy_grid_point.csv", point_policy_rows),
        ("test_summary.csv", _summary_rows(test_rows)),
        ("test_paired_vs_robust.csv", paired),
        ("test_paired_cluster_ci.csv", _cluster_bootstrap_rows(paired, args.bootstrap_replicates)),
        ("test_paired_complete_seed_cluster_ci.csv", [
            row for row in _cluster_bootstrap_rows(complete_seed_paired, args.bootstrap_replicates)
            if row["seed_scope"] == "all"
        ]),
        ("test_coverage.csv", coverage), ("test_intervals.csv", intervals),
    ):
        _write_csv(output / name, rows)
    report = {
        "run": str(run_dir), "output": str(output), "seeds": list(models),
        "seed_selected_by_validation_only": selected_seed,
        "policy": policy, "point_policy": point_policy,
        "test_comparison": "independently_policy_tuned_point_vs_robust" if point_policy is not None else "shared_policy_parameters",
        "lengths": args.lengths,
        "methods": METHODS, "candidate_pool_size": args.candidate_pool_size,
        "queries_per_direction_per_source": args.queries_per_direction,
        "directions": DIRECTIONS,
        "policy_query_count": len(policy_queries), "test_query_count": len(test_queries),
        "test_history": "previously_observed_re_evaluation_not_new_blind_test",
        "outcome_scope": "aggregate annotated music-expression VA; no listener emotion or transition preference claim",
        "relevance_scope": "VA-derived start/target similarity rank proxy; not genuine text or preference evaluation",
        "fixed_comparison": "same candidate pool, start/target anchors, N and proxy scores within each query and seed",
        "coverage_scope": "single-song all-test empirical coverage distinct from selected-song and entire-playlist empirical coverage",
        "inference_scope": "paired query-cluster bootstrap is exploratory because separately sampled query pools can share songs",
        "complete_seed_sensitivity": "additional query-cluster CI uses only queries successful for point and robust in every model seed",
        "data_audit_counts": audit["counts"],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/runs/route2_uncertainty_20260926/music_va_quantiles")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 5, 7, 10])
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--queries-per-direction", type=int, default=2)
    parser.add_argument("--lambda-grid", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--beam-grid", type=int, nargs="+", default=[16, 50])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--independent-point-policy", action="store_true",
                        help="Choose point's lambda/beam independently on policy using the same grid and criterion")
    report = run(parser.parse_args())
    print(json.dumps({
        "output": report["output"], "policy": report["policy"],
        "point_policy": report["point_policy"],
        "test_query_count": report["test_query_count"], "test_history": report["test_history"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
