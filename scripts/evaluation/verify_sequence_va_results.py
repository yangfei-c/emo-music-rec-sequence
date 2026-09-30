"""Independently audit the nine pilot VA runs; incomplete runs stay pending.

The interval score is recomputed by enumerating every corner for each VA axis,
without calling the production dynamic program, searcher, or label evaluator.
This verifies a reported playlist's cost, not global playlist optimality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import sys

# Keep this CPU audit small while the independent training queue is active.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))
from emomm_sequence.training.records import sha256, write_json

PROTOCOL = "frozen_anchors_on_eligible_bank_v1"
MODES = ("point", "raw_interval", "cqr_interval")
METHODS = ("a_grid", "a_refine", "b_set")
TOLERANCE = 2e-6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def archive(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def unit(array):
    array = np.asarray(array, dtype=np.float64)
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    assert np.isfinite(array).all() and np.all(norms > 1e-10)
    return array / norms


@lru_cache(maxsize=8)
def corners(length):
    assert 2 <= length <= 10, "This verifier is restricted to pilot lengths <= 10"
    return ((np.arange(2 ** length)[:, None] >> np.arange(length)) & 1).astype(float)


def scores(median, lower, upper, target, preference, has_preference):
    """All 2**N interval corners per axis; no production scoring imports."""
    length = len(median)
    wp, wr, ws = (.5, .25, .25) if has_preference else (2/3, 0., 1/3)
    step = (target[-1] - target[0]) / (length - 1)
    pref = wr * float(np.mean(preference))
    point = (wp * np.abs(median - target).mean() + pref
             + ws * np.abs(np.diff(median, axis=0) - step).mean() / 2)
    robust = pref
    for axis in range(2):
        values = lower[:, axis] + corners(length) * (upper[:, axis] - lower[:, axis])
        costs = wp * np.abs(values - target[:, axis]).sum(axis=1) / (2 * length)
        costs += ws * np.abs(np.diff(values, axis=1) - step[axis]).sum(axis=1) / (4 * (length - 1))
        robust += float(costs.max())
    return float(point), float(robust)


def inspect_training(root, mode, evaluation_manifest, record_hash):
    import torch
    from emomm_sequence.training.sequence_selector import load_model
    torch.set_num_threads(1)
    directory = root / "runs/pilot/set/seed_42" / f"va_{mode}"
    summary, manifest = read(directory / "summary.json"), read(directory / "manifest.json")
    selection, signatures = read(directory / "selection.json"), read(directory / "input_hashes.json")
    assert summary["status"] == "completed" and summary["parameters_changed"]
    assert summary["initial_state_sha256"] != summary["final_state_sha256"]
    assert summary["global_step"] > 0
    assert manifest["objective"] == mode and manifest["va_endpoint_protocol"] == PROTOCOL
    assert manifest["va_feature_dimensions"] == 7 and manifest["supervised_prediction_features"]
    assert manifest["variant"] == "set" and manifest["seed"] == 42
    assert manifest["limit_tasks"] == 0 and manifest["stage"] == "all"
    assert not any(manifest["model_overrides"].values())
    for key in ("planner", "route_b", "assets"):
        assert manifest["config"][key] == evaluation_manifest["config"][key]
    assert not selection["test_used"] and selection["input_hashes"] == signatures
    assert evaluation_manifest["checkpoint"] == selection
    assert selection["checkpoint_sha256"] == record_hash(directory / "best.pt")
    payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    assert payload["manifest"] == manifest and payload["input_hashes"] == signatures
    for key, expected in signatures.items():
        if key.startswith("va/"):
            path = root / "cache/va_extension" / key[3:]
        elif key.startswith("src/"):
            path = WORKSPACE / key
        elif key == "text_bank":
            path = root / "cache/text_bank.npz"
        else:
            path = root / "tasks/pilot" / key
        assert record_hash(path) == expected, (mode, key, "training input changed")
    model, loaded_selection = load_model(evaluation_manifest["config"], root, "pilot", "set", 42, "cpu", f"va_{mode}")
    assert loaded_selection == selection and model.project[0].in_features == 775
    assert all(torch.isfinite(tensor).all().item() for tensor in model.state_dict().values())
    history = rows(directory / "metrics.jsonl")
    selected = [row for row in history if row["phase"] == selection["phase"] and row["epoch"] == selection["epoch"]]
    assert len(selected) == 1 and selected[0]["selected"]
    assert abs(selected[0]["validation_cost"] - selection["metric"]) < 1e-12
    assert abs(summary["best_validation_cost"] - selection["metric"]) < 1e-12
    assert selection["metric"] <= min(row["validation_cost"] for row in history) + 1e-6
    for filename in ("manifest.json", "selection.json", "input_hashes.json", "summary.json", "metrics.jsonl", "invocations.jsonl"):
        record_hash(directory / filename)
    return {"updates": summary["global_step"], "elapsed_seconds": summary["elapsed_seconds"],
            "checkpoint_sha256": selection["checkpoint_sha256"], "feature_dimensions": 775,
            "selected_phase": selection["phase"], "selected_epoch": selection["epoch"],
            "strict_model_load_and_finite_parameters": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-incomplete", action="store_true", help="Audit completed runs and explicitly list pending ones")
    args = parser.parse_args()
    root = WORKSPACE / "artifacts/sequence_routes_v1"
    source_hashes = {}

    def record_hash(path):
        key = path.relative_to(WORKSPACE).as_posix()
        if key not in source_hashes:
            source_hashes[key] = sha256(path)
        return source_hashes[key]

    task_file = root / "tasks/pilot/evaluate_validation.jsonl"
    task_rows = [task for task in rows(task_file) if task["suite"] in ("e2", "e3")]
    tasks = {task["query_id"]: task for task in task_rows}
    assert len(tasks) == len(task_rows) == 624
    cache = root / "cache/va_extension"
    va_manifest = read(cache / "manifest.json")
    assert va_manifest["refit"] is False and va_manifest["playlist_coverage_guarantee"] is False
    text = unit(archive(root / "cache/text_bank.npz")["embeddings"])
    text_va = archive(cache / "text.npz")["predicted_va"].astype(np.float64)
    for path in (task_file, cache / "manifest.json", cache / "text.npz", root / "cache/text_bank.npz"):
        record_hash(path)
    banks, predictions, labels, lookups, label_lookups, eligibility = {}, {}, {}, {}, {}, {}
    for source in ("deam", "pmemo", "merge", "mtg"):
        bank_file = root / "banks" / f"{source}_validation.npz"
        pred_file = cache / f"{source}_validation.npz"
        label_file = root / "labels" / f"{source}_validation.npz"
        bank, pred, truth = archive(bank_file), archive(pred_file), archive(label_file)
        assert set(bank) == {"track_ids", "content_group_ids", "embeddings"}
        assert set(pred) == {"track_ids", "quantiles", "cqr_lower", "cqr_upper", "preference_distances", "preference_text_ids"}
        assert np.array_equal(bank["track_ids"], pred["track_ids"])
        for array in (pred["quantiles"], pred["cqr_lower"], pred["cqr_upper"], pred["preference_distances"]):
            assert np.isfinite(array).all() and np.all((array >= 0) & (array <= 1))
        assert np.all(np.diff(pred["quantiles"], axis=2) >= 0)
        assert np.all(pred["cqr_lower"] <= pred["quantiles"][:, :, 1])
        assert np.all(pred["quantiles"][:, :, 1] <= pred["cqr_upper"])
        bank["unit"] = unit(bank["embeddings"])
        banks[source], predictions[source], labels[source] = bank, pred, truth
        lookups[source] = {str(track): i for i, track in enumerate(bank["track_ids"])}
        label_lookups[source] = {str(track): i for i, track in enumerate(truth["track_ids"])}
        allowed = np.asarray(va_manifest["validation_eligibility"][source]["allowed_bank_indices"])
        assert len(set(allowed)) == len(allowed) and np.all((allowed >= 0) & (allowed < len(bank["track_ids"])))
        if source in ("deam", "pmemo"):
            ordered = sorted(set(map(str, bank["content_group_ids"])), key=lambda group: hashlib.sha256(f"20260906:{group}".encode()).hexdigest())
            cutoff = round(len(ordered) * .4) + round(len(ordered) * .3)
            expected_groups = set(ordered[cutoff:])
            expected = np.flatnonzero(np.isin(bank["content_group_ids"], list(expected_groups)))
            assert np.array_equal(allowed, expected), "Policy/calibration split changed"
        else:
            assert np.array_equal(allowed, np.arange(len(bank["track_ids"])))
        eligibility[source] = allowed
        for path in (bank_file, pred_file, label_file):
            record_hash(path)

    task_info = {}
    for query_id, task in tasks.items():
        source, pref_id = task["source"], task["preference_text_id"]
        allowed = eligibility[source]
        if task.get("indices") is not None:
            allowed = allowed[np.isin(allowed, task["indices"])]
        bank, pred = banks[source], predictions[source]
        x = bank["unit"][allowed]
        distances = np.clip((1 - x @ text[[task["start_text_id"], task["target_text_id"]]].T) / 2, 0, 1)
        if pref_id is not None:
            preference = np.clip((1 - x @ text[pref_id]) / 2, 0, 1)
            distances = .75 * distances + .25 * preference[:, None]
        # Two best choices per endpoint suffice to find the distinct-pair minimum.
        top = [np.argsort(distances[:, axis], kind="stable")[:2] for axis in range(2)]
        minimum = min(distances[i, 0] + distances[j, 1] for i in top[0] for j in top[1] if i != j)
        allowed_lookup = {str(bank["track_ids"][index]): i for i, index in enumerate(allowed)}
        task_info[query_id] = (allowed, allowed_lookup, distances, minimum)

    completed, pending, common_identity = [], [], {}
    baseline_config = None
    for mode in MODES:
        for method in METHODS:
            directory = root / "evaluations/pilot/va_retrained/validation" / mode / method / "seed_42/main"
            summary_file = directory / "summary.json"
            if not summary_file.exists() or read(summary_file).get("status") != "completed":
                pending.append({"mode": mode, "method": method})
                continue
            manifest, summary = read(directory / "manifest.json"), read(summary_file)
            assert read(directory / "progress.json")["status"] == "completed"
            assert manifest["task_sha256"] == record_hash(task_file)
            assert manifest["va_manifest_sha256"] == record_hash(cache / "manifest.json")
            assert manifest["mode"] == mode and manifest["method"] == method and manifest["seed"] == 42
            assert manifest["limit_tasks"] == 0 and manifest["va_endpoint_protocol"] == PROTOCOL
            assert manifest["validation_partition"] == "existing_policy_only_for_deam_pmemo"
            if baseline_config is None:
                baseline_config = manifest["config"]
            assert manifest["config"] == baseline_config
            for relative, expected in manifest["implementation_sha256"].items():
                assert record_hash(WORKSPACE / relative.replace("\\", "/")) == expected
            training = inspect_training(root, mode, manifest, record_hash) if method == "b_set" else None
            if method != "b_set":
                assert manifest["checkpoint"] is None
            outputs = rows(directory / "predictions.jsonl")
            assert len(outputs) == len(tasks) == summary["rows"]
            assert len({row["query_id"] for row in outputs}) == len(outputs)
            assert {row["query_id"] for row in outputs} == set(tasks)
            assert summary["failed"] == 0 and summary["selection_used_test"] is False
            maximum_errors = Counter()
            maximum_anchor_gap = 0.
            for row in outputs:
                task = tasks[row["query_id"]]
                source = task["source"]
                bank, pred, truth = banks[source], predictions[source], labels[source]
                allowed, allowed_lookup, distances, minimum = task_info[row["query_id"]]
                assert row["status"] == "ok" and row["mode"] == mode and row["method"] == method
                for field in ("query_family_id", "source", "suite", "split", "num_tracks"):
                    assert row[field] == task[field]
                assert row["seed"] == 42 and row["library_size"] == len(allowed)
                assert row["budget"] == min(task["budget"], len(allowed))
                assert row["va_endpoint_protocol"] == PROTOCOL and row["playlist_coverage_guarantee"] is False
                assert row["policy_retrained_on_va"] == (method == "b_set")
                identity = tuple(row[key] for key in ("start_id", "target_id", "library_size", "budget", "num_tracks"))
                assert common_identity.setdefault(row["query_id"], identity) == identity
                path, candidates = row["track_ids"], row["candidate_ids"]
                assert len(path) == len(set(path)) == row["num_tracks"]
                assert len(candidates) == len(set(candidates)) == row["budget"]
                assert set(path) <= set(candidates) <= set(allowed_lookup)
                assert path[0] == row["start_id"] and path[-1] == row["target_id"]
                index = np.asarray([lookups[source][track] for track in path])
                assert len(set(bank["content_group_ids"][index])) == len(path)
                anchor_gap = float(distances[allowed_lookup[path[0]], 0] + distances[allowed_lookup[path[-1]], 1] - minimum)
                maximum_anchor_gap = max(maximum_anchor_gap, abs(anchor_gap))
                assert abs(anchor_gap) <= TOLERANCE
                median = pred["quantiles"][index, :, 1].astype(np.float64)
                if mode == "point":
                    lower, upper = median, median
                elif mode == "raw_interval":
                    lower, upper = pred["quantiles"][index, :, 0].astype(np.float64), pred["quantiles"][index, :, 2].astype(np.float64)
                else:
                    lower, upper = pred["cqr_lower"][index].astype(np.float64), pred["cqr_upper"][index].astype(np.float64)
                pref_id = task["preference_text_id"]
                pref = np.zeros(len(path)) if pref_id is None else pred["preference_distances"][index, int(np.flatnonzero(pred["preference_text_ids"] == pref_id)[0])]
                target = np.linspace(text_va[task["start_text_id"]], text_va[task["target_text_id"]], len(path))
                point, robust = scores(median, lower, upper, target, pref, pref_id is not None)
                reconstructed = {"point_va_objective": point, "objective": robust}
                label_index = [label_lookups[source][track] for track in path]
                if "va" in truth:
                    y = truth["va"][label_index].astype(np.float64)
                    target_true = np.linspace(task["start_va"], task["target_va"], len(path))
                    step = (target_true[-1] - target_true[0]) / (len(path) - 1)
                    active = np.abs(step) > 1e-8
                    covered = ((y >= lower) & (y <= upper)).all(axis=1)
                    reconstructed.update(true_va_mae=float(np.abs(y - target_true).mean()),
                        true_endpoint_mae=float(np.abs(y[[0, -1]] - target_true[[0, -1]]).mean()),
                        true_step_deviation=float(np.abs(np.diff(y, axis=0) - step).mean()),
                        selected_song_joint_coverage=float(covered.mean()), selected_interval_width=float((upper - lower).mean()))
                    assert row["whole_playlist_covered"] == bool(covered.all())
                    assert row["calibration_source_known"] == (source in ("deam", "pmemo"))
                    if active.any():
                        reconstructed["true_reverse_fraction"] = float((np.diff(y, axis=0)[:, active] * np.sign(step[active]) < -1e-6).mean())
                    else:
                        assert row["true_reverse_fraction"] is None
                if "preference_targets" in truth and task.get("preference_label_index") is not None:
                    reconstructed["observed_preference_tag_match"] = float(truth["preference_targets"][label_index, task["preference_label_index"]].mean())
                for metric, expected in reconstructed.items():
                    difference = abs(expected - row[metric])
                    assert np.isfinite(row[metric]) and difference <= TOLERANCE, (mode, method, row["query_id"], metric, difference)
                    maximum_errors[metric] = max(maximum_errors[metric], difference)
            for filename in ("manifest.json", "summary.json", "predictions.jsonl"):
                record_hash(directory / filename)
            completed.append({"mode": mode, "method": method, "requests": len(outputs),
                              "suite_counts": dict(Counter(row["suite"] for row in outputs)),
                              "maximum_errors": dict(maximum_errors), "maximum_anchor_cost_gap": maximum_anchor_gap,
                              "training": training})
            print(f"verified {mode}/{method}: {len(outputs)} requests", flush=True)
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "scope": "pilot validation, seed 42; nine planned VA runs",
              "status": "complete" if not pending else "partial", "completed_runs": completed, "pending_runs": pending,
              "total_verified_outputs": sum(run["requests"] for run in completed), "tolerance": TOLERANCE,
              "policy_validation_counts": {source: len(allowed) for source, allowed in eligibility.items()},
              "score_reconstruction": "exhaustive interval corners separately on each axis; no production cost/evaluator calls",
              "anchor_verification": "identical IDs across completed runs; independent frozen distinct-pair cost within tolerance",
              "scope_limits": "Does not prove optimal playlists, model prediction accuracy, or theoretical post-selection coverage. B checkpoint verification runs only after training and evaluation complete.",
              "source_hashes": source_hashes, "verifier_sha256": sha256(__file__)}
    destination = root / "reports/pilot/va_results_verification.json"
    write_json(destination, report)
    print(json.dumps({"status": report["status"], "completed_runs": len(completed), "pending_runs": pending,
                      "verified_outputs": report["total_verified_outputs"]}, ensure_ascii=False))
    return 0 if not pending or args.allow_incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
