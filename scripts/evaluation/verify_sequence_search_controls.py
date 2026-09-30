"""Audit completed pilot search controls against frozen tasks and raw embeddings."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import sha256, write_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def unit(value):
    value = np.asarray(value, dtype=np.float64)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    if not np.isfinite(value).all() or np.any(norm < 1e-10):
        raise ValueError("Invalid embedding in audit input")
    return value / norm


def path_scores(vectors, task, text):
    """Direct float64 dot products; no evaluator or searcher cost function."""
    start, target = unit(text[task["start_text_id"]]), unit(text[task["target_text_id"]])
    n = len(vectors)
    fractions = np.linspace(0, 1, n) if n > 1 else np.array([0.5])
    direction = (1 - fractions[:, None]) * start + fractions[:, None] * target
    for index in np.flatnonzero(np.linalg.norm(direction, axis=1) < 1e-8):
        direction[index] = start if fractions[index] <= 0.5 else target
    emotion = float(np.clip((1 - np.sum(unit(direction) * vectors, axis=1)) / 2, 0, 1).mean())
    pref_id = task["preference_text_id"]
    preference = 0.0 if pref_id is None else float(np.clip((1 - vectors @ unit(text[pref_id])) / 2, 0, 1).mean())
    smooth = float(np.clip((1 - np.sum(vectors[:-1] * vectors[1:], axis=1)) / 2, 0, 1).mean()) if n > 1 else 0.0
    common = (2 * emotion + smooth) / 3 if pref_id is None else .5 * emotion + .25 * preference + .25 * smooth
    without_smooth = emotion if pref_id is None else (2 * emotion + preference) / 3
    return {"emotion_distance": emotion, "preference_distance": preference, "smooth_distance": smooth,
            "common_objective": common, "without_smooth_objective": without_smooth}


def main():
    root = ROOT / "artifacts/sequence_routes_v1"
    task_file = root / "tasks/pilot/evaluate_validation.jsonl"
    task_rows = read_rows(task_file)
    tasks = {row["query_id"]: row for row in task_rows}
    assert len(tasks) == len(task_rows) == 732
    text_file = root / "cache/text_bank.npz"
    with np.load(text_file, allow_pickle=False) as archive:
        text = archive["embeddings"]
    banks, bank_hashes = {}, {}
    for source in ("deam", "pmemo", "merge", "mtg"):
        path = root / "banks" / f"{source}_validation.npz"
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["track_ids"].astype(str)
            banks[source] = (ids, archive["content_group_ids"].astype(str), unit(archive["embeddings"]),
                             {track: i for i, track in enumerate(ids)})
        bank_hashes[source] = sha256(path)
    definitions = {"changed_planner_exact": ("a_grid", "b_set"),
                   "a_no_smooth": ("a_grid", "a_refine"),
                   "beam_200": ("a_grid", "a_refine", "b_set")}
    identity_fields = ("query_family_id", "source", "suite", "split", "method", "seed", "num_tracks",
                       "requested_budget", "budget", "library_size", "start_id", "target_id", "phrase_index")
    tolerance = 2e-6
    runs, source_files = [], {}
    for run_id, methods in definitions.items():
        for method in methods:
            parent = root / "evaluations/pilot/validation" / method / "seed_42"
            directory = parent / run_id
            summary, progress = read(directory / "summary.json"), read(directory / "progress.json")
            assert summary["status"] == progress["status"] == "completed"
            manifest = read(directory / "manifest.json")
            original_manifest = read(parent / "main/manifest.json")
            expected_manifest = dict(original_manifest)
            if run_id == "changed_planner_exact":
                expected_manifest.update(suite_filter=["e1"], planner="exact")
            elif run_id == "a_no_smooth":
                expected_manifest["no_smooth"] = True
            else:
                expected_manifest["beam_width"] = 200
            assert manifest == expected_manifest, (run_id, method, "unexpected manifest change")
            assert manifest["task_sha256"] == sha256(task_file)
            assert manifest["text_sha256"] == sha256(text_file)
            if method == "b_set":
                assert manifest["checkpoint"]["checkpoint_sha256"] == sha256(root / "runs/pilot/set/seed_42/main/best.pt")
            predictions = read_rows(directory / "predictions.jsonl")
            main_rows = read_rows(parent / "main/predictions.jsonl")
            baseline = {row["query_id"]: row for row in main_rows}
            expected_ids = {key for key, task in tasks.items() if run_id != "changed_planner_exact" or task["suite"] == "e1"}
            assert len(baseline) == len(main_rows) == 732
            assert len(predictions) == len(expected_ids) == summary["requests"] == summary["successful"]
            assert {row["query_id"] for row in predictions} == expected_ids
            max_error, same_candidates, same_paths = 0.0, 0, 0
            differences, changed_by_suite, no_smooth_differences = [], Counter(), []
            for row in predictions:
                original, task = baseline[row["query_id"]], tasks[row["query_id"]]
                assert all(row[field] == original[field] for field in identity_fields)
                assert row["status"] == original["status"] == "ok"
                ids, groups, vectors, lookup = banks[row["source"]]
                allowed = set(ids if task.get("indices") is None else ids[np.asarray(task["indices"], dtype=int)])
                assert len(allowed) == row["library_size"]
                for record in (row, original):
                    path, candidates = record["track_ids"], record["candidate_ids"]
                    assert len(path) == record["num_tracks"] == task["num_tracks"]
                    assert path[0] == record["start_id"] and path[-1] == record["target_id"]
                    assert set(path) <= set(candidates) <= allowed
                    assert len({groups[lookup[track]] for track in path}) == len(path)
                    assert len(candidates) == record["candidate_count"] == len(set(candidates))
                    assert len(candidates) <= record["budget"]
                scores = path_scores(vectors[[lookup[track] for track in row["track_ids"]]], task, text)
                original_scores = path_scores(vectors[[lookup[track] for track in original["track_ids"]]], task, text)
                for field in ("emotion_distance", "preference_distance", "smooth_distance"):
                    max_error = max(max_error, abs(scores[field] - row[field]), abs(original_scores[field] - original[field]))
                reported_target = scores["without_smooth_objective"] if run_id == "a_no_smooth" else scores["common_objective"]
                max_error = max(max_error, abs(reported_target - row["objective"]), abs(original_scores["common_objective"] - original["objective"]))
                assert max_error < tolerance, (run_id, method, row["query_id"], max_error)
                candidates_equal = row["candidate_ids"] == original["candidate_ids"]
                same_candidates += int(candidates_equal)
                same_paths += int(row["track_ids"] == original["track_ids"])
                if row["track_ids"] != original["track_ids"]:
                    changed_by_suite[row["suite"]] += 1
                if run_id in ("changed_planner_exact", "beam_200") and method in ("a_grid", "b_set"):
                    assert candidates_equal, "Changing the searcher changed frozen candidates"
                difference = scores["common_objective"] - original_scores["common_objective"]
                differences.append(difference)
                if run_id == "changed_planner_exact":
                    assert row["oracle_certified"] and row["candidate_oracle_status"] == row["oracle_status"] == "ok"
                    assert abs(row["solver_gap"]) < tolerance
                    assert abs(row["objective"] - row["candidate_optimal_cost"]) < tolerance
                    assert difference < tolerance
                if run_id == "a_no_smooth":
                    no_smooth_differences.append(scores["without_smooth_objective"] - original_scores["without_smooth_objective"])
            for filename in ("manifest.json", "predictions.jsonl", "summary.json"):
                path = directory / filename
                source_files[path.relative_to(root).as_posix()] = sha256(path)
            runs.append({"run_id": run_id, "method": method, "requests": len(predictions), "successful": len(predictions),
                         "suite_counts": dict(Counter(row["suite"] for row in predictions)),
                         "only_declared_manifest_changes": True, "legal_paths_and_candidates": True,
                         "identical_candidate_lists": same_candidates, "identical_paths": same_paths,
                         "changed_paths_by_suite": dict(changed_by_suite), "maximum_direct_score_error": max_error,
                         "common_cost_lower_equal_higher": {"lower": sum(d < -tolerance for d in differences),
                                                            "equal": sum(abs(d) <= tolerance for d in differences),
                                                            "higher": sum(d > tolerance for d in differences)},
                         "largest_common_cost_increase": max(differences),
                         "largest_no_smooth_cost_increase": max(no_smooth_differences) if no_smooth_differences else None})
    report = {"verified_utc": datetime.now(timezone.utc).isoformat(), "scope": "pilot validation; fixed fitted seed",
              "runs": runs, "total_outputs": sum(row["requests"] for row in runs), "all_checks_passed": True,
              "direct_score_tolerance": tolerance, "score_reconstruction": "float64 raw selected-song and text embeddings; no evaluator cost calls",
              "exact_certificate_scope": "recorded full/candidate certificates checked for consistency; not a fresh independent exact solve",
              "source_files": source_files, "bank_hashes": bank_hashes, "verifier_sha256": sha256(__file__)}
    write_json(root / "reports/pilot/search_controls_verification.json", report)
    print(json.dumps({key: report[key] for key in ("total_outputs", "all_checks_passed", "runs")}, indent=2))


if __name__ == "__main__":
    main()
