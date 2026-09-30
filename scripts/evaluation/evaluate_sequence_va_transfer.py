"""Controlled VA/CQR replanning of frozen-route pools (NOT VA policy retraining)."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_config, load_bank, read_jsonl
from emomm_sequence.inference.sequence_va import VAProblem, va_beam_search
from emomm_sequence.evaluation.sequence_benchmark import label_metrics
from emomm_sequence.training.sequence_selector import process_identity
from emomm_sequence.training.records import write_json, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="full")
    p.add_argument("--methods", nargs="+", default=["a_grid", "a_refine", "b_set"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()
    config, root = load_config(args.config)
    # This extension is a post-selection test. It must not consume calibration/policy songs.
    task_file = root / "tasks" / args.profile / "evaluate_test.jsonl"
    task_map = {t["query_id"]: t for t in read_jsonl(task_file)}
    cache_dir = root / "cache" / "va_extension"
    with np.load(cache_dir / "text.npz", allow_pickle=False) as z:
        text_va = z["predicted_va"]
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    label_cache = {}
    banks, predictions = {}, {}
    out = root / "evaluations" / args.profile / "va_transfer"
    if args.limit_tasks:
        out = out / f"diagnostic_{args.limit_tasks}"
    out.mkdir(parents=True, exist_ok=True)
    identity = process_identity()
    # Freeze input files before starting. Missing methods/seeds are a failure, not a skipped comparison.
    paths = []
    for method in args.methods:
        seeds = args.seeds if method in ("b_set", "direct_path", "dynamic_reduce", "random") else [args.seeds[0]]
        for seed in seeds:
            path = root / "evaluations" / args.profile / "test" / method / f"seed_{seed}" / "main" / "predictions.jsonl"
            if not (path.parent / "summary.json").is_file():
                raise FileNotFoundError(f"finish source test evaluation first: {path.parent}")
            paths.append((method, seed, path))
    expected = {str(path): sha256(path) for _, _, path in paths}
    write_json(out / "inputs.json", {"sources": expected, "task_sha256": sha256(task_file),
                                    "supervised_assets": manifest["input_hashes"],
                                    "scope": "frozen_candidate_pool_VA_replanning; not a retrained B-VA policy"})
    output = out / "predictions.jsonl"
    if output.exists():
        raise FileExistsError("VA transfer output exists; preserve it and choose a fresh version")
    rows = []
    def progress(status, **extra):
        write_json(out / "progress.json", {**identity, "status": status, "scope": args.profile,
                   "updated_utc": datetime.now(timezone.utc).isoformat(), "rows": len(rows), **extra})
    progress("running")
    try:
        for method, seed, file in paths:
            requests = [r for r in read_jsonl(file) if r["suite"] in ("e2", "e3")]
            if args.limit_tasks:
                requests = requests[:args.limit_tasks]
            for original in requests:
                task = task_map[original["query_id"]]
                source = task["source"]
                if source not in banks:
                    banks[source] = load_bank(root, source, "test")
                    with np.load(cache_dir / f"{source}_test.npz", allow_pickle=False) as z:
                        predictions[source] = {k: z[k] for k in z.files}
                    if not np.array_equal(banks[source].ids, predictions[source]["track_ids"]):
                        raise ValueError("VA cache ID order changed")
                bank, pred = banks[source], predictions[source]
                lookup = {str(i): j for j, i in enumerate(bank.ids)}
                candidates = [lookup[i] for i in original["candidate_ids"]]
                start, target = lookup[original["start_id"]], lookup[original["target_id"]]
                query_path = np.linspace(text_va[task["start_text_id"]], text_va[task["target_text_id"]], task["num_tracks"])
                pref_text_id = task["preference_text_id"]
                pref = np.zeros(len(bank.ids))
                if pref_text_id is not None:
                    col = int(np.flatnonzero(pred["preference_text_ids"] == pref_text_id)[0])
                    pref = pred["preference_distances"][:, col]
                median = pred["quantiles"][:, :, 1]
                for mode in ("point", "raw_interval", "cqr_interval"):
                    if mode == "point":
                        lower = upper = median
                    elif mode == "raw_interval":
                        lower, upper = pred["quantiles"][:, :, 0], pred["quantiles"][:, :, 2]
                    else:
                        lower, upper = pred["cqr_lower"], pred["cqr_upper"]
                    problem = VAProblem(bank.ids, median, lower, upper, query_path, pref, start, target, pref_text_id is not None)
                    begin = time.perf_counter()
                    result = va_beam_search(problem, candidates, width=config["planner"]["beam_width"])
                    row = {"query_id": task["query_id"], "query_family_id": task["query_family_id"], "source": source,
                           "candidate_method": method, "seed": seed, "mode": mode, "num_tracks": task["num_tracks"],
                           "budget": len(candidates), "status": result.status, "va_objective": result.objective,
                           "search_seconds": time.perf_counter() - begin, "track_ids": bank.ids[list(result.path)].tolist(),
                           "policy_retrained_on_va": False, "playlist_coverage_guarantee": False}
                    if result.status == "ok":
                        row["point_va_objective"] = problem.point_cost(result.path)
                        row.update(label_metrics(root, bank, task, result.path, label_cache))
                        if source in ("deam", "pmemo", "merge") and mode != "point":
                            labels = label_cache[(source, "test")]
                            iy = [labels["lookup"][str(bank.ids[i])] for i in result.path]
                            y = labels["va"][iy]
                            covered = ((y >= lower[list(result.path)] - 1e-12) & (y <= upper[list(result.path)] + 1e-12)).all(axis=1)
                            row["selected_song_joint_coverage"] = float(covered.mean())
                            row["whole_playlist_covered"] = bool(covered.all())
                            row["selected_interval_width"] = float((upper[list(result.path)] - lower[list(result.path)]).mean())
                            row["calibration_source_known"] = source in ("deam", "pmemo")
                    rows.append(row)
                    with output.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                progress("running")
        write_json(out / "summary.json", {"status": "completed", "rows": len(rows), "scope": "VA/CQR replan of frozen candidate pools",
                                          "failed": sum(r["status"] != "ok" for r in rows), "not_equivalent_to_retrained_va_selector": True})
        progress("completed", expected_outputs=[str(output), str(out / "summary.json")])
    except Exception as exc:
        progress("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
