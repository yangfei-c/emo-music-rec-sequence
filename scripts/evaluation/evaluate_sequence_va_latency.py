"""Measure all nine VA combinations in one idle process, using existing algorithms.

No audio/text encoding, disk reads, checkpoint loading or label metrics are timed.
An incomplete or overlapping attempt is never merged with a later attempt.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from unittest.mock import patch

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"
import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_sequence_latency import competing_jobs, select_timing_tasks
from emomm_sequence.data.sequence_tasks import load_bank, load_config, read_jsonl, write_jsonl
from emomm_sequence.data import sequence_va_tasks as va_tasks
from emomm_sequence.inference.sequence_va import va_beam_search
from emomm_sequence.training.records import environment, sha256, write_json
from emomm_sequence.training.sequence_selector import load_model, process_identity

MODES = ("point", "raw_interval", "cqr_interval")
METHODS = ("a_grid", "a_refine", "b_set")
COMBINATIONS = tuple((mode, method) for mode in MODES for method in METHODS)
TOLERANCE = 2e-6


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


class SearchMeter:
    """Wrap existing beam calls, including the calls made inside va_refine."""

    def __init__(self):
        self.seconds = 0.0
        self.calls = 0
        self.expansions = 0
        self.candidate_union = set()

    def __call__(self, problem, candidates, *args, **kwargs):
        begin = time.perf_counter()
        result = va_beam_search(problem, candidates, *args, **kwargs)
        self.seconds += time.perf_counter() - begin
        self.calls += 1
        self.expansions += result.expansions
        self.candidate_union.update(map(int, candidates))
        return result


def measure_request(inputs, source_bank, task, texts, method, model, device, width):
    """Time the production operations; validate and format only after the timer."""
    synchronize(device)
    baseline = torch.cuda.memory_allocated() if device == "cuda" else 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    begin = time.perf_counter()
    bank, problem, _ = inputs.make(source_bank, task, texts)
    setup_seconds = time.perf_counter() - begin
    budget = min(task["budget"], problem.m)
    meter = SearchMeter()
    inference_begin = time.perf_counter()
    candidate_begin = time.perf_counter()
    if method == "b_set":
        if model is None:
            raise ValueError("B requires its separately trained VA model")
        with torch.inference_mode():
            output = model.select(torch.from_numpy(problem.x).to(device),
                                  torch.from_numpy(problem.query).to(device),
                                  problem.n, budget, problem.start, problem.target)
        candidates = np.asarray(output.indices)
    else:
        candidates = va_tasks.va_candidates(problem, budget)
    synchronize(device)
    candidate_seconds = time.perf_counter() - candidate_begin
    refine_seconds = 0.0
    if method == "a_refine":
        # This process runs one request at a time. The patch is restored even if
        # search fails; the production source and algorithm are unchanged.
        refine_begin = time.perf_counter()
        with patch.object(va_tasks, "va_beam_search", meter):
            candidates, result = va_tasks.va_refine(problem, candidates, width)
        refine_seconds = time.perf_counter() - refine_begin - meter.seconds
    elif method in ("a_grid", "b_set"):
        result = meter(problem, candidates, width)
    else:
        raise ValueError(method)
    synchronize(device)
    inference_seconds = time.perf_counter() - inference_begin
    peak = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    indices = list(result.path)
    if (result.status != "ok" or len(indices) != problem.n or len(set(indices)) != problem.n
            or indices[0] != problem.start or indices[-1] != problem.target
            or len(set(bank.groups[indices].tolist())) != problem.n
            or len(candidates) != budget or len(set(candidates.tolist())) != budget
            or not set(indices).issubset(set(candidates.tolist()))):
        raise ValueError("invalid playlist or candidate budget during timing")
    # Algorithm-level counts follow the existing implementation, not hardware
    # FLOPs: each full vector scored/sorted counts as one scan of M songs.
    proposal_scans = (meter.calls - 1) * (problem.n - 2) if method == "a_refine" else 0
    initial_scans = budget - 2 if method == "b_set" else problem.n
    full_scans = initial_scans + proposal_scans
    return {"query_id": task["query_id"], "query_family_id": task["query_family_id"],
            "source": task["source"], "suite": task["suite"], "split": task["split"],
            "mode": inputs.mode, "method": method, "seed": 42, "num_tracks": problem.n,
            "requested_budget": task["budget"], "budget": budget, "library_size": problem.m,
            "start_id": str(bank.ids[problem.start]), "target_id": str(bank.ids[problem.target]),
            "track_ids": bank.ids[indices].tolist(), "candidate_ids": bank.ids[candidates].tolist(),
            "status": result.status, "unique_content": True, "objective": result.objective,
            "va_endpoint_protocol": va_tasks.VA_ENDPOINT_PROTOCOL,
            "setup_seconds": setup_seconds, "candidate_seconds": candidate_seconds,
            "search_seconds": meter.seconds, "refinement_seconds": max(0.0, refine_seconds),
            "inference_seconds": inference_seconds, "total_seconds": setup_seconds + inference_seconds,
            "timing_overhead_seconds": max(0.0, inference_seconds - candidate_seconds - meter.seconds - refine_seconds),
            "full_scans": full_scans, "scored_song_occurrences": full_scans * problem.m,
            "solver_calls": meter.calls, "search_expansions": meter.expansions,
            "cumulative_candidates": len(meter.candidate_union),
            "resident_cuda_mib": baseline / 2**20, "peak_cuda_mib": peak / 2**20,
            "incremental_peak_cuda_mib": (peak - baseline) / 2**20}


def verify_parity(row, original, totals):
    fields = ("query_id", "query_family_id", "source", "suite", "split", "mode", "method", "seed",
              "num_tracks", "budget", "library_size", "start_id", "target_id", "va_endpoint_protocol")
    if any(row[key] != original[key] for key in fields) or original["status"] != "ok":
        raise ValueError("timing changed the main request, eligible library or endpoint contract")
    delta = abs(row["objective"] - original["objective"])
    if not np.isfinite(delta) or delta > TOLERANCE:
        raise ValueError("timing output objective changed from the completed main evaluation")
    totals["max_absolute_objective_difference"] = max(totals["max_absolute_objective_difference"], delta)
    totals["different_paths"] += int(row["track_ids"] != original["track_ids"])
    totals["different_candidate_sets"] += int(set(row["candidate_ids"]) != set(original["candidate_ids"]))
    totals["compared_outputs"] += 1


def inspect_inputs(root, config, profile, tasks):
    split = "validation" if profile == "pilot" else "test"
    task_file = root / "tasks" / profile / f"evaluate_{split}.jsonl"
    expected_ids = {t["query_id"] for t in read_jsonl(task_file) if t["suite"] in ("e2", "e3")}
    references, files, pending, checkpoints = {}, {}, [], {}
    for mode, method in COMBINATIONS:
        directory = root / "evaluations" / profile / "va_retrained" / split / mode / method / "seed_42/main"
        if not (directory / "summary.json").exists():
            pending.append(f"{mode}/{method}")
            continue
        summary, manifest = read(directory / "summary.json"), read(directory / "manifest.json")
        if (summary["status"] != "completed" or summary["failed"] != 0 or manifest["config"] != config
                or manifest["va_endpoint_protocol"] != va_tasks.VA_ENDPOINT_PROTOCOL
                or manifest["task_sha256"] != sha256(task_file) or manifest["limit_tasks"] != 0
                or manifest["mode"] != mode or manifest["method"] != method or manifest["seed"] != 42):
            raise ValueError(f"incompatible or unsuccessful main evaluation: {mode}/{method}")
        records = read_jsonl(directory / "predictions.jsonl")
        lookup = {row["query_id"]: row for row in records}
        if (len(lookup) != len(records) or summary["rows"] != len(records) or lookup.keys() != expected_ids
                or not {t["query_id"] for t in tasks} <= lookup.keys()):
            raise ValueError("duplicate or missing main requests")
        references[mode, method] = lookup
        for filename in ("summary.json", "manifest.json", "predictions.jsonl"):
            path = directory / filename
            files[str(path)] = sha256(path)
        if method == "b_set":
            training = root / "runs" / profile / "set/seed_42" / f"va_{mode}"
            selection, training_manifest = read(training / "selection.json"), read(training / "manifest.json")
            if (read(training / "summary.json")["status"] != "completed"
                    or selection != manifest["checkpoint"] or selection.get("test_used", True)
                    or training_manifest.get("objective") != mode
                    or training_manifest.get("va_feature_dimensions") != 7
                    or training_manifest.get("va_endpoint_protocol") != va_tasks.VA_ENDPOINT_PROTOCOL
                    or sha256(training / "best.pt") != selection["checkpoint_sha256"]):
                raise ValueError(f"incomplete or incompatible VA model: {mode}")
            for key in ("planner", "assets", "route_b"):
                if training_manifest["config"][key] != config[key]:
                    raise ValueError("VA training and timing configurations differ")
            checkpoints[mode] = selection["checkpoint_sha256"]
            for filename in ("best.pt", "manifest.json", "selection.json", "summary.json"):
                path = training / filename
                files[str(path)] = sha256(path)
    paths = [root / "tasks" / profile / f"evaluate_{split}.jsonl", root / "cache/text_bank.npz",
             root / "cache/va_extension/manifest.json", root / "cache/va_extension/text.npz"]
    for source in sorted({t["source"] for t in tasks}):
        paths.extend((root / "banks" / f"{source}_{split}.npz",
                      root / "cache/va_extension" / f"{source}_{split}.npz"))
    for path in paths:
        files[str(path)] = sha256(path)
    return references, files, pending, checkpoints


def summarize(rows):
    groups = defaultdict(list)
    keys = ("mode", "method", "suite", "source", "num_tracks", "requested_budget", "budget", "library_size")
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    table = []
    for key, values in sorted(groups.items()):
        item = dict(zip(keys, key))
        item.update(unique_requests=len({r["query_id"] for r in values}), measurements=len(values),
                    p50_seconds=float(np.median([r["total_seconds"] for r in values])),
                    p95_seconds=float(np.quantile([r["total_seconds"] for r in values], .95)))
        for field in ("setup_seconds", "candidate_seconds", "search_seconds", "refinement_seconds",
                      "timing_overhead_seconds", "full_scans", "scored_song_occurrences", "solver_calls",
                      "search_expansions", "cumulative_candidates"):
            item[f"mean_{field}"] = float(np.mean([r[field] for r in values]))
        for field in ("resident_cuda_mib", "peak_cuda_mib", "incremental_peak_cuda_mib"):
            item[field] = max(r[field] for r in values)
        table.append(item)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    parser.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--per-stratum", type=int, default=4)
    parser.add_argument("--run-id", default="va_isolated_latency_v1")
    parser.add_argument("--preflight", action="store_true", help="Inspect inputs only; does not time or load models")
    parser.add_argument("--defer-if-busy", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1 or args.per_stratum < 1 or not args.run_id.replace("_", "").isalnum():
        parser.error("positive counts and an alphanumeric run ID are required")
    busy = competing_jobs()
    if busy and not args.preflight:
        print(json.dumps({"status": "deferred_busy", "processes": busy}))
        return 0 if args.defer_if_busy else 2
    config, _ = load_config(args.config)
    root = args.root.resolve()
    split = "validation" if args.profile == "pilot" else "test"
    tasks = select_timing_tasks(read_jsonl(root / "tasks" / args.profile / f"evaluate_{split}.jsonl"), args.per_stratum)
    references, files, pending, checkpoints = inspect_inputs(root, config, args.profile, tasks)
    directory = root / "reports" / args.profile / args.run_id
    if args.preflight or pending:
        report = {"status": "pending_main_results" if pending else "ready_not_measured", "pending": pending,
                  "requests_per_combination": len(tasks), "combinations": len(COMBINATIONS),
                  "planned_measurements": len(tasks) * len(COMBINATIONS) * args.repeats,
                  "competing_jobs": busy, "updated_utc": datetime.now(timezone.utc).isoformat()}
        write_json(directory / "preflight.json", report)
        print(json.dumps(report))
        return 2 if pending else 0
    sources = [Path(__file__), Path(__file__).with_name("benchmark_sequence_latency.py")]
    sources.extend(WORKSPACE / name for name in (
        "src/emomm_sequence/data/sequence_va_tasks.py", "src/emomm_sequence/data/sequence_tasks.py",
        "src/emomm_sequence/inference/sequence_va.py", "src/emomm_sequence/inference/trajectory_candidates.py",
        "src/emomm_sequence/inference/sequence_planner.py", "src/emomm_sequence/losses/sequence_cost.py",
        "src/emomm_sequence/models/sequence_selector.py", "src/emomm_sequence/training/sequence_selector.py"))
    files.update({str(path): sha256(path) for path in sources})
    files[str((WORKSPACE / args.config).resolve())] = sha256(WORKSPACE / args.config)
    fingerprint = {"files_sha256": files, "config": config, "checkpoints": checkpoints,
                   "selected_tasks_sha256": hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest(),
                   "device": args.device, "threads": 4, "repeats": args.repeats, "per_stratum": args.per_stratum,
                   "warmup_passes": 1, "fitted_seed": 42, "combinations": [list(c) for c in COMBINATIONS]}
    summary_file = directory / "summary.json"
    if summary_file.exists():
        existing = read(summary_file)
        if (existing["status"] != "completed" or existing["fingerprint"] != fingerprint or existing["overlap_samples"]
                or sha256(Path(existing["measurements_file"])) != existing["measurements_sha256"]):
            raise ValueError("completed timing changed; use a new --run-id")
        print(json.dumps({"status": "completed_existing", "summary": str(summary_file)}))
        return 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    attempt = directory / stamp
    write_jsonl(attempt / "tasks.jsonl", tasks)
    write_json(attempt / "manifest.json", {"fingerprint": fingerprint, "environment": environment()})
    identity, stop, overlaps = process_identity(), threading.Event(), []

    def watch():
        while not stop.wait(1):
            found = competing_jobs()
            if found:
                overlaps.append({"utc": datetime.now(timezone.utc).isoformat(), "processes": found})

    def require_idle():
        found = competing_jobs()
        if overlaps or found:
            if found:
                overlaps.append({"utc": datetime.now(timezone.utc).isoformat(), "processes": found})
            raise RuntimeError("concurrent project computation; timing attempt invalid")

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    begin, rows, orders = time.monotonic(), [], []
    parity = {"max_absolute_objective_difference": 0.0, "different_paths": 0,
              "different_candidate_sets": 0, "compared_outputs": 0, "objective_tolerance": TOLERANCE}
    try:
        require_idle()
        banks = {source: load_bank(root, source, split) for source in {t["source"] for t in tasks}}
        with np.load(root / "cache/text_bank.npz", allow_pickle=False) as data:
            texts = {k: data[k] for k in data.files}
        inputs = {mode: va_tasks.VAInputs(root, mode) for mode in MODES}
        models = {}
        for mode in MODES:
            models[mode], _ = load_model(config, root, args.profile, "set", 42, args.device, f"va_{mode}")
            if models[mode].project[0].in_features != config["assets"]["embedding_dim"] + 7:
                raise ValueError("VA checkpoint feature size changed")
            # Preload prediction files before any measured request.
            for source, bank in banks.items():
                inputs[mode].make(bank, next(t for t in tasks if t["source"] == source), texts)
        require_idle()
        for repetition in range(args.repeats + 1):
            combinations = list(COMBINATIONS)
            random.Random(20260930 + repetition).shuffle(combinations)
            orders.append([list(c) for c in combinations])
            for mode, method in combinations:
                require_idle()
                for task in tasks:
                    row = measure_request(inputs[mode], banks[task["source"]], task, texts, method,
                                          models[mode] if method == "b_set" else None,
                                          args.device, config["planner"]["beam_width"])
                    verify_parity(row, references[mode, method][task["query_id"]], parity)
                    if repetition:
                        rows.append({**row, "timing_repeat": repetition})
                    write_json(attempt / "progress.json", {**identity, "status": "running", "pass": repetition,
                               "mode": mode, "method": method, "query_id": task["query_id"], "rows": len(rows),
                               "updated_utc": datetime.now(timezone.utc).isoformat()})
                require_idle()
        require_idle()
        for filename, expected in files.items():
            if sha256(Path(filename)) != expected:
                raise ValueError(f"timing input changed during measurement: {filename}")
        require_idle()
        measurements = attempt / "measurements.jsonl"
        write_jsonl(measurements, rows)
        result = {"status": "completed", "fingerprint": fingerprint, "attempt": str(attempt),
                  "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begin,
                  "requests_per_combination": len(tasks), "measurements": len(rows), "orders": orders,
                  "tables": summarize(rows), "quality_parity": parity, "overlap_samples": overlaps,
                  "measurements_file": str(measurements), "measurements_sha256": sha256(measurements),
                  "scope": "cached-feature VA inference, setup plus candidate generation/refinement plus search",
                  "counter_scope": "candidate scans only: A N full orderings, refinement (N-2) per completed proposal round; B (B-2) score vectors; excludes setup, projection and beam work from scan count",
                  "limits": ["excludes encoding, prediction/model loading, disk I/O, output validation and label metrics",
                             "all three B models resident throughout; peak CUDA allocation and increase above baseline both reported",
                             "refinement residual includes proposal generation and timing wrapper overhead; search time sums original solver calls",
                             "algorithm scan counters are declared operations, not measured FLOPs; search expansions are observed",
                             "project Python workers checked every second and between combinations; other OS activity may remain",
                             "one fitted seed; timing repetitions do not increase independent quality sample size",
                             "quality parity allows same-objective alternative paths; full main outputs remain quality evidence"]}
        stop.set()
        watcher.join(timeout=2)
        require_idle()
        lines = ["# VA 九种组合的独立重复计时", "", f"设备 {args.device}；4 线程；预热 1 轮，正式 {args.repeats} 轮；每组合 {len(tasks)} 个固定请求。", "",
                 "| VA | 方法 | 来源 | N | 请求 B / 实际 B | p50 / ms | p95 / ms | 峰值 / MiB |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for item in result["tables"]:
            lines.append(f"| {item['mode']} | {item['method']} | {item['source']} | {item['num_tracks']} | {item['requested_budget']} / {item['budget']} | {item['p50_seconds']*1000:.3f} | {item['p95_seconds']*1000:.3f} | {item['peak_cuda_mib']:.2f} |")
        lines.extend(["", "特征和预测已缓存；目标构建、候选生成、候选改进和搜索分别记录，不包含编码、模型载入、磁盘或质量评价。",
                      "三个 B 模型在整个过程常驻，summary.json 同时报告总峰值及相对常驻基线的显存增量。",
                      "所有重跑输出与主结果核对任务、端点及目标值；本子集只测成本，不替代完整实验的质量评价。",
                      "扫描次数按现有算法操作推算，搜索扩展和调用次数实际记录；这些不是 FLOPs。计时重复也不是更多独立质量样本。"])
        (directory / "重复计时.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        write_json(summary_file, result)
        write_json(attempt / "progress.json", {**identity, "status": "completed", "summary": str(summary_file)})
        print(json.dumps({"status": "completed", "summary": str(summary_file), "measurements": len(rows)}))
        return 0
    except Exception as exc:
        write_json(attempt / "progress.json", {**identity, "status": "invalid_or_failed", "error": repr(exc), "overlaps": overlaps})
        raise
    finally:
        stop.set()
        watcher.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
