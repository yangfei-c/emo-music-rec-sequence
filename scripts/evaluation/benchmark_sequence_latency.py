"""Repeat the existing evaluator after project training has stopped.

This measures cached-feature inference, not encoding or audio I/O. Each attempt
has fresh output directories; interrupted measurements are never pooled with a
new process. The shared evaluator is reused without changing model or search.
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

# Set before importing NumPy / Torch, identically for every method.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "4"
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import ROOT, load_config, read_jsonl, write_jsonl
from emomm_sequence.evaluation.sequence_benchmark import METHODS, evaluate
from emomm_sequence.training.records import environment, sha256, write_json
from emomm_sequence.training.sequence_selector import process_identity


def competing_jobs():
    """Identify live Python training/evaluation workers by their command line."""
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            info = process.info
            if info["pid"] == os.getpid() or "python" not in (info["name"] or "").lower():
                continue
            scripts = [Path(arg).name.lower() for arg in (info["cmdline"] or []) if arg.lower().endswith(".py")]
            if any(name.startswith(("train_", "evaluate_", "extract_")) or
                   name in {"run_pipeline.py", "benchmark_sequence_latency.py"} for name in scripts):
                found.append({"pid": info["pid"], "created_utc": datetime.fromtimestamp(info["create_time"], timezone.utc).isoformat(),
                              "scripts": scripts})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def select_timing_tasks(tasks, per_stratum=4):
    """Select request families without consulting scores or measured latency."""
    groups = defaultdict(dict)
    for task in tasks:
        if task["suite"] not in ("e2", "e3"):
            continue
        key = (task["suite"], task["source"], task["num_tracks"], task["budget"])
        family = task["query_family_id"]
        previous = groups[key].get(family)
        if previous is None or (task["phrase_index"], task["query_id"]) < (previous["phrase_index"], previous["query_id"]):
            groups[key][family] = task
    selected = []
    for key, values in sorted(groups.items()):
        families = sorted(values, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
        selected.extend(values[family] for family in families[:per_stratum])
    if not selected or len({t["query_id"] for t in selected}) != len(selected):
        raise ValueError("empty or duplicate timing manifest")
    return selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--per-stratum", type=int, default=4)
    p.add_argument("--defer-if-busy", action="store_true")
    p.add_argument("--run-id", default="isolated_latency_v1")
    p.add_argument("--tasks-file", type=Path, help="Optional frozen extension task list")
    p.add_argument("--main-run-id", default="main", help="Completed quality run to check timing outputs against")
    p.add_argument("--model-run-id", default="main", help="Validation-selected training run used by the measured learned methods")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = p.parse_args()
    if (args.repeats < 1 or args.per_stratum < 1 or len(set(args.methods)) != len(args.methods)
            or any(not value.replace("_", "").isalnum() for value in (args.run_id, args.main_run_id, args.model_run_id))):
        raise ValueError("positive repeat/stratum counts and an alphanumeric run ID are required")
    busy = competing_jobs()
    if busy:
        print(json.dumps({"status": "deferred_busy", "processes": busy}))
        return 0 if args.defer_if_busy else 2
    config, _ = load_config(args.config)
    root = args.root.resolve()
    split = "validation" if args.profile == "pilot" else "test"
    source_tasks = args.tasks_file.resolve() if args.tasks_file else root / "tasks" / args.profile / f"evaluate_{split}.jsonl"
    tasks = select_timing_tasks(read_jsonl(source_tasks), args.per_stratum)
    task_digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    bank_files = sorted({root / "banks" / f"{task['source']}_{task['split']}.npz" for task in tasks})
    main_outputs = {}
    main_files = {}
    quality_selections = {}
    for method in args.methods:
        main_file = root / "evaluations" / args.profile / split / method / "seed_42" / args.main_run_id / "predictions.jsonl"
        if not (main_file.parent / "summary.json").exists():
            raise ValueError(f"main evaluation has not completed: {method}")
        original = read_jsonl(main_file)
        manifest = json.loads((main_file.parent / "manifest.json").read_text(encoding="utf-8"))
        if manifest["task_sha256"] != sha256(source_tasks):
            raise ValueError("quality run and timing use different source task manifests")
        if manifest["config"] != config:
            raise ValueError("quality run and timing use different algorithm configurations")
        quality_selections[method] = manifest["checkpoint"]
        main_outputs[method] = {row["query_id"]: row for row in original}
        if len(original) != len(main_outputs[method]):
            raise ValueError("duplicate main evaluation requests")
        main_files[method] = sha256(main_file)
    checkpoints = {}
    for method, variant in (("b_set", "set"), ("direct_path", "direct_path"), ("dynamic_reduce", "dynamic_reduce")):
        if method not in args.methods:
            continue
        directory = root / "runs" / args.profile / variant / "seed_42" / args.model_run_id
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        digest = sha256(directory / "best.pt")
        if summary["status"] != "completed" or digest != selection["checkpoint_sha256"]:
            raise ValueError(f"incomplete or changed checkpoint: {method}")
        if selection != quality_selections[method]:
            raise ValueError(f"timing checkpoint differs from the completed quality run: {method}")
        checkpoints[method] = digest
    sources = ("scripts/evaluation/benchmark_sequence_latency.py", "src/emomm_sequence/evaluation/sequence_benchmark.py",
               "src/emomm_sequence/losses/sequence_cost.py", "src/emomm_sequence/inference/sequence_planner.py",
               "src/emomm_sequence/inference/trajectory_candidates.py", "src/emomm_sequence/models/sequence_selector.py",
               "src/emomm_sequence/models/sequence_reduction.py")
    fingerprint = {"source_tasks_sha256": sha256(source_tasks), "selected_tasks_sha256": task_digest,
                   "text_sha256": sha256(root / "cache/text_bank.npz"), "checkpoints": checkpoints,
                   "bank_sha256": {str(path.relative_to(root)): sha256(path) for path in bank_files},
                   "main_predictions_sha256": main_files,
                   "code_sha256": {s: sha256(ROOT / s) for s in sources}, "config": config,
                   "device": args.device, "repeats": args.repeats, "per_stratum": args.per_stratum,
                   "methods": list(args.methods), "main_run_id": args.main_run_id, "model_run_id": args.model_run_id,
                   "fitted_seed": 42, "warmup_passes": 1, "threads": 4}
    directory = root / "reports" / args.profile / args.run_id
    summary_file = directory / "summary.json"
    if summary_file.exists():
        existing = json.loads(summary_file.read_text(encoding="utf-8"))
        if (existing["fingerprint"] != fingerprint or existing["status"] != "completed"
                or sha256(Path(existing["measurements_file"])) != existing["measurements_sha256"]):
            raise ValueError("timing protocol changed; use a new --run-id")
        print(json.dumps({"status": "completed_existing", "summary": str(summary_file)}))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    attempt = directory / stamp
    task_file = attempt / "tasks.jsonl"
    write_jsonl(task_file, tasks)
    write_json(attempt / "manifest.json", {"fingerprint": fingerprint, "environment": environment(),
               "counterbalancing": "fixed-seed shuffled method order in each pass; one unreported warmup pass"})
    stop = threading.Event()
    overlaps = []
    def watch():
        while not stop.wait(1):
            competitors = competing_jobs()
            if competitors:
                overlaps.append({"utc": datetime.now(timezone.utc).isoformat(), "processes": competitors})
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    identity = process_identity()
    begin = time.monotonic()
    rows = []
    orders = []
    quality_parity = {"max_absolute_objective_difference": 0.0, "different_paths": 0, "compared_outputs": 0,
                      "objective_tolerance": 2e-6}
    expected_ids = {t["query_id"] for t in tasks}
    try:
        for repetition in range(args.repeats + 1):
            if overlaps or competing_jobs():
                raise RuntimeError("competing project computation detected; timing attempt invalid")
            methods = list(args.methods)
            random.Random(20260930 + repetition).shuffle(methods)
            orders.append(methods)
            run_id = f"timing_{stamp}_{repetition:02d}"
            write_json(attempt / "progress.json", {**identity, "status": "running", "pass": repetition,
                       "warmup": repetition == 0, "measured_passes": args.repeats})
            evaluate(config, root, argparse.Namespace(profile=args.profile, split=split, device=args.device,
                     tasks_file=str(task_file), suites=None, limit_tasks=0, seeds=[42], seed=42, methods=methods,
                     model_run_id=args.model_run_id, run_id=run_id, beam_width=0, planner="beam", no_smooth=False))
            for method in methods:
                path = root / "evaluations" / args.profile / split / method / "seed_42" / run_id / "predictions.jsonl"
                batch = read_jsonl(path)
                if len(batch) != len(tasks) or {r["query_id"] for r in batch} != expected_ids:
                    raise ValueError("incomplete or mismatched timing requests")
                if any(r["status"] != "ok" or not r["unique_content"] or len(r["track_ids"]) != r["num_tracks"] for r in batch):
                    raise ValueError("invalid output in timing experiment")
                for row in batch:
                    original = main_outputs[method][row["query_id"]]
                    delta = abs(row["objective"] - original["objective"])
                    if original["status"] != "ok" or delta > quality_parity["objective_tolerance"]:
                        raise ValueError("timing output quality changed from the completed main experiment")
                    if (row["start_id"], row["target_id"], row["library_size"]) != (original["start_id"], original["target_id"], original["library_size"]):
                        raise ValueError("timing changed the endpoint or library contract")
                    quality_parity["max_absolute_objective_difference"] = max(quality_parity["max_absolute_objective_difference"], delta)
                    quality_parity["different_paths"] += int(row["track_ids"] != original["track_ids"])
                    quality_parity["compared_outputs"] += 1
                if repetition:
                    rows.extend({**r, "timing_repeat": repetition, "timing_source": str(path)} for r in batch)
            if overlaps or competing_jobs():
                raise RuntimeError("competing project computation detected; timing attempt invalid")
        stop.set()
        watcher.join(timeout=2)
        if overlaps:
            raise RuntimeError("competing project computation detected; timing attempt invalid")
        groups = defaultdict(list)
        for row in rows:
            groups[(row["method"], row["suite"], row["source"], row["num_tracks"], row["requested_budget"], row["budget"])].append(row)
        table = []
        for key, values in sorted(groups.items()):
            item = dict(zip(("method", "suite", "source", "num_tracks", "requested_budget", "budget"), key))
            # Preserve individual repetitions; they are timing replicates, not new requests.
            item.update(unique_requests=len({r["query_id"] for r in values}), measurements=len(values),
                        p50_seconds=float(np.median([r["total_seconds"] for r in values])),
                        p95_seconds=float(np.quantile([r["total_seconds"] for r in values], .95)),
                        peak_cuda_mib=max(r["peak_cuda_mib"] for r in values))
            table.append(item)
        write_jsonl(attempt / "measurements.jsonl", rows)
        result = {"status": "completed", "fingerprint": fingerprint, "attempt": str(attempt),
                  "measurements_file": str(attempt / "measurements.jsonl"),
                  "measurements_sha256": sha256(attempt / "measurements.jsonl"),
                  "updated_utc": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.monotonic() - begin,
                  "requests_per_method": len(tasks), "measurements": len(rows), "orders": orders, "tables": table,
                  "overlap_samples": overlaps, "quality_parity": quality_parity,
                  "scope": "shared evaluator; cached-feature steady-process inference",
                  "limits": ["excludes audio/text encoding, checkpoint loading, disk I/O and metric computation",
                             "total_seconds adds measured objective setup to candidate/search time; shared setup may be reused within a pass",
                             "project Python training/evaluation workers monitored every second; unrelated OS activity is not excluded",
                             "timing repetitions are not independent scientific test samples; one fitted seed is used",
                             "Frozen methods on the declared quality run only; VA timing remains separate",
                             "requested and effective budgets stay separate, including saturated libraries"]}
        write_json(summary_file, result)
        write_json(attempt / "progress.json", {**identity, "status": "completed", "summary": str(summary_file)})
        text = ["# 独立运行的重复计时", "", f"设备：{args.device}；线程数：4；预热 1 轮，正式重复 {args.repeats} 轮。",
                "", "各方法使用相同固定请求、现有最优权重和原评价器；运行期间按秒检查项目训练/评价进程，发现并发则整次测量无效。",
                "", "| 方法 | 来源 | N | 请求 B / 实际 B | 不同请求 | 测量次数 | p50 / ms | p95 / ms |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
        for item in table:
            text.append(f"| {item['method']} | {item['source']} | {item['num_tracks']} | {item['requested_budget']} / {item['budget']} | {item['unique_requests']} | {item['measurements']} | {item['p50_seconds']*1000:.3f} | {item['p95_seconds']*1000:.3f} |")
        text.extend(["", "这是缓存特征后的目标构建、候选生成和搜索计时，不包含音频/文本编码、模型载入、磁盘 I/O 或评价指标计算。",
                     "无法排除操作系统的其他后台活动；多次计时不代表更多独立请求，也不估计训练种子稳定性。",
                     "质量结论仍使用完整 pilot 的配对指标；本次子集不用于重新选择模型或更新质量成绩。"])
        (directory / "重复计时.md").write_text("\n".join(text) + "\n", encoding="utf-8")
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
