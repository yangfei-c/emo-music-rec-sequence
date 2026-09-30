"""Observed training/inference cost ledger, including non-amortizable training."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import read_jsonl
from emomm_sequence.training.records import write_json, sha256
from emomm_sequence.training.sequence_selector import task_selector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--latency-run-id", default="isolated_latency_v1")
    p.add_argument("--va-latency-run-id", default="va_isolated_latency_v1")
    args = p.parse_args()
    if any(not value.replace("_", "").isalnum() for value in (args.latency_run_id, args.va_latency_run_id)):
        p.error("latency run IDs must contain only letters, digits and underscores")
    training = []
    for file in sorted((args.root / "runs" / args.profile).glob("*/seed_*/*/summary.json")):
        if file.parent.name.startswith("startup") or "resume_check" in file.parent.name:
            continue
        summary = json.loads(file.read_text(encoding="utf-8"))
        if summary["status"] != "completed":
            continue
        manifest = json.loads((file.parent / "manifest.json").read_text(encoding="utf-8"))
        selection = json.loads((file.parent / "selection.json").read_text(encoding="utf-8"))
        actual_hash = sha256(file.parent / "best.pt")
        if actual_hash != selection["checkpoint_sha256"]:
            raise ValueError("selected checkpoint hash changed")
        parameters = summary.get("trainable_parameters")
        if parameters is None:
            model = task_selector(manifest["config"], manifest["variant"], manifest.get("objective", "frozen"),
                                  manifest.get("va_feature_dimensions", 6), **manifest.get("model_overrides", {}))
            parameters = sum(v.numel() for v in model.parameters() if v.requires_grad)
        training.append({"variant": summary["variant"], "run_id": file.parent.name, "seed": summary["seed"],
                         "objective": manifest.get("objective", "frozen"), "parameters": parameters,
                         "elapsed_training_seconds": summary["elapsed_seconds"], "optimizer_updates": summary["global_step"],
                         "selected_phase": selection["phase"], "selected_epoch_zero_based": selection["epoch"],
                         "checkpoint_sha256": actual_hash, "source": str(file)})
    grouped = defaultdict(list)
    split = "validation" if args.profile == "pilot" else "test"
    for file in sorted((args.root / "evaluations" / args.profile / split).glob("*/seed_*/main/predictions.jsonl")):
        if not (file.parent / "summary.json").exists():
            continue
        for row in read_jsonl(file):
            grouped[(row["method"], row["suite"], row["num_tracks"], row["requested_budget"])].append(row)
    inference = []
    for key, rows in sorted(grouped.items()):
        item = dict(zip(("method", "suite", "num_tracks", "requested_budget"), key))
        item.update(requests=len(rows), fitted_seeds=sorted({r["seed"] for r in rows}),
                    p50_seconds=float(np.median([r["total_seconds"] for r in rows])),
                    p95_seconds=float(np.quantile([r["total_seconds"] for r in rows], .95)),
                    peak_cuda_mib=max(r["peak_cuda_mib"] for r in rows),
                    failed=sum(r["status"] != "ok" for r in rows))
        for field in ("setup_seconds", "candidate_seconds", "full_scans", "scored_song_occurrences", "search_expansions", "solver_calls", "cumulative_candidates"):
            item[f"mean_{field}"] = float(np.mean([r[field] for r in rows]))
        # For A-refine this residual includes candidate refinement as well as search.
        item["mean_search_and_refinement_seconds"] = float(np.mean([r["inference_seconds"] - r["candidate_seconds"] for r in rows]))
        inference.append(item)
    teacher_costs = []
    for directory in sorted((args.root / "cache").glob("teachers*")):
        if not directory.is_dir():
            continue
        entries = [json.loads(f.read_text(encoding="utf-8")) for f in directory.glob("*.json")]
        known = [r["generation_seconds"] for r in entries if "generation_seconds" in r]
        teacher_costs.append({"cache": directory.name, "tasks": len(entries), "tasks_with_recorded_generation_time": len(known),
                              "recorded_seconds": float(sum(known)), "total_cost_known": bool(entries) and len(known) == len(entries),
                              "cost_scope": "one-time cache generation; may be shared across runs; not an extra additive charge if already inside training time"})
    lookup = {(r["method"], r["suite"], r["num_tracks"], r["requested_budget"]): r for r in inference}
    main_b = [r for r in training if r["variant"] == "set" and r["run_id"] == "main" and r["objective"] == "frozen"]
    amortization = []
    if main_b:
        average_training = float(np.mean([r["elapsed_training_seconds"] for r in main_b]))
        for (method, suite, n, budget), b in lookup.items():
            if method != "b_set":
                continue
            for reference in ("a_grid", "a_refine", "direct_path", "dynamic_reduce"):
                a = lookup.get((reference, suite, n, budget))
                if a is None:
                    continue
                saved = a["p50_seconds"] - b["p50_seconds"]
                amortization.append({"reference": reference, "suite": suite, "num_tracks": n, "requested_budget": budget,
                    "seconds_saved_per_request": saved, "training_seconds_per_fitted_seed": average_training,
                    "queries_to_amortize_B_training": average_training / saved if saved > 0 else None,
                    "interpretation": "provisional cost-only estimate, requiring comparable quality" if saved > 0 else "no inference speedup; no cost amortization",
                    "excludes": "unrecorded teacher generation, reference training, and initial asset/model loading"})
    isolated_file = args.root / "reports" / args.profile / args.latency_run_id / "summary.json"
    isolated = None
    if isolated_file.exists():
        measured = json.loads(isolated_file.read_text(encoding="utf-8"))
        if measured["status"] != "completed" or measured["overlap_samples"]:
            raise ValueError("isolated timing report is not valid and complete")
        isolated = {"summary": str(isolated_file), "sha256": sha256(isolated_file),
                    "requests_per_method": measured["requests_per_method"], "measurements": measured["measurements"],
                    "device": measured["fingerprint"]["device"], "quality_parity": measured["quality_parity"],
                    "scope": measured["scope"]}
    va_file = args.root / "reports" / args.profile / args.va_latency_run_id / "summary.json"
    isolated_va = None
    if va_file.exists():
        measured = json.loads(va_file.read_text(encoding="utf-8"))
        protocol = measured["fingerprint"]
        combinations = {(mode, method) for mode in ("point", "raw_interval", "cqr_interval")
                        for method in ("a_grid", "a_refine", "b_set")}
        expected = measured["requests_per_combination"] * len(combinations) * protocol["repeats"]
        if (measured["status"] != "completed" or measured["overlap_samples"]
                or {tuple(c) for c in protocol["combinations"]} != combinations
                or measured["measurements"] != expected
                or measured["quality_parity"]["compared_outputs"] != expected + measured["requests_per_combination"] * 9
                or sha256(Path(measured["measurements_file"])) != measured["measurements_sha256"]):
            raise ValueError("VA isolated timing is incomplete or changed")
        isolated_va = {"summary": str(va_file), "sha256": sha256(va_file), "measurements": measured["measurements"],
                       "requests_per_combination": measured["requests_per_combination"], "device": protocol["device"],
                       "quality_parity": measured["quality_parity"], "scope": measured["scope"]}
    result = {"updated_utc": datetime.now(timezone.utc).isoformat(), "profile": args.profile,
              "report_source_sha256": sha256(Path(__file__)),
              "training": training, "inference": inference, "teacher_generation": teacher_costs, "amortization": amortization,
              "isolated_latency": isolated, "requested_latency_run_id": args.latency_run_id,
              "isolated_va_latency": isolated_va, "requested_va_latency_run_id": args.va_latency_run_id,
              "limits": ["inference/amortization here retain development timings; isolated_latency links a separate controlled repeat measurement when available",
                         "full_scans/scored_song_occurrences are declared algorithm-level counters, not measured FLOPs",
                         "training elapsed sums resumed training segments, but excludes initialization before the training timer",
                         "early teacher caches did not record generation time; a historical total cannot be reconstructed",
                         "A-refine residual timing combines refinement and search; a separate per-solver timer was not recorded"]}
    out = args.root / "reports" / args.profile
    write_json(out / "resources.json", result)
    lines = ["# 实测资源记录", "", "这是开发运行的成本记录，尚非隔离环境的性能排名。", "",
             "| 模型 | run | seed | 参数 | 更新次数 | 训练秒数 | 验证选中的轮次 |", "|---|---|---:|---:|---:|---:|---|"]
    for r in training:
        lines.append(f"| {r['variant']} | {r['run_id']} | {r['seed']} | {r['parameters']} | {r['optimizer_updates']} | {r['elapsed_training_seconds']:.1f} | {r['selected_phase']} / {r['selected_epoch_zero_based']}（从 0 开始） |")
    lines.extend(["", "逐场景延迟分位数、显存、扫描/扩展计数、教师缓存成本与摊销计算见 resources.json。",
                  "B 比参照方法更慢时，摊销查询数记为 null，并明确不存在推理加速收益；不会用负数或无穷大冒充收益。",
                  "第一轮 Frozen 教师缓存未单独记录生成耗时，因此不能补造这部分历史数据。新生成缓存已经记录耗时与代码摘要；共享缓存成本不重复叠加到各模型总耗时。",
                  "A-refine 的剩余耗时包含局部候选改进和搜索，尚不能拆成纯求解器时间。完整训练结论仍应同时考虑质量，不能仅以速度替代算法效果。"])
    lines.extend(["", (f"独立重复计时已完成，见同目录 {args.latency_run_id}/重复计时.md 和 summary.json；上面的开发计时保留原值，未与独立计时混算。"
                       if isolated else "独立重复计时尚未完成；入口会等待项目训练/评价进程退出后，预热并执行五轮正式测量。")])
    lines.extend(["", (f"VA 九种组合的独立计时已完成，见 {args.va_latency_run_id}/重复计时.md；单独记录目标构建、候选生成、局部改进与纯搜索耗时。"
                       if isolated_va else "VA 九种组合的独立计时尚未完成；须等三种 B-VA 训练及九组评价完整且计算进程空闲后执行。已有 VA 开发计时不用于速度排名。")])
    (out / "资源记录.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"completed_training_runs": len(training), "inference_groups": len(inference)}))


if __name__ == "__main__":
    main()
