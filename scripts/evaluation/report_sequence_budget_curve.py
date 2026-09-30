"""Keep requested budgets separate and compare complete paired budget curves."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_sequence_routes import family_bootstrap
from emomm_sequence.data.sequence_tasks import load_bank, load_config, read_jsonl
from emomm_sequence.training.records import sha256, write_json

METHODS = ("a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce")
METRICS = ("objective", "true_va_mae", "true_endpoint_mae", "true_step_deviation", "observed_preference_tag_match")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(records):
    """Each source/family contributes once to each N/requested-B comparison."""
    groups = defaultdict(list)
    keys = ("method", "suite", "source", "num_tracks", "requested_budget", "budget", "library_size")
    paired = defaultdict(dict)
    for row in records:
        groups[tuple(row[k] for k in keys)].append(row)
        if row["method"] in paired[row["query_id"]]:
            raise ValueError("duplicate request/method")
        paired[row["query_id"]][row["method"]] = row
    table = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(keys, key))
        item.update(requests=len(rows), failed=sum(r["status"] != "ok" for r in rows),
                    saturated=all(r["budget_saturated"] for r in rows),
                    development_p50_seconds=float(np.median([r["total_seconds"] for r in rows])),
                    development_p95_seconds=float(np.quantile([r["total_seconds"] for r in rows], .95)))
        for metric in METRICS:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            item[metric] = float(np.mean(values)) if values else None
            item[f"{metric}_observed_requests"] = len(values)
        table.append(item)
    contrasts = []
    contract = ("query_family_id", "source", "suite", "split", "seed", "num_tracks", "requested_budget",
                "budget", "library_size", "start_id", "target_id", "phrase_index")
    for reference in (method for method in METHODS if method != "b_set"):
        clusters, counts, excluded = defaultdict(lambda: defaultdict(dict)), Counter(), Counter()
        for methods in paired.values():
            if set(methods) != set(METHODS):
                raise ValueError("all five methods must be complete; do not compare an intersection")
            b, a = methods["b_set"], methods[reference]
            if any(b[field] != a[field] for field in contract):
                raise ValueError("paired request, budget, endpoint or library mismatch")
            group = (b["suite"], b["num_tracks"], b["requested_budget"])
            counts[group] += 1
            for metric in METRICS:
                key = (*group, metric)
                if metric != "objective" and (b["status"] != "ok" or a["status"] != "ok"):
                    excluded[key] += 1
                    continue
                if b.get(metric) is None or a.get(metric) is None:
                    continue
                families = clusters[key][b["source"]]
                if b["query_family_id"] in families:
                    raise ValueError("duplicate family within a source/N/budget cell")
                families[b["query_family_id"]] = b[metric] - a[metric]
        for (suite, n, budget, metric), sources in sorted(clusters.items()):
            contrasts.append({"contrast": f"b_set_minus_{reference}", "suite": suite, "num_tracks": n,
                              "requested_budget": budget, "metric": metric,
                              "paired_requests": counts[suite, n, budget],
                              "metric_requests": sum(len(v) for v in sources.values()),
                              "excluded_failure_pairs": excluded[suite, n, budget, metric],
                              "favors_b": "positive" if metric == "observed_preference_tag_match" else "negative",
                              **family_bootstrap(sources)})
    return table, contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--run-id", default="budget_curve_v1")
    parser.add_argument("--latency-run-id", default="budget_curve_latency_v1")
    args = parser.parse_args()
    if any(not v.replace("_", "").isalnum() for v in (args.run_id, args.latency_run_id)):
        parser.error("run IDs must contain letters, digits and underscores only")
    config, root = load_config(args.config)
    split = "validation" if args.profile == "pilot" else "test"
    task_dir = root / "tasks" / args.profile / "budget_curve" / split
    task_file = task_dir / "tasks.jsonl"
    manifest = read(task_dir / "manifest.json")
    tasks = read_jsonl(task_file)
    lookup = {t["query_id"]: t for t in tasks}
    if len(lookup) != len(tasks) or sha256(task_file) != manifest["task_sha256"]:
        raise ValueError("changed or duplicate budget tasks")
    if manifest["source_tasks_sha256"] != sha256(root / "tasks" / args.profile / f"evaluate_{split}.jsonl"):
        raise ValueError("primary task manifest changed")
    banks = {source: load_bank(root, source, split) for source in manifest["bank_sha256"]}
    for source, expected in manifest["bank_sha256"].items():
        if sha256(root / "banks" / f"{source}_{split}.npz") != expected:
            raise ValueError("budget curve bank changed")
    directories = {method: root / "evaluations" / args.profile / split / method / "seed_42" / args.run_id for method in METHODS}
    pending = [method for method, directory in directories.items() if not (directory / "summary.json").exists()]
    out = root / "reports" / args.profile / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if pending:
        result = {"status": "pending_evaluations", "pending_methods": pending, "requests_per_method": len(tasks),
                  "task_sha256": manifest["task_sha256"], "entire_blueprint_complete": False}
        write_json(out / "preflight.json", result)
        print(json.dumps(result))
        return 2
    records, sources = [], {}
    bank_ids = {name: {str(v): i for i, v in enumerate(bank.ids)} for name, bank in banks.items()}
    for method, directory in directories.items():
        summary, evaluation = read(directory / "summary.json"), read(directory / "manifest.json")
        rows = read_jsonl(directory / "predictions.jsonl")
        if (summary["status"] != "completed" or summary["requests"] != len(tasks) or len(rows) != len(tasks)
                or {r["query_id"] for r in rows} != lookup.keys() or evaluation["task_sha256"] != manifest["task_sha256"]
                or evaluation["config"] != config or evaluation["limit_tasks"] != 0
                or evaluation["beam_width"] != 0 or evaluation.get("no_smooth", False)
                or evaluation.get("planner", "beam") != "beam"):
            raise ValueError(f"incomplete or changed curve evaluation: {method}")
        for filename in ("summary.json", "manifest.json", "predictions.jsonl"):
            sources[str(directory / filename)] = sha256(directory / filename)
        if evaluation["checkpoint"] is not None:
            variant = "set" if method == "b_set" else method
            selected = root / "runs" / args.profile / variant / "seed_42/main"
            selection = read(selected / "selection.json")
            if (read(selected / "summary.json")["status"] != "completed" or selection != evaluation["checkpoint"]
                    or sha256(selected / "best.pt") != selection["checkpoint_sha256"] or selection["test_used"]):
                raise ValueError("curve checkpoint changed or is not validation-selected")
            sources[str(selected / "best.pt")] = selection["checkpoint_sha256"]
        for row in rows:
            task = lookup[row["query_id"]]
            bank = banks[task["source"]]
            if (row["method"] != method or row["seed"] != 42 or row["requested_budget"] != task["budget"]
                    or any(row[k] != task[k] for k in ("query_family_id", "source", "suite", "split", "num_tracks", "phrase_index"))
                    or row["library_size"] != len(bank.ids) or row["budget"] != min(task["budget"], len(bank.ids))
                    or row["budget_saturated"] != (task["budget"] >= len(bank.ids))):
                raise ValueError("request or budget changed")
            known = bank_ids[task["source"]]
            if (not set(row["candidate_ids"]) <= known.keys() or len(set(row["candidate_ids"])) != row["candidate_count"]
                    or not set(row["track_ids"]) <= set(row["candidate_ids"])):
                raise ValueError("invalid candidates or playlist membership")
            if row["status"] == "ok":
                ids = row["track_ids"]
                if (len(ids) != task["num_tracks"] or len(set(ids)) != len(ids) or ids[0] != row["start_id"]
                        or ids[-1] != row["target_id"] or len({bank.groups[known[i]] for i in ids}) != len(ids)):
                    raise ValueError("invalid playlist identity, length or endpoints")
            elif row["objective"] != 1.0:
                raise ValueError("failed request must keep the declared penalty")
        records.extend(rows)
    table, contrasts = aggregate(records)
    timing_file = root / "reports" / args.profile / args.latency_run_id / "summary.json"
    timing = None
    if timing_file.exists():
        measured = read(timing_file)
        protocol = measured["fingerprint"]
        if (measured["status"] != "completed" or measured["overlap_samples"] or protocol["main_run_id"] != args.run_id
                or protocol["source_tasks_sha256"] != manifest["task_sha256"] or set(protocol["methods"]) != set(METHODS)):
            raise ValueError("wrong or incomplete independent curve timing")
        for method, expected in protocol["main_predictions_sha256"].items():
            if sha256(directories[method] / "predictions.jsonl") != expected:
                raise ValueError("quality outputs changed after timing")
        measured_rows = read_jsonl(Path(measured["attempt"]) / "measurements.jsonl")
        timed_tasks = read_jsonl(Path(measured["attempt"]) / "tasks.jsonl")
        expected = Counter((method, task["query_id"], repetition) for method in METHODS for task in timed_tasks
                           for repetition in range(1, protocol["repeats"]+1))
        if (len(measured_rows) != measured["measurements"] or
                Counter((r["method"], r["query_id"], r["timing_repeat"]) for r in measured_rows) != expected):
            raise ValueError("unbalanced or incomplete curve timings")
        timing = {"summary": str(timing_file), "sha256": sha256(timing_file),
                  "measurements": measured["measurements"], "tables": measured["tables"],
                  "scope": measured["scope"], "quality_parity": measured["quality_parity"]}
    with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    result = {"status": "completed" if timing else "quality_complete_timing_pending", "profile": args.profile,
              "run_id": args.run_id, "requests_per_method": len(tasks), "outputs": len(records),
              "failed": sum(r["status"] != "ok" for r in records), "task_sha256": manifest["task_sha256"],
              "source_sha256": sha256(Path(__file__)), "input_files_sha256": sources, "tables": table,
              "paired_comparisons": contrasts, "independent_timing": timing, "saturation": manifest["saturation"],
              "limits": ["single fitted seed, held-out phrasing of the original request families, not new independent families",
                         "source-equal paired request-family bootstrap; no seed uncertainty or multiplicity adjustment",
                         "missing preference tags are not confirmed negatives; proxy and independent labels stay separate",
                         "saturated budgets are full-bank selection, not evidence of candidate reduction",
                         "direct-path/dynamic-reduction do not spend global B in the same way as candidate-set methods",
                         "development timing is retained separately; performance claims require completed isolated timing"],
              "entire_blueprint_complete": False}
    lines = ["# 固定请求的候选预算曲线", "", f"状态：{result['status']}；五种方法各 {len(tasks)} 请求，失败 {result['failed']}。", "",
             "请求 B=32/64/128/256，N=5/10；采用原有验证最优权重，不在本曲线上重新训练或选择模型。",
             "PMEmo 的 pilot 曲库为 116 首，请求 B=128 与 256 均为全库饱和；两档分别保留，不能作为缩减优势证据。", "",
             "| 对照 | 包 | N | 请求 B | 指标 | B−对照 | 请求族 95% 区间 |", "|---|---|---:|---:|---|---:|---|"]
    for row in contrasts:
        interval = row["ci95"]
        ci = f"[{interval[0]:.6f}, {interval[1]:.6f}]" if interval else "不可计算"
        lines.append(f"| {row['contrast']} | {row['suite']} | {row['num_tracks']} | {row['requested_budget']} | {row['metric']} | {row['mean']:.6f} | {ci} |")
    lines.extend(["", "逐来源的请求数、失败数、真实标签观察数及饱和状态见 results.csv；不能将多档预算当成独立请求。",
                  "独立计时与开发计时分开，详见 summary.json。正式种子与测试仍按原扩大门槛决定。"])
    (out / "预算曲线.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    write_json(out / "summary.json", result)
    print(json.dumps({"status": result["status"], "outputs": len(records), "rows": len(table), "contrasts": len(contrasts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
