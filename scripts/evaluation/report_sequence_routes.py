"""Aggregate only compatible runs; retain failures and paired query families."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import read_jsonl
from emomm_sequence.training.records import write_json


def mean_present(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None


def family_bootstrap(per_source, repeats=2000):
    """Keep a shared emotion request correlated across music sources."""
    rng = np.random.default_rng(20260929)
    sets = [set(values) for values in per_source.values()]
    mean = float(np.mean([np.mean(list(values.values())) for values in per_source.values()]))
    same = all(s == sets[0] for s in sets)
    disjoint = len(set.union(*sets)) == sum(map(len, sets))
    if same:
        families = sorted(sets[0])
        matrix = np.array([[values[f] for f in families] for values in per_source.values()])
        boot = np.array([matrix[:, rng.integers(len(families), size=len(families))].mean() for _ in range(repeats)])
        clusters = len(families)
        scheme = "same_family_draw_shared_across_sources"
    elif disjoint:
        vectors = [list(values.values()) for values in per_source.values()]
        boot = np.array([np.mean([np.mean(rng.choice(v, len(v), replace=True)) for v in vectors]) for _ in range(repeats)])
        clusters = sum(map(len, vectors))
        scheme = "source_stratified_distinct_families"
    else:
        return {"mean": mean, "ci95": None, "query_family_clusters": len(set.union(*sets)),
                "bootstrap_scheme": "incomplete_overlapping_families_descriptive_only"}
    return {"mean": mean, "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "query_family_clusters": clusters, "bootstrap_scheme": scheme}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--run-id", default="main")
    args = p.parse_args()
    groups, paired, run_states = defaultdict(list), defaultdict(dict), []
    for path in (args.root / "evaluations" / args.profile).glob(f"*/*/seed_*/{args.run_id}/predictions.jsonl"):
        manifest = json.loads((path.parent / "manifest.json").read_text(encoding="utf-8"))
        complete = (path.parent / "summary.json").exists()
        run_states.append({"path": str(path.parent), "completed": complete})
        for row in read_jsonl(path):
            key = (row["split"], row["suite"], row["source"], row["method"], row["num_tracks"], row["budget"])
            groups[key].append(row)
            pairkey = (row["split"], manifest["task_sha256"], row["query_id"])
            paired[pairkey].setdefault(row["method"], []).append(row)
    columns = ["split", "suite", "source", "method", "num_tracks", "budget", "rows", "successful", "failure_rate",
               "objective", "candidate_gap", "solver_gap", "true_va_mae", "true_endpoint_mae", "true_step_deviation",
               "observed_preference_tag_match", "p50_seconds", "p95_seconds", "peak_cuda_mib"]
    table = []
    for key, rows in sorted(groups.items()):
        row = dict(zip(columns[:6], key))
        row.update(rows=len(rows), successful=sum(r["status"] == "ok" for r in rows),
                   failure_rate=float(np.mean([r["status"] != "ok" for r in rows])))
        for field in columns[9:16]:
            row[field] = mean_present(rows, field)
        row["p50_seconds"], row["p95_seconds"] = [float(x) for x in np.quantile([r["total_seconds"] for r in rows], [.5, .95])]
        row["peak_cuda_mib"] = max(r["peak_cuda_mib"] for r in rows)
        table.append(row)
    comparisons = []
    for reference in ("a_grid", "a_refine", "direct_path", "dynamic_reduce"):
        clusters = defaultdict(list)
        for key, methods in paired.items():
            if "b_set" not in methods or reference not in methods:
                continue
            b, a = methods["b_set"], methods[reference]
            family = (key[0], b[0]["source"], b[0]["suite"], b[0]["query_family_id"])
            clusters[family].append(float(np.mean([r["objective"] for r in b]) - np.mean([r["objective"] for r in a])))
        for split in ("validation", "test", "all"):
            for suite in ("e1", "e2", "e3"):
                per_source = defaultdict(dict)
                for (sp, source, su, family), vals in clusters.items():
                    if sp == split and su == suite:
                        per_source[source][family] = float(np.mean(vals))
                if not per_source:
                    continue
                comparisons.append({"split": split, "suite": suite, "contrast": f"b_set_minus_{reference}",
                                    **family_bootstrap(per_source),
                                    "interpretation": "negative favors B; request uncertainty conditional on this music library"})
        for metric in ("true_va_mae", "true_endpoint_mae", "true_step_deviation", "observed_preference_tag_match"):
            independent = defaultdict(list)
            for key, methods in paired.items():
                if "b_set" not in methods or reference not in methods:
                    continue
                b, a = methods["b_set"], methods[reference]
                vb, va = mean_present(b, metric), mean_present(a, metric)
                if vb is None or va is None:
                    continue
                family = (key[0], b[0]["source"], b[0]["suite"], b[0]["query_family_id"])
                independent[family].append(vb - va)
            for split in ("validation", "test", "all"):
                for suite in ("e2", "e3"):
                    per_source = defaultdict(dict)
                    for (sp, source, su, family), values in independent.items():
                        if sp == split and su == suite:
                            per_source[source][family] = float(np.mean(values))
                    if per_source:
                        comparisons.append({"split": split, "suite": suite, "contrast": f"b_set_minus_{reference}",
                                            "metric": metric, **family_bootstrap(per_source),
                                            "interpretation": "positive favors B" if metric == "observed_preference_tag_match" else "negative favors B"})
    for comparison in comparisons:
        comparison.setdefault("metric", "objective")
        comparison["interval_scope"] = "requests conditional on observed trained seeds and fixed song bank; not seed uncertainty"
    out = args.root / "reports" / args.profile
    if args.run_id != "main":
        out = out / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(table)
    summary = {"profile": args.profile, "run_id": args.run_id, "runs": run_states, "tables": table,
               "paired_comparisons": comparisons, "all_observed_runs_completed": bool(run_states) and all(x["completed"] for x in run_states),
               "entire_blueprint_complete": False,
               "limits": ["surrogate objective is not independent emotion evidence", "historical test exposure",
                          "paper scores on other datasets are not head-to-head scores", "missing planned runs remain incomplete"]}
    write_json(out / "summary.json", summary)
    lines = [f"# 双路线实验结果：{args.profile}", "", "## Material Passport", "", "- Origin Mode: run/aggregate",
             "- Verification Status: 已读取逐请求输出；不等于整个蓝图完成", "", f"已有输出 {len(run_states)} 个运行目录，汇总 {len(table)} 行分组结果。",
             "", "## 评价边界", "", "优化目标是冻结模型上的代理代价；真实 VA 与偏好标签另列。训练 checkpoint 只由验证集选择。",
             "", "## 配对比较", "", "B−对照小于零表示 B 的共同目标更低；区间按请求族聚类、来源等权。", "",
             "| 划分 | 包 | 指标 | 对照 | 平均差 | 95% 区间 | 请求族数 |", "|---|---|---|---|---:|---|---:|"]
    for c in comparisons:
        ci = f"[{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}]" if c["ci95"] else "请求族未完整对齐，暂不计算"
        lines.append(f"| {c['split']} | {c['suite']} | {c['metric']} | {c['contrast']} | {c['mean']:.6f} | {ci} | {c['query_family_clusters']} |")
    lines.extend(["", "完整分组指标见 results.csv，逐请求歌曲、失败及时间见 evaluations。尚未出现的计划实验不视为已完成。",
                  "", "## 最新论文对比", "", "同任务复现方法的结果可作直接对照；论文原表的不同数据集/指标数值仅作背景。文献核验另见 reports/literature_comparison.md（生成后）。"])
    (out / "实验结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "runs": len(run_states), "rows": len(table)}))


if __name__ == "__main__":
    main()
