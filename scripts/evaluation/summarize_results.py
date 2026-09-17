"""Create source-linked reporting numbers without rounding intermediate results."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import write_json


def stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "seeds": [42, 43, 44], "values": [float(v) for v in values]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output = ROOT / "results" / args.run_id
    comparison = json.loads((output / "retrieval_comparison.json").read_text(encoding="utf-8"))
    train_root = ROOT / "artifacts/runs" / args.run_id
    retriever = json.loads((train_root / "retriever/aggregate.json").read_text(encoding="utf-8"))
    relation = json.loads((train_root / "music_va/aggregate.json").read_text(encoding="utf-8"))
    text = json.loads((train_root / "text_va/aggregate.json").read_text(encoding="utf-8"))
    base = comparison["baseline"]["results"]
    ours = [r["results"] for r in comparison["ours"]["runs"]]
    rows = []
    for source in ("merge", "song_describer"):
        for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
            baseline = base["semantic_pairs"][source]["text_to_music"][metric]
            values = [r["semantic_pairs"][source]["text_to_music"][metric] for r in ours]
            rows.append({"task": "text_to_music", "dataset": source, "metric": metric, "clamp3": baseline, "ours": stats(values), "delta": float(np.mean(values) - baseline)})
    for factor in ("preference", "affect"):
        for metric in ("precision_at_5", "precision_at_10", "ndcg_at_10", "average_precision"):
            baseline = base[f"{factor}_tags"]["macro"][metric]
            values = [r[f"{factor}_tags"]["macro"][metric] for r in ours]
            rows.append({"task": factor, "dataset": "mtg_jamendo", "metric": metric, "clamp3": baseline, "ours": stats(values), "delta": float(np.mean(values) - baseline)})
    va = {source: {metric: stats([r["results"]["test"]["va"][source][metric] for r in relation["runs"]]) for metric in ("valence_ccc", "arousal_ccc", "valence_rmse", "arousal_rmse", "mean_rmse")} for source in ("deam", "pmemo")}
    relations = {metric: stats([r["results"]["test"]["relations"]["overall"][metric] for r in relation["runs"]]) for metric in ("accuracy_v", "accuracy_a", "macro_f1_v", "macro_f1_a", "nll_v", "nll_a")}
    text_va = {metric: stats([r["results"]["test"][metric] for r in text["runs"]]) for metric in ("valence_ccc", "arousal_ccc", "valence_rmse", "arousal_rmse")}
    audit = json.loads((ROOT / "artifacts/features/clamp3/clean/content_split_report.json").read_text(encoding="utf-8"))
    data_counts = {Path(item["output"]).stem: item["split_counts"] for item in audit["files"]}
    summary = {"run_id": args.run_id, "selected_retriever_seed": comparison["selected_seed"], "retrieval": rows, "music_va": va, "music_relations": relations, "independent_text_va": text_va,
        "best_epochs": {"retriever": {str(r["seed"]): r["best_epoch"] for r in retriever["runs"]}, "music_va": {str(r["seed"]): r["best_epoch"] for r in relation["runs"]}},
        "data_counts": data_counts, "protocol": comparison["protocol"], "relation_test_pairs": relation["runs"][0]["results"]["test"]["relations"]["overall"]["count"],
        "interpretation": "Interpret signed differences in the metric rows. No evidence of human emotion regulation efficacy.",
        "source_files": [f"results/{args.run_id}/retrieval_comparison.json", f"artifacts/runs/{args.run_id}/music_va/aggregate.json", f"artifacts/runs/{args.run_id}/retriever/aggregate.json", "artifacts/features/clamp3/clean/content_split_report.json"]}
    write_json(output / "summary.json", summary)
    with (output / "retrieval_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "dataset", "metric", "CLaMP3", "Ours_mean", "Ours_std", "difference"])
        writer.writeheader()
        writer.writerows({"task": r["task"], "dataset": r["dataset"], "metric": r["metric"], "CLaMP3": r["clamp3"], "Ours_mean": r["ours"]["mean"], "Ours_std": r["ours"]["std"], "difference": r["delta"]} for r in rows)
    print(json.dumps({"main_retrieval": [r for r in rows if r["metric"] in {"recall_at_10", "ndcg_at_10"}], "music_va": va, "text_va": text_va, "best_epochs": summary["best_epochs"]}, indent=2))


if __name__ == "__main__":
    main()
