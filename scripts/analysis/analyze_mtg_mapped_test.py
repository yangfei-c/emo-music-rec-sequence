"""Post-hoc reporting of final MTG test results; never used for model selection."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.training.records import write_json


SOURCE = ROOT / "artifacts/runs/mtg_mapped_final_test_20260914/evaluation/test_execution.json"
MAPPING = ROOT / "artifacts/runs/mtg_label_audit_20260914/evaluation/mapping.csv"
OUTPUT_JSON = ROOT / "artifacts/runs/mtg_mapped_final_test_20260914/evaluation/label_analysis.json"
OUTPUT_CSV = ROOT / "artifacts/runs/mtg_mapped_final_test_20260914/evaluation/label_analysis.csv"
METRICS = ("ndcg_at_10", "average_precision", "roc_auc", "precision_at_10", "recall_at_10")


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    result = json.loads(SOURCE.read_text(encoding="utf-8"))
    with MAPPING.open(encoding="utf-8-sig", newline="") as stream:
        mapping = {
            row["official_label_id"]: row
            for row in csv.DictReader(stream)
            if row["factor"] == "preference"
        }
    forms = ("canonical", "qwen_runtime")
    rows: list[dict] = []
    for form in forms:
        ours = result["methods"][f"ours_mapped_three_view/{form}"]["per_label_mean"]
        frozen = result["methods"][f"frozen_clamp3/{form}"]["per_label_mean"]
        for ours_row, base_row in zip(ours, frozen, strict=True):
            label = ours_row["label"]
            audit = mapping[label]
            row = {
                "query_form": form,
                "official_label_id": label,
                "display_name": audit["display_name"],
                "canonical_prompt": audit["canonical_prompt"],
                "mapping_changed": audit["stored_legacy_prompt"] != audit["canonical_prompt"],
                "issues": audit["issues"],
                "train_tracks": int(audit["train_tracks"]),
                "test_tracks": int(ours_row["positives"]),
            }
            for metric in METRICS:
                row[f"ours_{metric}"] = float(ours_row[metric])
                row[f"frozen_{metric}"] = float(base_row[metric])
                row[f"delta_{metric}"] = float(ours_row[metric] - base_row[metric])
            rows.append(row)

    group_summary = {}
    for form in forms:
        selected = [row for row in rows if row["query_form"] == form]
        group_summary[form] = {}
        for group_name, predicate in (
            ("all", lambda row: True),
            ("mapping_changed", lambda row: row["mapping_changed"]),
            ("mapping_unchanged", lambda row: not row["mapping_changed"]),
            ("lexical_expansion", lambda row: "identifier_needs_lexical_expansion" in row["issues"]),
            ("musical_context", lambda row: "prompt_needs_musical_context" in row["issues"]),
            ("semantic_ambiguity", lambda row: "semantically_ambiguous_without_context" in row["issues"]),
        ):
            subset = [row for row in selected if predicate(row)]
            group_summary[form][group_name] = {
                "labels": len(subset),
                **{
                    f"mean_delta_{metric}": mean([row[f"delta_{metric}"] for row in subset])
                    for metric in METRICS
                },
                "labels_improved_ndcg_at_10": sum(row["delta_ndcg_at_10"] > 0 for row in subset),

            }

    canonical = [row for row in rows if row["query_form"] == "canonical"]
    ranking = sorted(canonical, key=lambda row: row["delta_ndcg_at_10"], reverse=True)
    report = {
        "scope": "post-hoc final-test error analysis; not used to select or tune a model",
        "group_summary": group_summary,
        "largest_canonical_ndcg_gains": ranking[:15],
        "largest_canonical_ndcg_regressions": list(reversed(ranking[-15:])),
    }
    write_json(OUTPUT_JSON, report)
    fieldnames = list(rows[0])
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report["group_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
