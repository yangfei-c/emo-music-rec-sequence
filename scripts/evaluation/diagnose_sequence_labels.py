"""Explain completed paths using evaluator-only label oracles and validity counts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_bank, read_jsonl, write_jsonl
from emomm_sequence.data.sequence_va_tasks import VAInputs, VA_ENDPOINT_PROTOCOL
from emomm_sequence.evaluation.sequence_label_oracle import positional_va_oracles
from emomm_sequence.training.records import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    args = p.parse_args()
    base = args.root / "evaluations" / args.profile
    files = sorted([*base.glob("*/*/seed_*/main/predictions.jsonl"),
                    *(base / "va_retrained").glob("*/*/*/seed_*/main/predictions.jsonl")])
    files = [f for f in files if (f.parent / "summary.json").exists()]
    tasks, banks, labels, oracle_cache = {}, {}, {}, {}
    va_inputs = None
    records, valid, provenance, completed_runs = [], defaultdict(list), {}, 0
    for file in files:
        manifest = json.loads((file.parent / "manifest.json").read_text(encoding="utf-8"))
        # Generalization manifests have their own reports; this diagnostic is for main tasks.
        if manifest.get("custom_task_file"):
            continue
        is_va = "va_retrained" in file.parts
        if is_va and manifest.get("va_endpoint_protocol") != VA_ENDPOINT_PROTOCOL:
            raise ValueError("retired VA protocol cannot enter main diagnostics")
        provenance[str(file)] = sha256(file)
        completed_runs += 1
        for row in read_jsonl(file):
            split = row["split"]
            if split not in tasks:
                path = args.root / "tasks" / args.profile / f"evaluate_{split}.jsonl"
                tasks[split] = {t["query_id"]: t for t in read_jsonl(path)}
                provenance[str(path)] = sha256(path)
            if manifest["task_sha256"] != provenance[str(args.root / "tasks" / args.profile / f"evaluate_{split}.jsonl")]:
                raise ValueError("task manifest changed after evaluation")
            task = tasks[split][row["query_id"]]
            key = (row["source"], split)
            if key not in banks:
                banks[key] = load_bank(args.root, *key)
                bank_file = args.root / "banks" / f"{key[0]}_{key[1]}.npz"
                provenance[str(bank_file)] = sha256(bank_file)
            actual_task = task
            if is_va:
                if va_inputs is None:
                    va_inputs = VAInputs(args.root, "point")
                actual_task = va_inputs.eligible_task(task)
            bank = banks[key].subset(actual_task.get("indices"))
            lookup = {str(i): j for j, i in enumerate(bank.ids)}
            scope = "va" if is_va else "frozen"
            mode = row.get("mode", "frozen")
            ids = row["track_ids"]
            known = all(i in lookup for i in ids)
            full = len(ids) == task["num_tracks"]
            unique = known and len({str(bank.groups[lookup[i]]) for i in ids}) == len(ids)
            anchors = bool(ids) and (ids[0], ids[-1]) == (row["start_id"], row["target_id"])
            legal = row["status"] == "ok" and known and full and unique and anchors
            valid[(scope, split, mode, row["method"], row["suite"])].append(
                {"success": row["status"] == "ok", "known": known, "full": full, "unique": unique, "anchors": anchors, "legal": legal})
            if row["status"] == "ok" and not legal:
                raise AssertionError(f"successful playlist violates output contract: {file}/{row['query_id']}")
            if row["suite"] != "e2" or not legal:
                continue
            if key not in labels:
                path = args.root / "labels" / f"{key[0]}_{key[1]}.npz"
                with np.load(path, allow_pickle=False) as z:
                    labels[key] = (z["va"], {str(v): i for i, v in enumerate(z["track_ids"])})
                provenance[str(path)] = sha256(path)
            values, label_lookup = labels[key]
            va = values[[label_lookup[str(i)] for i in bank.ids]]
            targets = np.linspace(task["start_va"], task["target_va"], task["num_tracks"])
            start, target = lookup[row["start_id"]], lookup[row["target_id"]]
            okey = (key, tuple(bank.ids), tuple(task["start_va"]), tuple(task["target_va"]), len(targets), start, target)
            if okey not in oracle_cache:
                oracle_cache[okey] = positional_va_oracles(va, targets, start, target)
            oracle = oracle_cache[okey]
            actual = float(np.abs(va[[lookup[i] for i in ids]] - targets).mean())
            if abs(actual - row["true_va_mae"]) > 1e-7:
                raise AssertionError("stored true-VA metric cannot be reproduced")
            interior_gap = actual - oracle["fixed_endpoint_min_va_mae"]
            if interior_gap < -1e-7:
                raise AssertionError("observed path beats exact positional-label oracle")
            records.append({"scope": scope, "mode": mode, "split": split, "method": row["method"], "seed": row["seed"],
                            "query_id": row["query_id"], "query_family_id": row["query_family_id"], "source": row["source"],
                            "num_tracks": task["num_tracks"], "budget": row["budget"], "phrase_index": task["phrase_index"],
                            "held_out_policy_wording": task["phrase_index"] == 2, "true_va_mae": actual,
                            "track_ids": ids, "true_va_values": va[[lookup[i] for i in ids]].tolist(),
                            "target_va_path": targets.tolist(),
                            "free_endpoint_min_va_mae": oracle["free_endpoint_min_va_mae"],
                            "fixed_endpoint_min_va_mae": oracle["fixed_endpoint_min_va_mae"],
                            "fixed_endpoint_va_penalty": oracle["fixed_endpoint_va_penalty"],
                            "interior_selection_va_gap": interior_gap,
                            "true_reverse_fraction": row.get("true_reverse_fraction"),
                            "oracle_certified_for": oracle["certified_for"]})
    validity = [{**dict(zip(("scope", "split", "mode", "method", "suite"), key)), "requests": len(rows),
                 **{f"{field}_fraction": float(np.mean([r[field] for r in rows])) for field in rows[0]}}
                for key, rows in sorted(valid.items())]
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in ("scope", "split", "mode", "method", "source", "num_tracks", "held_out_policy_wording"))].append(row)
    metrics = ("true_va_mae", "free_endpoint_min_va_mae", "fixed_endpoint_va_penalty", "interior_selection_va_gap", "true_reverse_fraction")
    table = []
    for key, rows in sorted(groups.items()):
        item = dict(zip(("scope", "split", "mode", "method", "source", "num_tracks", "held_out_policy_wording"), key))
        item["rows"] = len(rows)
        for metric in metrics:
            values = [r[metric] for r in rows if r[metric] is not None]
            item[metric] = float(np.mean(values)) if values else None
        table.append(item)
    out = args.root / "reports" / args.profile / "label_diagnostics"
    write_jsonl(out / "per_request.jsonl", records)
    paired = defaultdict(dict)
    for row in records:
        if row["scope"] == "frozen":
            paired[(row["split"], row["query_id"], row["seed"])][row["method"]] = row
    comparisons = []
    for methods in paired.values():
        if "b_set" in methods and "a_refine" in methods:
            b, a = methods["b_set"], methods["a_refine"]
            comparisons.append({"b": b, "a": a, "difference": b["true_va_mae"] - a["true_va_mae"]})
    examples = []
    for direction, reverse in (("largest_B_deterioration", True), ("largest_B_improvement", False)):
        seen = set()
        for item in sorted(comparisons, key=lambda r: r["difference"], reverse=reverse):
            family = (item["b"]["split"], item["b"]["source"], item["b"]["query_family_id"])
            if family in seen or (item["difference"] <= 0 if reverse else item["difference"] >= 0):
                continue
            seen.add(family)
            examples.append({"selection": direction, "B_minus_A_refine_true_va_mae": item["difference"],
                             "B": item["b"], "A_refine": item["a"]})
            if len(seen) == 3:
                break
    write_json(out / "case_studies.json", {"selection_rule": "posthoc largest and smallest paired VA gaps; at most one per source/request family",
               "not_independent_confirmatory_evidence": True, "examples": examples})
    write_json(out / "summary.json", {"completed_input_runs": completed_runs, "oracle_instances": len(oracle_cache),
               "implementation_sha256": {str(path): sha256(path) for path in (Path(__file__), Path(__file__).resolve().parents[2] / "src/emomm_sequence/evaluation/sequence_label_oracle.py")},
               "input_hashes": provenance, "validity": validity, "tables": table,
               "decomposition": "actual_VA_MAE = free_endpoint_min + fixed_endpoint_penalty + interior_selection_gap",
               "boundary": "exact references optimize ONLY positional VA MAE; exclude preference/step costs; evaluation labels never enter policies",
               "wording_boundary": "phrase 2 held out from candidate policy training, not from CLaMP3 pretraining"})
    print(json.dumps({"output": str(out), "rows": len(records), "oracle_instances": len(oracle_cache)}))


if __name__ == "__main__":
    main()
