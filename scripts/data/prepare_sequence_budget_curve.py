"""Freeze the blueprint's four-budget curve without changing primary tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_bank, load_config, read_jsonl, write_jsonl
from emomm_sequence.training.records import sha256, write_json

BUDGETS = (32, 64, 128, 256)
LENGTHS = (5, 10)


def build_tasks(tasks, profile, split):
    families = defaultdict(list)
    for task in tasks:
        if task["suite"] in ("e2", "e3") and task["phrase_index"] == 2:
            if task["split"] != split:
                raise ValueError("source task split mismatch")
            families[task["source"], task["query_family_id"]].append(task)
    if not families:
        raise ValueError("no held-out-phrase request families")
    output = []
    for key, variants in sorted(families.items()):
        base = min(variants, key=lambda t: (t["num_tracks"], t["budget"], t["query_id"]))
        varying = {"query_id", "num_tracks", "budget"}
        contract = {k: v for k, v in base.items() if k not in varying}
        if any({k: v for k, v in task.items() if k not in varying} != contract for task in variants):
            raise ValueError(f"a request family changes more than length and budget: {key}")
        for n in LENGTHS:
            for budget in BUDGETS:
                output.append({**base, "num_tracks": n, "budget": budget,
                               "origin_query_id": base["query_id"], "extension": "frozen_budget_curve",
                               "query_id": f"{profile}_{split}_budget_curve_{len(output):06d}"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    args = parser.parse_args()
    _, root = load_config(args.config)
    split = "validation" if args.profile == "pilot" else "test"
    source = root / "tasks" / args.profile / f"evaluate_{split}.jsonl"
    tasks = build_tasks(read_jsonl(source), args.profile, split)
    banks = {name: load_bank(root, name, split) for name in {t["source"] for t in tasks}}
    saturation = []
    for name, bank in sorted(banks.items()):
        if len(bank.ids) < max(LENGTHS) or len(set(bank.groups.tolist())) != len(bank.ids):
            raise ValueError("bank too small or duplicate contents")
        saturation.append({"source": name, "library_size": len(bank.ids),
                           "effective_budgets": {str(b): min(b, len(bank.ids)) for b in BUDGETS},
                           "saturated_requested_budgets": [b for b in BUDGETS if b >= len(bank.ids)]})
    directory = root / "tasks" / args.profile / "budget_curve" / split
    path = directory / "tasks.jsonl"
    if path.exists():
        if read_jsonl(path) != tasks:
            raise ValueError("budget tasks changed; preserve and version the old directory")
    else:
        write_jsonl(path, tasks)
    manifest = {"profile": args.profile, "split": split, "requests": len(tasks),
                "source_tasks_sha256": sha256(source), "task_sha256": sha256(path),
                "generator_sha256": sha256(Path(__file__)), "budgets": list(BUDGETS), "lengths": list(LENGTHS),
                "phrase_index": 2, "selection": "all existing pilot/full E2 and E3 families, no outcome-based selection",
                "bank_sha256": {name: sha256(root / "banks" / f"{name}_{split}.npz") for name in sorted(banks)},
                "saturation": saturation, "scope": "Frozen objective; original checkpoints, no retraining or selection on this curve"}
    manifest_file = directory / "manifest.json"
    if manifest_file.exists() and json.loads(manifest_file.read_text(encoding="utf-8")) != manifest:
        raise ValueError("budget manifest changed; preserve and version it")
    write_json(manifest_file, manifest)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
