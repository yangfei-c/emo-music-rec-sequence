"""Paired A-VA and separately retrained B-VA using policy-only validation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_config, load_bank, read_jsonl
from emomm_sequence.data.sequence_va_tasks import VAInputs, VA_ENDPOINT_PROTOCOL, va_candidates, va_refine
from emomm_sequence.inference.sequence_va import va_beam_search
from emomm_sequence.evaluation.sequence_benchmark import label_metrics
from emomm_sequence.training.sequence_selector import load_model, process_identity
from emomm_sequence.training.records import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--mode", choices=["point", "raw_interval", "cqr_interval"], required=True)
    p.add_argument("--methods", nargs="+", choices=["a_grid", "a_refine", "b_set"], default=["a_grid", "a_refine", "b_set"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--model-run-id")
    p.add_argument("--run-id", default="main")
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    config, root = load_config(args.config)
    inputs = VAInputs(root, args.mode)
    task_file = root / "tasks" / args.profile / f"evaluate_{args.split}.jsonl"
    tasks = [t for t in read_jsonl(task_file) if t["suite"] in ("e2", "e3")]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    with np.load(root / "cache" / "text_bank.npz", allow_pickle=False) as z:
        texts = {k: z[k] for k in z.files}
    banks = {s: load_bank(root, s, args.split) for s in {t["source"] for t in tasks}}
    label_cache = {}
    identity = process_identity()
    for method in args.methods:
        model, selection = (None, None)
        if method == "b_set":
            run_id = args.model_run_id or f"va_{args.mode}"
            model, selection = load_model(config, root, args.profile, "set", args.seed, args.device, run_id)
            if model.project[0].in_features != config["assets"]["embedding_dim"] + 7:
                raise ValueError("B-VA requires a model trained with the VA features and objective")
            manifest_path = root / "runs" / args.profile / "set" / f"seed_{args.seed}" / run_id / "manifest.json"
            training_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if training_manifest.get("objective") != args.mode:
                raise ValueError("wrong VA training objective")
            if training_manifest.get("va_endpoint_protocol") != VA_ENDPOINT_PROTOCOL:
                raise ValueError("B-VA checkpoint used a different endpoint protocol; retrain under the frozen anchors")
        directory = root / "evaluations" / args.profile / "va_retrained" / args.split / args.mode / method / f"seed_{args.seed}" / args.run_id
        directory.mkdir(parents=True, exist_ok=True)
        manifest = {"task_sha256": sha256(task_file), "mode": args.mode, "method": method, "seed": args.seed,
                    "limit_tasks": args.limit_tasks, "checkpoint": selection,
                    "va_manifest_sha256": sha256(inputs.directory / "manifest.json"),
                    "va_endpoint_protocol": VA_ENDPOINT_PROTOCOL,
                    "implementation_sha256": {str(file.relative_to(Path(__file__).resolve().parents[2])): sha256(file)
                       for file in (Path(__file__).resolve(),
                                    Path(__file__).resolve().parents[2] / "src/emomm_sequence/data/sequence_va_tasks.py",
                                    Path(__file__).resolve().parents[2] / "src/emomm_sequence/inference/sequence_va.py")},
                    "validation_partition": "existing_policy_only_for_deam_pmemo", "config": config}
        mf = directory / "manifest.json"
        if mf.exists() and json.loads(mf.read_text(encoding="utf-8")) != manifest:
            raise ValueError("VA evaluation inputs changed; use a fresh run ID")
        write_json(mf, manifest)
        output = directory / "predictions.jsonl"
        rows = read_jsonl(output) if output.exists() else []
        done = {r["query_id"] for r in rows}
        def progress(status, **extra):
            write_json(directory / "progress.json", {**identity, "status": status, "scope": args.profile,
                       "updated_utc": datetime.now(timezone.utc).isoformat(), "rows": len(rows), "total": len(tasks), **extra})
        progress("running")
        try:
            for task in tasks:
                if task["query_id"] in done:
                    continue
                bank, problem, actual_task = inputs.make(banks[task["source"]], task, texts)
                begin = time.perf_counter()
                budget = min(task["budget"], problem.m)
                if model is not None:
                    with torch.inference_mode():
                        out = model.select(torch.from_numpy(problem.x).to(args.device), torch.from_numpy(problem.query).to(args.device),
                                           problem.n, budget, problem.start, problem.target)
                    candidates = np.asarray(out.indices)
                else:
                    candidates = va_candidates(problem, budget)
                if method == "a_refine":
                    candidates, result = va_refine(problem, candidates, config["planner"]["beam_width"])
                else:
                    result = va_beam_search(problem, candidates, config["planner"]["beam_width"])
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                row = {"query_id": task["query_id"], "query_family_id": task["query_family_id"], "source": task["source"],
                       "suite": task["suite"], "method": method, "mode": args.mode, "seed": args.seed, "split": args.split,
                       "num_tracks": problem.n, "budget": budget, "library_size": problem.m,
                       "objective": result.objective, "status": result.status, "inference_seconds": time.perf_counter() - begin,
                       "track_ids": bank.ids[list(result.path)].tolist(), "candidate_ids": bank.ids[candidates].tolist(),
                       "start_id": str(bank.ids[problem.start]), "target_id": str(bank.ids[problem.target]),
                       "va_endpoint_protocol": VA_ENDPOINT_PROTOCOL,
                       "policy_retrained_on_va": method == "b_set", "playlist_coverage_guarantee": False}
                if result.status == "ok":
                    row.update(label_metrics(root, bank, actual_task, result.path, label_cache))
                    row["point_va_objective"] = problem.point_cost(result.path)
                    if "va" in label_cache[(bank.source, bank.split)]:
                        labels = label_cache[(bank.source, bank.split)]
                        y = labels["va"][[labels["lookup"][str(bank.ids[i])] for i in result.path]]
                        lo, hi = problem.lower[list(result.path)], problem.upper[list(result.path)]
                        covered = ((y >= lo) & (y <= hi)).all(axis=1)
                        row.update(selected_song_joint_coverage=float(covered.mean()), whole_playlist_covered=bool(covered.all()),
                                   selected_interval_width=float((hi - lo).mean()), calibration_source_known=bank.source in ("deam", "pmemo"))
                with output.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                rows.append(row)
                progress("running")
            write_json(directory / "summary.json", {"status": "completed", "rows": len(rows), "failed": sum(r["status"] != "ok" for r in rows),
                                                      "mode": args.mode, "method": method, "selection_used_test": False})
            progress("completed", expected_outputs=[str(output), str(directory / "summary.json")])
        except Exception as exc:
            progress("failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
