"""Paired inference benchmarks; true song labels are loaded only here."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch

from emomm_sequence.data.sequence_tasks import ROOT, load_config, load_bank, read_jsonl, make_task_problem
from emomm_sequence.inference.sequence_planner import Plan, beam_search, exact_search
from emomm_sequence.inference.trajectory_candidates import select_candidates, independent_path, refine
from emomm_sequence.training.sequence_selector import load_model, process_identity
from emomm_sequence.training.records import write_json, sha256, environment

METHODS = ("random", "three_route", "independent", "a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce")


def label_metrics(root, bank, task, path, cache):
    key = (bank.source, bank.split)
    if key not in cache:
        with np.load(root / "labels" / f"{bank.source}_{bank.split}.npz", allow_pickle=False) as z:
            cache[key] = {k: z[k] for k in z.files}
        cache[key]["lookup"] = {str(i): j for j, i in enumerate(cache[key]["track_ids"])}
    labels = cache[key]
    ix = [labels["lookup"][str(bank.ids[i])] for i in path]
    result = {}
    if "va" in labels:
        y = labels["va"][ix]
        start, target = np.asarray(task["start_va"]), np.asarray(task["target_va"])
        target_path = np.linspace(start, target, len(path))
        step = (target - start) / max(len(path) - 1, 1)
        active = np.abs(step) > 1e-8
        result.update(true_va_mae=float(np.abs(y - target_path).mean()),
                      true_endpoint_mae=float(np.abs(y[[0, -1]] - target_path[[0, -1]]).mean()),
                      true_step_deviation=float(np.abs(np.diff(y, axis=0) - step).mean()) if len(path) > 1 else 0.0,
                      true_reverse_fraction=float((np.diff(y, axis=0)[:, active] * np.sign(step[active]) < -1e-6).mean()) if active.any() and len(path) > 1 else None)
    pref = task.get("preference_label_index")
    if "preference_targets" in labels and pref is not None:
        result["observed_preference_tag_match"] = float(labels["preference_targets"][ix, pref].mean())
    return result


def evaluate(config, root, args):
    if getattr(args, "planner", "beam") == "exact" and any(m in args.methods for m in ("a_refine", "independent", "direct_path", "dynamic_reduce")):
        raise ValueError("the exact-planner ablation supports random/three_route/a_grid/b_set; other methods have their own decoding")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    tasks_file = Path(args.tasks_file) if getattr(args, "tasks_file", None) else root / "tasks" / args.profile / f"evaluate_{args.split}.jsonl"
    tasks = read_jsonl(tasks_file)
    if args.suites:
        tasks = [t for t in tasks if t["suite"] in args.suites]
    if args.limit_tasks:
        tasks = tasks[:args.limit_tasks]
    if not tasks:
        raise ValueError("empty evaluation request list")
    with np.load(root / "cache" / "text_bank.npz", allow_pickle=False) as z:
        text = {k: z[k] for k in z.files}
    banks, labels = {}, {}
    problem_cache = {}
    identity = process_identity()
    seeds = args.seeds or [args.seed]
    task_hash = sha256(tasks_file)
    for method in args.methods:
        actual_seeds = seeds if method in ("random", "b_set", "direct_path", "dynamic_reduce") else [seeds[0]]
        for seed in actual_seeds:
            model = None
            variant = "set" if method == "b_set" else method
            selection = None
            if method in ("b_set", "direct_path", "dynamic_reduce"):
                model, selection = load_model(config, root, args.profile, variant, seed, device, args.model_run_id)
            outdir = root / "evaluations" / args.profile / args.split / method / f"seed_{seed}" / args.run_id
            outdir.mkdir(parents=True, exist_ok=True)
            manifest = {"task_sha256": task_hash, "text_sha256": sha256(root / "cache" / "text_bank.npz"),
                        "config": config, "method": method, "seed": seed, "suite_filter": args.suites,
                        "limit_tasks": args.limit_tasks, "checkpoint": selection, "beam_width": args.beam_width}
            if getattr(args, "tasks_file", None):
                manifest["custom_task_file"] = str(tasks_file.resolve())
            if getattr(args, "no_smooth", False):
                manifest["no_smooth"] = True
            if getattr(args, "planner", "beam") != "beam":
                manifest["planner"] = args.planner
            if args.profile == "full":
                manifest["code_hashes"] = {relative: sha256(ROOT / relative) for relative in (
                    "src/emomm_sequence/evaluation/sequence_benchmark.py", "src/emomm_sequence/losses/sequence_cost.py",
                    "src/emomm_sequence/inference/sequence_planner.py", "src/emomm_sequence/inference/trajectory_candidates.py",
                    "src/emomm_sequence/models/sequence_selector.py", "src/emomm_sequence/models/sequence_reduction.py")}
            mf = outdir / "manifest.json"
            if mf.exists() and json.loads(mf.read_text(encoding="utf-8")) != manifest:
                raise ValueError("evaluation manifest changed; use a new --run-id")
            write_json(mf, manifest)
            write_json(outdir / "environment.json", environment())
            records_file = outdir / "predictions.jsonl"
            records = read_jsonl(records_file) if records_file.exists() else []
            done = {r["query_id"] for r in records}
            if len(done) != len(records):
                raise ValueError("duplicate stored evaluation IDs")
            begin_run = time.monotonic()
            def progress(status, **extra):
                write_json(outdir / "progress.json", {**identity, "status": status, "scope": args.profile,
                           "method": method, "seed": seed, "completed_requests": len(records), "total_requests": len(tasks),
                           "updated_utc": datetime.now(timezone.utc).isoformat(), **extra})
            progress("running")
            try:
                for task in tasks:
                    if task["query_id"] in done:
                        continue
                    key = (task["source"], task["split"])
                    if key not in banks:
                        banks[key] = load_bank(root, *key)
                    pkey = json.dumps({k: task[k] for k in ("source", "split", "indices", "start_text_id", "target_text_id", "preference_text_id", "num_tracks")}, sort_keys=True)
                    if pkey not in problem_cache:
                        before = time.perf_counter()
                        bank, problem = make_task_problem(banks[key], task, text)
                        if getattr(args, "no_smooth", False):
                            # Keep anchors fixed; remove the smooth term and renormalize node weights.
                            node_mass = 0.75 if task["preference_text_id"] is not None else 2 / 3
                            problem.nodes /= node_mass
                            problem.edge_weight = 0.0
                        problem_cache[pkey] = (bank, problem, time.perf_counter() - before)
                        # Large-bank objectives are cheap to recreate; cap retained memory.
                        if len(problem_cache) > 40:
                            problem_cache.pop(next(iter(problem_cache)))
                    bank, problem, setup_seconds = problem_cache[pkey]
                    budget = min(task["budget"], problem.m)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    begin = time.perf_counter()
                    stats = {"full_scans": problem.n, "solver_calls": 1}
                    width = args.beam_width or config["planner"]["beam_width"]
                    def search(candidates):
                        if getattr(args, "planner", "beam") == "exact":
                            return exact_search(problem, candidates)
                        return beam_search(problem, candidates, width=width)
                    if model is not None:
                        before_select = time.perf_counter()
                        with torch.no_grad():
                            out = model.select(torch.from_numpy(problem.x).to(device), torch.from_numpy(problem.query).to(device),
                                               problem.n, budget, problem.start, problem.target)
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        candidates = np.asarray(out.indices)
                        candidate_seconds = time.perf_counter() - before_select
                        stats["full_scans"] += (budget - 2) if variant == "set" else (problem.n - 2)
                        if variant == "dynamic_reduce":
                            stats["cumulative_candidates"] = len(out.proposed_indices)
                            stats["local_scored_occurrences"] = out.local_scored_occurrences
                            stats["dynamic_k"] = out.dynamic_k
                            stats["budget_scope"] = f"per_step_top{out.dynamic_k}_not_global_B"
                        if variant == "set":
                            result = search(candidates)
                        else:
                            result = Plan(tuple(out.indices), problem.cost(out.indices))
                            stats["solver_calls"] = 0
                    elif method == "independent":
                        path = independent_path(problem)
                        candidates = np.asarray(path)
                        result = Plan(path, problem.cost(path))
                        candidate_seconds = time.perf_counter() - begin
                        stats["solver_calls"] = 0
                    else:
                        candidates = select_candidates(problem, budget, method, seed)
                        candidate_seconds = time.perf_counter() - begin
                        if method == "a_refine":
                            candidates, result, extra = refine(problem, candidates, width=width,
                                                               rounds=config["route_a"]["refinement_rounds"],
                                                               proposals=config["route_a"]["refinement_proposals_per_position"])
                            extra["full_scans"] += stats["full_scans"]
                            stats.update(extra)
                        else:
                            result = search(candidates)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - begin
                    row = {"query_id": task["query_id"], "query_family_id": task["query_family_id"], "source": task["source"],
                           "suite": task["suite"], "split": task["split"], "method": method, "seed": seed,
                           "num_tracks": problem.n, "requested_budget": task["budget"], "budget": budget,
                           "candidate_count": len(candidates), "library_size": problem.m, "budget_saturated": budget == problem.m,
                           "phrase_index": task["phrase_index"], "status": result.status, "objective": result.objective,
                           "candidate_ids": problem.ids[candidates].tolist(), "track_ids": problem.ids[list(result.path)].tolist(),
                           "start_id": str(problem.ids[problem.start]), "target_id": str(problem.ids[problem.target]),
                           "setup_seconds": setup_seconds, "candidate_seconds": candidate_seconds,
                           "inference_seconds": elapsed, "total_seconds": elapsed + setup_seconds,
                           "peak_cuda_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
                           "search_expansions": result.expansions, **stats}
                    row.setdefault("cumulative_candidates", len(candidates))
                    row["scored_song_occurrences"] = row["full_scans"] * problem.m + row.get("local_scored_occurrences", 0)
                    if result.status == "ok":
                        row.update(problem.parts(result.path))
                        row.update(label_metrics(root, bank, task, result.path, labels))
                        row["unique_content"] = len(set(bank.groups[list(result.path)].tolist())) == problem.n
                        if not row["unique_content"]:
                            raise AssertionError("repeated audio content in output playlist")
                    if task["suite"] == "e1":
                        oracle_key = hashlib.sha256(problem.x.tobytes() + problem.nodes.tobytes() + str((problem.start, problem.target)).encode()).hexdigest()
                        oracle_file = root / "cache" / "exact" / f"{oracle_key}.json"
                        if oracle_file.exists():
                            exact = json.loads(oracle_file.read_text(encoding="utf-8"))
                        else:
                            full = exact_search(problem)
                            exact = {"objective": full.objective, "certified": full.certified, "status": full.status}
                            write_json(oracle_file, exact)
                        local = exact_search(problem, candidates)
                        row.update(oracle_certified=exact["certified"] and local.certified,
                                   oracle_status=exact["status"], candidate_oracle_status=local.status)
                        if row["oracle_certified"] and exact["status"] == "ok" and local.status == "ok":
                            row.update(full_optimal_cost=exact["objective"], candidate_optimal_cost=local.objective,
                                       candidate_gap=local.objective - exact["objective"],
                                       solver_gap=result.objective - local.objective if result.status == "ok" else None)
                            if row["candidate_gap"] < -1e-6 or (row["solver_gap"] is not None and row["solver_gap"] < -1e-6):
                                raise AssertionError("negative certified gap beyond numeric tolerance")
                    with records_file.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    records.append(row)
                    done.add(task["query_id"])
                    progress("running", current_query=task["query_id"])
                    if len(records) % 25 == 0:
                        print(json.dumps({"method": method, "seed": seed, "completed": len(records), "total": len(tasks)}), flush=True)
                summary = {"status": "completed", "method": method, "seed": seed, "requests": len(records),
                           "successful": sum(r["status"] == "ok" for r in records),
                           "mean_objective": float(np.mean([r["objective"] for r in records])),
                           "wall_seconds_this_invocation": time.monotonic() - begin_run, "task_sha256": task_hash}
                write_json(outdir / "summary.json", summary)
                progress("completed", expected_outputs=[str(records_file), str(outdir / "summary.json")])
                print(json.dumps(summary), flush=True)
            except Exception as exc:
                progress("failed", error=f"{type(exc).__name__}: {exc}")
                raise
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--split", choices=["validation", "test", "all"], default="validation")
    p.add_argument("--tasks-file", help="Separate frozen manifest for generalization; use a fresh --run-id")
    p.add_argument("--no-smooth", action="store_true", help="Ablation; preserve endpoints and renormalize node weights")
    p.add_argument("--planner", choices=["beam", "exact"], default="beam", help="Exact solver ablation is restricted to small libraries")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=["a_grid", "a_refine"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--suites", nargs="+", choices=["e1", "e2", "e3"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-id", default="main")
    p.add_argument("--model-run-id", default="main")
    p.add_argument("--limit-tasks", type=int, default=0)
    p.add_argument("--beam-width", type=int, default=0)
    args = p.parse_args()
    config, root = load_config(args.config)
    evaluate(config, root, args)
