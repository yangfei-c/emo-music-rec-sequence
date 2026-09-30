"""Measure large-bank heuristic reference gaps without claiming an optimum."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_config, load_bank, read_jsonl, make_task_problem, write_jsonl
from emomm_sequence.inference.trajectory_candidates import select_candidates, refine
from emomm_sequence.inference.sequence_planner import beam_search
from emomm_sequence.training.records import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--root", type=Path)
    p.add_argument("--run-id", default="main")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--reference-budget", type=int, default=512)
    p.add_argument("--reference-beam", type=int, default=200)
    p.add_argument("--methods", nargs="+", default=["a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce"])
    args = p.parse_args()
    config, root = load_config(args.config)
    if args.root is not None:
        root = args.root.resolve()
    task_file = root / "tasks" / args.profile / f"evaluate_{args.split}.jsonl"
    tasks = {r["query_id"]: r for r in read_jsonl(task_file)}
    with np.load(root / "cache" / "text_bank.npz", allow_pickle=False) as z:
        texts = {k: z[k] for k in z.files}
    methods = {}
    hashes = {}
    for method in args.methods:
        paths = sorted((root / "evaluations" / args.profile / args.split / method).glob("seed_*/main/predictions.jsonl"))
        if not paths:
            raise FileNotFoundError(f"finish primary {method} evaluation")
        for path in paths:
            if not (path.parent / "summary.json").exists():
                raise ValueError(f"primary run is incomplete: {path.parent}")
            methods[(method, path.parents[1].name)] = {r["query_id"]: r for r in read_jsonl(path) if r["suite"] in ("e2", "e3")}
            hashes[str(path)] = sha256(path)
    expected = {k for k, task in tasks.items() if task["suite"] in ("e2", "e3")}
    if any(set(records) != expected for records in methods.values()):
        raise ValueError("all methods must contain the complete E2/E3 manifest; do not silently intersect partial runs")
    directory = root / "reports" / args.profile / "references"
    if args.run_id != "main":
        directory = directory / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"task_sha256": sha256(task_file), "inputs": hashes, "reference_budget": args.reference_budget,
                "reference_beam": args.reference_beam, "certified_optimum": False,
                "negative_gaps_clipped": False, "split": args.split}
    mf = directory / "manifest.json"
    if mf.exists() and json.loads(mf.read_text(encoding="utf-8")) != manifest:
        raise ValueError("reference inputs changed; preserve this version and choose a different output root")
    write_json(mf, manifest)
    banks, references = {}, {}
    output = []
    for query_id in sorted(expected):
        task = tasks[query_id]
        source = task["source"]
        if source not in banks:
            banks[source] = load_bank(root, source, args.split)
        _, problem = make_task_problem(banks[source], task, texts)
        key = hashlib.sha256(problem.nodes.tobytes() + str((source, problem.start, problem.target)).encode()).hexdigest()
        if key not in references:
            pool = select_candidates(problem, min(problem.m, args.reference_budget), "a_grid")
            _, reference, _ = refine(problem, pool, width=args.reference_beam)
            references[key] = reference
        reference = references[key]
        lookup = {str(v): i for i, v in enumerate(problem.ids)}
        for (method, seed), records in methods.items():
            observed = records[query_id]
            local = beam_search(problem, [lookup[v] for v in observed["candidate_ids"]], width=args.reference_beam)
            output.append({"query_id": query_id, "query_family_id": task["query_family_id"], "source": source,
                           "suite": task["suite"], "method": method, "seed": seed, "num_tracks": problem.n,
                           "budget": observed["budget"], "method_status": observed["status"], "reference_status": reference.status,
                           "pool_reference_status": local.status, "method_objective": observed["objective"],
                           "reference_objective": reference.objective, "pool_reference_objective": local.objective,
                           "reference_gap": observed["objective"] - reference.objective if reference.status == "ok" else None,
                           "pool_reference_gap": local.objective - reference.objective if local.status == reference.status == "ok" else None,
                           "ordering_reference_gap": observed["objective"] - local.objective if local.status == "ok" else None,
                           "reference_certified": False, "negative_gaps_preserved": True})
    write_jsonl(directory / "predictions.jsonl", output)
    write_json(directory / "summary.json", {"status": "completed", "rows": len(output), "reference_instances": len(references),
                "scope": "heuristic reference, not exact candidate/solver error decomposition",
                "references_failed": sum(r.status != "ok" for r in references.values())})
    print(json.dumps({"rows": len(output), "reference_instances": len(references)}))


if __name__ == "__main__":
    main()
