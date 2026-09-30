"""Sequential experiment queue with real child handles, logs and completion evidence."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import ROOT, load_config
from emomm_sequence.training.records import write_json
from emomm_sequence.training.sequence_selector import process_identity, checkpoint_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    config, root = load_config(args.config)
    directory = root / "pipeline"
    directory.mkdir(parents=True, exist_ok=True)
    progress_file = directory / "progress.json"
    old = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}
    if old.get("pid"):
        try:
            process = psutil.Process(old["pid"])
            born = datetime.fromisoformat(old["created_utc"]).timestamp()
            if abs(process.create_time() - born) < 1 and process.is_running():
                raise RuntimeError("another sequence pipeline is already alive")
        except psutil.NoSuchProcess:
            pass
    if old and old.get("scope") == args.profile and not args.resume:
        raise FileExistsError("pipeline exists; use --resume to preserve completed steps")
    previous_steps = old.get("steps", []) if old.get("scope") == args.profile else []
    completed = {s["name"]: s for s in previous_steps if s.get("exit_code") == 0 and s.get("expected_outputs") and all(Path(x).exists() for x in s["expected_outputs"])}
    state = {**process_identity(), "scope": args.profile, "status": "running", "steps": [], "active_command": None,
             "started_utc": datetime.now(timezone.utc).isoformat(), "expected_outputs": []}
    def publish():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(progress_file, state)
    base = [sys.executable, "-u"]
    def job(name, argv, outputs):
        return (name, base + argv, [str(x) for x in outputs])
    jobs = [job("prepare", ["scripts/data/prepare_sequence_tasks.py", "--config", args.config, "--profile", args.profile],
                [root / "tasks" / args.profile / "manifest.json"])]
    seeds = config["profiles"][args.profile]["seeds"]
    split = "validation" if args.profile == "pilot" else "test"
    # Frozen route is operational before any new model is trained.
    a_methods = ["random", "three_route", "independent", "a_grid", "a_refine"]
    jobs.append(job("evaluate_a", ["scripts/evaluation/evaluate_sequence_routes.py", "--config", args.config,
                                  "--profile", args.profile, "--split", split, "--methods", *a_methods,
                                  "--seeds", *map(str, seeds)],
                    [root / "evaluations" / args.profile / split / m / f"seed_{seeds[0]}" / "main" / "summary.json" for m in a_methods]))
    for variant in ("set", "direct_path", "dynamic_reduce"):
        for seed in seeds:
            run = checkpoint_path(root, args.profile, variant, seed)
            argv = ["scripts/training/train_sequence_selector.py", "--profile", args.profile, "--variant", variant,
                    "--seed", str(seed), "--stage", "all"]
            if (run / "last.pt").exists():
                argv.append("--resume")
            jobs.append(job(f"train_{variant}_{seed}", argv, [run / "summary.json", run / "best.pt", run / "selection.json"]))
        method = "b_set" if variant == "set" else variant
        jobs.append(job(f"evaluate_{method}", ["scripts/evaluation/evaluate_sequence_routes.py", "--config", args.config,
                        "--profile", args.profile, "--split", split, "--methods", method, "--seeds", *map(str, seeds)],
                        [root / "evaluations" / args.profile / split / method / f"seed_{s}" / "main" / "summary.json" for s in seeds]))
    jobs.append(job("report", ["scripts/evaluation/report_sequence_routes.py", "--root", str(root), "--profile", args.profile],
                    [root / "reports" / args.profile / "summary.json", root / "reports" / args.profile / "results.csv",
                     root / "reports" / args.profile / "实验结果.md"]))
    state["planned_steps"] = [name for name, _, _ in jobs]
    state["expected_outputs"] = jobs[-1][2]
    publish()
    try:
        for name, cmd, outputs in jobs:
            if name in completed:
                state["steps"].append(completed[name])
                publish()
                continue
            log = directory / f"{args.profile}_{name}_{int(time.time())}.log"
            record = {"name": name, "command": cmd, "expected_outputs": outputs, "log": str(log),
                      "started_utc": datetime.now(timezone.utc).isoformat()}
            state["steps"].append(record)
            with log.open("w", encoding="utf-8") as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1"})
                record["pid"] = child.pid
                record["created_utc"] = datetime.fromtimestamp(psutil.Process(child.pid).create_time(), timezone.utc).isoformat()
                state["active_command"] = cmd
                state["active_step"] = name
                state["child_pid"] = child.pid
                publish()
                while child.poll() is None:
                    time.sleep(10)
                    publish()
                record["exit_code"] = child.returncode
                record["finished_utc"] = datetime.now(timezone.utc).isoformat()
                if child.returncode != 0 or not all(Path(x).exists() for x in outputs):
                    state["status"] = "incomplete_timeout" if child.returncode == 2 else "failed"
                    state["failure_step"] = name
                    publish()
                    return child.returncode or 1
            publish()
        state["status"] = "completed"
        state["active_command"] = None
        state["active_step"] = None
        state["child_pid"] = None
        state["entire_blueprint_complete"] = False
        state["remaining_scope"] = ["independent E4 and parser", "VA/CQR extension", "generalization and ablations", "literature evidence comparison"]
        publish()
        return 0
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = repr(exc)
        publish()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
