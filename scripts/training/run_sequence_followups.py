"""Bounded continuation and predeclared pilot extensions; never restart crashes."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import ROOT, load_config
from emomm_sequence.training.sequence_selector import process_identity
from emomm_sequence.training.records import write_json


def verified_alive(record):
    if not record.get("pid") or not record.get("created_utc"):
        return False
    try:
        process = psutil.Process(record["pid"])
        return process.is_running() and abs(process.create_time() - datetime.fromisoformat(record["created_utc"]).timestamp()) < 1
    except psutil.NoSuchProcess:
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--max-training-segments", type=int, default=3)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    config, root = load_config(args.config)
    directory = root / "followups"
    directory.mkdir(parents=True, exist_ok=True)
    progress_file = directory / "progress.json"
    previous = json.loads(progress_file.read_text(encoding="utf-8")) if progress_file.exists() else {}
    if verified_alive(previous):
        raise RuntimeError("another followup supervisor is alive")
    if previous and not args.resume:
        raise FileExistsError("use --resume to preserve followup records")
    state = {**process_identity(), "scope": "pilot", "status": "waiting_for_primary", "steps": previous.get("steps", []),
             "attempts": previous.get("attempts", []), "active_command": None,
             "entire_blueprint_complete": False, "max_training_segments": args.max_training_segments}
    def publish():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(progress_file, state)
    publish()
    def command(name, argv):
        stamp = int(time.time())
        log = directory / f"{name}_{stamp}.log"
        record = {"name": name, "command": [sys.executable, "-u", *argv], "log": str(log)}
        state["steps"] = [r for r in state["steps"] if r["name"] != name] + [record]
        state["attempts"].append(record)
        state["active_step"] = name
        state["active_command"] = record["command"]
        state["status"] = "running"
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(record["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"})
            record.update(pid=child.pid, created_utc=datetime.fromtimestamp(psutil.Process(child.pid).create_time(), timezone.utc).isoformat())
            while child.poll() is None:
                publish()
                time.sleep(10)
            record["exit_code"] = child.returncode
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        publish()
        return child.returncode
    def completed(name):
        return any(r["name"] == name and r.get("exit_code") == 0 for r in state["steps"])
    def allowed_continuation(training_progress):
        elapsed = float(training_progress["elapsed_seconds"])
        limit = config["profiles"]["pilot"]["max_run_seconds"]
        # Each segment ends after a batch. Allow <120s of batch/validation overhead.
        return elapsed < (args.max_training_segments - 1) * limit + 120
    try:
        # Wait using OS identity; a stale state file is never enough to restart.
        primary_path = root / "pipeline" / "progress.json"
        while True:
            primary = json.loads(primary_path.read_text(encoding="utf-8"))
            if verified_alive(primary):
                state["status"] = "waiting_for_primary"
                state["primary_pid"] = primary["pid"]
                publish()
                time.sleep(10)
                continue
            if primary["status"] == "completed":
                break
            if primary["status"] != "incomplete_timeout":
                raise RuntimeError(f"primary requires diagnosis; not restarting {primary['status']}")
            step = primary["steps"][-1]
            if step.get("exit_code") != 2 or verified_alive(step):
                raise RuntimeError("primary has no verified terminal timeout")
            run = Path(step["expected_outputs"][0]).parent
            training = json.loads((run / "progress.json").read_text(encoding="utf-8"))
            if training["status"] != "incomplete_timeout" or verified_alive(training) or not allowed_continuation(training):
                raise RuntimeError("training continuation cap reached or process still alive")
            code = command("continue_primary", ["scripts/training/run_sequence_experiments.py", "--profile", "pilot", "--resume"])
            if code not in (0, 2):
                raise RuntimeError(f"primary crashed with exit {code}")
        jobs = [("prepare_generalization", ["scripts/data/prepare_sequence_generalization.py", "--profile", "pilot", "--split", "validation"], None)]
        def train_job(run_id, options):
            jobs.append((f"train_{run_id}", ["scripts/training/train_sequence_selector.py", "--profile", "pilot", "--variant", "set",
                         "--seed", "42", "--run-id", run_id, *options], root / "runs" / "pilot" / "set" / "seed_42" / run_id))
        def eval_job(name, options):
            jobs.append((name, ["scripts/evaluation/evaluate_sequence_routes.py", "--profile", "pilot", "--split", "validation",
                               "--seeds", "42", *options], None))
        for kind in ("lengths", "artist_disjoint", "all_library"):
            split = "all" if kind == "all_library" else "validation"
            jobs.append((f"generalization_{kind}", ["scripts/evaluation/evaluate_sequence_routes.py", "--profile", "pilot", "--split", split,
                         "--methods", "a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce", "--seeds", "42",
                         "--tasks-file", str(root / "tasks" / "pilot" / "generalization" / "validation" / f"{kind}.jsonl"),
                         "--run-id", f"generalization_{kind}"], None))
        for run_id, options in (
            ("ablation_no_set", ["--stage", "all", "--no-set"]),
            ("ablation_no_length", ["--stage", "all", "--no-length"]),
            ("ablation_warm_only", ["--stage", "warm"]),
            ("ablation_rl_only", ["--stage", "rl", "--no-aux-warm"])):
            train_job(run_id, options)
            eval_job(f"evaluate_{run_id}", ["--methods", "b_set", "--model-run-id", run_id, "--run-id", run_id])
        eval_job("changed_planner_exact", ["--methods", "a_grid", "b_set", "--planner", "exact", "--suites", "e1", "--run-id", "changed_planner_exact"])
        eval_job("a_no_smooth", ["--methods", "a_grid", "a_refine", "--no-smooth", "--run-id", "a_no_smooth"])
        eval_job("beam_200", ["--methods", "a_grid", "a_refine", "b_set", "--beam-width", "200", "--run-id", "beam_200"])
        for mode in ("point", "raw_interval", "cqr_interval"):
            train_job(f"va_{mode}", ["--stage", "all", "--objective", mode])
            jobs.append((f"evaluate_va_{mode}", ["scripts/evaluation/evaluate_sequence_va.py", "--profile", "pilot", "--split", "validation",
                         "--mode", mode, "--methods", "a_grid", "a_refine", "b_set", "--device", "cuda"], None))
        for run_id in ("main", "generalization_lengths", "generalization_artist_disjoint", "generalization_all_library", "changed_planner_exact", "beam_200"):
            jobs.append((f"report_{run_id}", ["scripts/evaluation/report_sequence_routes.py", "--root", str(root), "--profile", "pilot", "--run-id", run_id], None))
        jobs.append(("report_extensions", ["scripts/evaluation/report_sequence_extensions.py", "--root", str(root), "--profile", "pilot"], None))
        state["planned_steps"] = [name for name, _, _ in jobs]
        for name, argv, run in jobs:
            if completed(name):
                continue
            if run is not None and (run / "last.pt").exists():
                argv = [*argv, "--resume"]
            while True:
                code = command(name, argv)
                if code == 0:
                    break
                if code != 2 or run is None:
                    raise RuntimeError(f"{name} failed with exit {code}; inspect its log")
                training = json.loads((run / "progress.json").read_text(encoding="utf-8"))
                if verified_alive(training) or training.get("status") != "incomplete_timeout" or not allowed_continuation(training):
                    raise RuntimeError(f"{name} reached bounded continuation cap")
                if "--resume" not in argv:
                    argv = [*argv, "--resume"]
        state.update(status="completed", active_step=None, active_command=None, expected_outputs=[str(root / "reports" / "pilot" / "summary.json"),
                     str(root / "reports" / "pilot" / "va_summary.json")], remaining_scope=["pilot expansion gate", "formal three-seed runs if gate passes", "final synthesis and figures"])
        publish()
        return 0
    except Exception as exc:
        state.update(status="failed", error=repr(exc))
        publish()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
