"""Prepare and execute the fixed 2026-09-13 comparison; never replace deployment.

Usage: python scripts/training/run_controlled_comparison.py --prepare
       python scripts/training/run_controlled_comparison.py --run
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import sha256, write_json

RUN = "controlled_comparison_20260913"
OUT = ROOT / "artifacts/runs" / RUN
REPORT = ROOT / "results" / RUN
VARIANTS = {
    "no_va": {"text_va_weight": 0.0, "music_va_weight": 0.0},
    "text_va": {"text_va_weight": 0.25, "music_va_weight": 0.0},
    "music_va": {"text_va_weight": 0.0, "music_va_weight": 0.5},
    "full": {"text_va_weight": 0.25, "music_va_weight": 0.5},
    "shared_capacity_matched": {"text_va_weight": 0.25, "music_va_weight": 0.5},
    "clip_style_transfer": {"text_va_weight": 0.25, "music_va_weight": 0.5},
}


def prepare():
    if (REPORT / "plan.json").exists() or OUT.exists():
        raise FileExistsError("Comparison already prepared: preserve existing records")
    base = yaml.safe_load((ROOT / "configs/training/retriever.yaml").read_text(encoding="utf-8"))
    plan = {
        "run_id": RUN, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": [42, 43, 44], "max_epochs": 25,
        "selection": "mean(validation MERGE bidirectional MRR, preference NDCG@10, affect NDCG@10)",
        "test_policy": "existing previously inspected test partitions; no test-driven tuning in this experiment",
        "primary_ablation": "full minus no_va on MTG affect macro NDCG@10",
        "secondary": ["MERGE text-to-music R@10", "Song Describer text-to-music R@10", "MTG preference NDCG@10", "auxiliary VA CCC"],
        "uncertainty": "mean and sample SD over all 3 seeds; paired seed differences, no significance claim",
        "paper_control": "CLIP-Adapter feature block transferred to both CLaMP3 towers, shared across factors; not original few-shot image-classification replication",
        "capacity_control": "two shared residual heads, width 192, versus six independent heads width 64",
        "deployment": "unchanged; no model promotion or web edits",
        "timeout_seconds_per_variant": 1800,
        "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in [
            ROOT / "scripts/training/train_retriever.py", ROOT / "scripts/training/run_controlled_comparison.py",
            ROOT / "src/emomm_sequence/retrieval/model.py", ROOT / "src/emomm_sequence/training/selection.py",
        ]},
        "variants": [],
    }
    for name, weights in VARIANTS.items():
        config = copy.deepcopy(base)
        config["name"] = "Ours" if name == "full" else name
        config["run_id"] = f"{RUN}/{name}"
        config["loss"].update(weights)
        config["evaluation"]["selection_metric"] = "retrieval_only"
        config["model"]["adapter_kind"] = "residual"
        if name in {"shared_capacity_matched", "clip_style_transfer"}:
            config["model"].update(adapter_sharing="across_factors", bottleneck_dim=192)
        if name == "clip_style_transfer":
            config["model"].update(adapter_kind="clip_style", adapter_blend=0.2, dropout=0.0)
        config_path = REPORT / "configs" / f"{name}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-u", "scripts/training/train_retriever.py", "--config",
                   str(config_path.relative_to(ROOT)), "--output", str((OUT / name).relative_to(ROOT))]
        plan["variants"].append({"name": name, "config": str(config_path.relative_to(ROOT)), "command": command})
    write_json(REPORT / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run():
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    for path, expected in plan["code_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise RuntimeError(f"Source changed after plan freeze: {path}")
    status_path = REPORT / "execution.json"
    if status_path.exists():
        raise FileExistsError("Execution already exists: do not overwrite or silently retry")
    records = []
    for variant in plan["variants"]:
        name = variant["name"]
        log_path = REPORT / "logs" / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record = {"name": name, "started_utc": datetime.now(timezone.utc).isoformat(),
                  "command": variant["command"], "log": str(log_path.relative_to(ROOT)), "status": "running"}
        records.append(record)
        write_json(status_path, records)
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(variant["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            record["pid"] = process.pid
            write_json(status_path, records)
            print(f"START {name} pid={process.pid}", flush=True)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f"ALIVE {name} elapsed={elapsed:.0f}s log_bytes={log_path.stat().st_size}", flush=True)
                    if elapsed > plan["timeout_seconds_per_variant"]:
                        process.terminate()
                        process.wait(timeout=30)
                        record.update(status="timeout", duration_seconds=elapsed)
                        write_json(status_path, records)
                        raise TimeoutError(f"{name}: 30-minute experiment limit; stopped without retry")
        record.update(exit_code=process.returncode, duration_seconds=time.monotonic() - started,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      status="completed" if process.returncode == 0 else "failed")
        write_json(status_path, records)
        print(f"END {name} code={process.returncode} seconds={record['duration_seconds']:.1f}", flush=True)
        if process.returncode:
            raise RuntimeError(f"Training failed: {name}. See {log_path}; no automatic retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    prepare() if args.prepare else run()
