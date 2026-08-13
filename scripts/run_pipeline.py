"""按 configs/pipeline.yaml 依次检查、训练或评估全部基础任务。"""

import gc
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from msmmr.experiments import EXPERIMENTS, run_named_experiment
from msmmr.training.engine import RunOptions
from msmmr.training.utilities import load_yaml


PIPELINE_CONFIG = ROOT / "configs/pipeline.yaml"


def build_run_options(mode: str, pipeline: dict, experiment: dict) -> RunOptions:
    shared = {
        "max_eval_batches": experiment.get("max_eval_batches"),
        "print_every": experiment.get("print_every"),
        "evaluate_test": bool(pipeline.get("evaluate_test", True)),
    }
    if mode == "smoke":
        return RunOptions(
            smoke_test=True,
            target_epochs=int(experiment.get("target_epochs", 2)),
            max_train_batches=int(experiment.get("max_train_batches", 3)),
            **shared,
        )
    if mode == "train":
        return RunOptions(
            target_epochs=experiment.get("target_epochs"),
            resume_from=experiment.get("resume_from"),
            init_weights=experiment.get("init_weights"),
            **shared,
        )
    if mode == "evaluate":
        return RunOptions(
            evaluate_from=experiment.get("checkpoint", "latest"), **shared
        )
    raise ValueError("pipeline mode 只支持 smoke、train 或 evaluate")


def run_validations(config: dict, dry_run: bool) -> None:
    for script in config.get("scripts", []):
        path = ROOT / "validation" / script
        if not path.is_file():
            raise FileNotFoundError(f"验证脚本不存在：{path}")
        print(f"[VALIDATE] {path.name}", flush=True)
        if not dry_run:
            subprocess.run([sys.executable, "-u", str(path)], cwd=ROOT, check=True)


def main() -> None:
    config = load_yaml(PIPELINE_CONFIG)
    mode, dry_run = str(config.get("mode", "smoke")).lower(), bool(
        config.get("dry_run", True)
    )
    selected_experiments = {
        name: value
        for name, value in config.get("experiments", {}).items()
        if value.get("enabled", False)
    }
    unknown = sorted(set(selected_experiments) - EXPERIMENTS.keys())
    if unknown:
        raise ValueError(f"pipeline.yaml 含未知实验：{unknown}")
    print(
        f"pipeline mode={mode} | dry_run={dry_run} | "
        f"evaluate_test={bool(config.get('evaluate_test', True))}"
    )
    if bool(config.get("validations", {}).get("enabled", False)):
        run_validations(config["validations"], dry_run)
    for name, experiment in selected_experiments.items():
        options = build_run_options(mode, config, experiment)
        print(f"[EXPERIMENT] {name} | {options}", flush=True)
        if dry_run:
            continue
        run_named_experiment(name, options)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not selected_experiments:
        print("[INFO] 没有启用实验。")


if __name__ == "__main__":
    main()
