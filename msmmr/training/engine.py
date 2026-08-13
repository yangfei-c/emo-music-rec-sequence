from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from msmmr.training.checkpoints import (
    atomic_torch_save,
    build_checkpoint,
    load_model_state,
    load_resume_state,
    resolve_checkpoint_path,
    resolve_resume_path,
)
from msmmr.training.utilities import (
    atomic_json_save,
    logged_output,
    make_scheduler,
    save_yaml_once,
    scalar_metrics,
)
from msmmr.training.loops import evaluate, train_epoch


@dataclass
class RunOptions:
    """训练与评估选项；由脚本变量或 YAML 构造，不使用命令行参数。"""

    smoke_test: bool = False
    target_epochs: int | None = None
    resume_from: str | Path | None = None
    init_weights: str | Path | None = None
    evaluate_from: str | Path | None = None
    output_dir: str | Path | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    print_every: int | None = None
    evaluate_test: bool = False

    def validate(self) -> None:
        if self.resume_from and self.init_weights:
            raise ValueError("resume_from 和 init_weights 不能同时设置")
        if self.resume_from and self.output_dir:
            raise ValueError("续训沿用原目录，不能同时设置 output_dir")
        if self.evaluate_from and any(
            (
                self.smoke_test,
                self.resume_from,
                self.init_weights,
                self.output_dir,
                self.target_epochs,
                self.max_train_batches,
            )
        ):
            raise ValueError(
                "evaluate_from 是独立评估模式，不能与训练、续训或新输出目录选项并用"
            )
        for name in (
            "target_epochs",
            "max_train_batches",
            "max_eval_batches",
            "print_every",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} 必须大于 0")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    output_key: str
    score_key: str
    metric_fn: Callable[[np.ndarray, np.ndarray, dict[str, np.ndarray]], dict]
    metric_fields: tuple[tuple[str, str], ...]
    prediction_payload: Callable[
        [np.ndarray, np.ndarray, dict, dict[str, np.ndarray]], dict
    ]
    dataset_name: str | None = None
    default_print_every: int = 100
    forward_batch: Callable | None = None
    metadata_keys: tuple[str, ...] = ()


def prepare_run(
    outputs_root: Path, options: RunOptions, seed: int
) -> tuple[Path, Path | None, dict | None]:
    options.validate()
    checkpoint_path = (
        resolve_checkpoint_path(options.evaluate_from, outputs_root, "best.pt")
        if options.evaluate_from
        else resolve_resume_path(options.resume_from, outputs_root)
    )
    if checkpoint_path:
        return (
            checkpoint_path.parent,
            checkpoint_path,
            torch.load(checkpoint_path, map_location="cpu", weights_only=False),
        )
    if options.output_dir:
        output_dir = Path(options.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        prefix = "smoke_" if options.smoke_test else ""
        output_dir = (
            outputs_root / f"{prefix}seed{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir, None, None


def metric_text(metrics: dict, fields: tuple[tuple[str, str], ...]) -> str:
    return " | ".join(f"{label}={metrics[key]:.4f}" for label, key in fields)


def _validate_checkpoint(checkpoint: dict, task: TaskSpec) -> None:
    legacy_names = {
        "music_mtg_full183": {"mtg"},
        "music_mtg_mood56": {"mtg_mood"},
    }
    if task.name.startswith("music_va_"):
        legacy_names[task.name] = {"va"}
    accepted = {None, task.name, *legacy_names.get(task.name, set())}
    if checkpoint.get("task_name") not in accepted:
        raise ValueError(f"checkpoint 任务不是 {task.name}")
    if task.dataset_name and checkpoint.get("dataset_name") not in (
        None,
        task.dataset_name,
    ):
        raise ValueError(f"checkpoint 数据集不是 {task.dataset_name}")


def _save_evaluation(
    output_dir: Path,
    stage: str,
    task: TaskSpec,
    metrics: dict,
    predictions: np.ndarray,
    targets: np.ndarray,
    metadata: dict[str, np.ndarray],
) -> None:
    prefix = stage.lower()
    atomic_json_save(output_dir / f"{prefix}_metrics.json", scalar_metrics(metrics))
    np.savez_compressed(
        output_dir / f"{prefix}_predictions.npz",
        **task.prediction_payload(predictions, targets, metrics, metadata),
    )


def _evaluate_checkpoint(
    *,
    task: TaskSpec,
    options: RunOptions,
    output_dir: Path,
    checkpoint: dict,
    loaders: dict,
    model,
    criterion,
    device: torch.device,
    print_every: int,
    amp_enabled: bool,
) -> dict:
    _validate_checkpoint(checkpoint, task)
    load_model_state(model, checkpoint)
    stages = ["validation"] + (["test"] if options.evaluate_test else [])
    result = {"output_dir": output_dir, "best_path": output_dir / "best.pt"}
    for stage in stages:
        metrics, predictions, targets, metadata = evaluate(
            stage=stage.upper(),
            model=model,
            loader=loaders[stage],
            criterion=criterion,
            task=task,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=options.max_eval_batches,
            print_every=max(1, print_every * 2),
        )
        _save_evaluation(
            output_dir, stage, task, metrics, predictions, targets, metadata
        )
        result[f"{stage}_metrics"] = scalar_metrics(metrics)
        print(
            f"[{stage.upper()}] epoch={checkpoint.get('epoch')} | "
            f"loss={metrics['loss']:.6f} | "
            f"{metric_text(metrics, task.metric_fields)}",
            flush=True,
        )
    atomic_json_save(
        output_dir / "completed.json",
        {
            "completed": True,
            "best_epoch": int(checkpoint.get("epoch", -1)),
            "best_score": float(checkpoint.get("best_score", float("nan"))),
            "validation_evaluated": True,
            "test_evaluated": options.evaluate_test,
        },
    )
    return result


def run_experiment(
    *,
    task: TaskSpec,
    options: RunOptions,
    output_dir: Path,
    checkpoint_path: Path | None,
    checkpoint: dict | None,
    dataset_config: dict,
    train_config: dict,
    run_config: dict,
    datasets: dict,
    loaders: dict,
    model,
    criterion,
    optimizer,
    device: torch.device,
    setup_lines: tuple[str, ...] = (),
) -> dict:
    training = train_config["training"]
    print_every = options.print_every or task.default_print_every
    amp_enabled = bool(training.get("amp", True)) and device.type == "cuda"
    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    mode = "EVALUATION" if options.evaluate_from else "TRAINING"
    print(f"\n================ {task.name.upper()} {mode} ================", flush=True)
    print(
        f"device={device} | gpu={gpu} | parameters={parameters:,} | amp={amp_enabled}",
        flush=True,
    )
    print(
        "train/val/test="
        f"{len(datasets['train'])}/{len(datasets['validation'])}/"
        f"{len(datasets['test'])}",
        flush=True,
    )
    for line in setup_lines:
        print(line, flush=True)
    if options.evaluate_from:
        if checkpoint is None:
            raise ValueError("评估模式没有加载 checkpoint")
        print(f"checkpoint={checkpoint_path} | output={output_dir}", flush=True)
        return _evaluate_checkpoint(
            task=task,
            options=options,
            output_dir=output_dir,
            checkpoint=checkpoint,
            loaders=loaders,
            model=model,
            criterion=criterion,
            device=device,
            print_every=print_every,
            amp_enabled=amp_enabled,
        )

    saved_epochs = checkpoint.get("target_epochs") if checkpoint else None
    epochs = int(options.target_epochs or saved_epochs or training["max_epochs"])
    max_train, max_eval = options.max_train_batches, options.max_eval_batches
    if options.smoke_test:
        epochs = options.target_epochs or 2
        max_train, max_eval, print_every = (
            max_train or 3,
            max_eval or 2,
            options.print_every or 1,
        )
    train_batches = (
        min(len(loaders["train"]), max_train) if max_train else len(loaders["train"])
    )
    scheduler_config = train_config.get("scheduler", {})
    scheduler = make_scheduler(
        optimizer,
        epochs * train_batches,
        float(scheduler_config.get("warmup_ratio", 0.05)),
        float(scheduler_config.get("min_learning_rate", 1.0e-6)),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    clip_norm = float(training.get("gradient_clip_norm", 1.0))
    max_skips = int(training.get("max_consecutive_amp_skips", 8))
    patience = int(training.get("early_stopping_patience", 8))
    min_delta = float(training.get("early_stopping_min_delta", 5.0e-4))
    start_epoch, best_score, bad_epochs, history = 1, -math.inf, 0, []

    if checkpoint:
        _validate_checkpoint(checkpoint, task)
        if (
            options.target_epochs
            and saved_epochs
            and options.target_epochs != saved_epochs
        ):
            print(
                f"[WARN] 目标轮数从 {saved_epochs} 改为 {options.target_epochs}，"
                "余弦轨迹不再严格等价。",
                flush=True,
            )
        start_epoch, best_score, bad_epochs, history = load_resume_state(
            checkpoint, model, optimizer, scheduler, scaler
        )
        print(
            f"[RESUME] {checkpoint_path} | "
            f"epoch={start_epoch - 1}->{start_epoch} | best={best_score:.6f}",
            flush=True,
        )
    elif options.init_weights:
        weights_path = Path(options.init_weights).expanduser().resolve()
        load_model_state(
            model, torch.load(weights_path, map_location=device, weights_only=False)
        )
        print(f"[INIT] 已加载模型权重：{weights_path}", flush=True)

    print(
        f"batch={loaders['train'].batch_size} | epochs={epochs} | output={output_dir}",
        flush=True,
    )
    save_yaml_once(output_dir / "run_config.yaml", run_config)
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n[EPOCH {epoch}/{epochs}]", flush=True)
        train_metrics = train_epoch(
            task=task,
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            clip_norm=clip_norm,
            max_consecutive_amp_skips=max_skips,
            max_batches=max_train,
            print_every=print_every,
        )
        validation_metrics, _, _, _ = evaluate(
            stage="VAL",
            model=model,
            loader=loaders["validation"],
            criterion=criterion,
            task=task,
            device=device,
            amp_enabled=amp_enabled,
            max_batches=max_eval,
            print_every=max(1, print_every * 2),
        )
        score = (
            -validation_metrics["loss"]
            if options.smoke_test
            else float(validation_metrics[task.score_key])
        )
        if not math.isfinite(score):
            raise RuntimeError("validation 选模指标为 NaN/Inf")
        improved = score > best_score + min_delta
        best_score, bad_epochs = (
            (score, 0) if improved else (best_score, bad_epochs + 1)
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": scalar_metrics(validation_metrics),
                "best_score": best_score,
            }
        )
        atomic_json_save(output_dir / "history.json", history)
        state = build_checkpoint(
            epoch=epoch,
            target_epochs=epochs,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            bad_epochs=bad_epochs,
            history=history,
            dataset_config=dataset_config,
            train_config=train_config,
            task_name=task.name,
            dataset_name=task.dataset_name,
        )
        atomic_torch_save(last_path, state)
        if improved:
            atomic_torch_save(best_path, state)
            print(f"[BEST] epoch={epoch}", flush=True)
        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} | "
            f"val_loss={validation_metrics['loss']:.6f} | "
            f"{metric_text(validation_metrics, task.metric_fields)} | "
            f"AMP_skips={train_metrics['amp_skipped_steps']} "
            f"| bad_epochs={bad_epochs}/{patience}",
            flush=True,
        )
        if bad_epochs >= patience:
            print(f"[EARLY STOP] 连续 {patience} 个 epoch 未提升。", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("没有生成最佳 checkpoint")
    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    final_options = RunOptions(
        max_eval_batches=max_eval,
        print_every=print_every,
        evaluate_test=options.evaluate_test,
    )
    result = _evaluate_checkpoint(
        task=task,
        options=final_options,
        output_dir=output_dir,
        checkpoint=best_checkpoint,
        loaders=loaders,
        model=model,
        criterion=criterion,
        device=device,
        print_every=print_every,
        amp_enabled=amp_enabled,
    )
    result.update(best_path=best_path, last_path=last_path, history=history)
    if not options.evaluate_test:
        print("[TEST LOCKED] 已落盘最佳验证集指标与预测；测试集未读取。", flush=True)
    return result


def execute_experiment(
    *,
    output_root: Path,
    options: RunOptions,
    seed: int,
    task,
    dataset_config: dict,
    train_config: dict,
    run_config: dict,
    datasets: dict,
    loaders: dict,
    model,
    criterion,
    optimizer,
    device,
    setup_lines: tuple[str, ...],
    prepare_output: Callable[[Path], None] | None = None,
):
    """准备输出目录和日志，再进入统一训练或独立评估流程。"""
    output_dir, checkpoint_path, checkpoint = prepare_run(output_root, options, seed)
    if prepare_output and checkpoint_path is None:
        prepare_output(output_dir)
    log_name = "evaluation.log" if options.evaluate_from else "train.log"
    with logged_output(output_dir / log_name):
        print(time.strftime("[RUN] %Y-%m-%d %H:%M:%S"))
        if checkpoint_path:
            label = "evaluate" if options.evaluate_from else "resume"
            print(f"[RUN] {label}={checkpoint_path}")
        return run_experiment(
            task=task,
            options=options,
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            dataset_config=dataset_config,
            train_config=train_config,
            run_config=run_config,
            datasets=datasets,
            loaders=loaders,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            setup_lines=setup_lines,
        )
