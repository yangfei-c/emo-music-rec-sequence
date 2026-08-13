"""任务无关的单轮训练与评估循环。"""

import time

import numpy as np
import torch

from msmmr.training.utilities import (
    gpu_memory,
    move_batch,
    nonfinite_batch_message,
    nonfinite_gradient_names,
)


def forward_task(
    task, model, batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if task.forward_batch:
        return task.forward_batch(model, batch, device)
    features, mask, targets = move_batch(batch, device)
    return model(features, mask)[task.output_key], targets


def _append_metadata(
    store: dict[str, list], batch: dict, keys: tuple[str, ...]
) -> None:
    for key in keys:
        if key not in batch:
            raise KeyError(f"batch 缺少评价元数据：{key}")
        values = batch[key]
        store[key].extend(
            values.detach().cpu().tolist()
            if isinstance(values, torch.Tensor)
            else list(values)
        )


def train_epoch(
    *,
    task,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    amp_enabled: bool,
    clip_norm: float,
    max_consecutive_amp_skips: int,
    max_batches: int | None,
    print_every: int,
) -> dict:
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = skipped_steps = consecutive_skips = 0
    started = time.perf_counter()
    print(f"  [TRAIN] 共 {total} batches", flush=True)
    for step, batch in enumerate(loader, 1):
        if step > total:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            prediction, targets = forward_task(task, model, batch, device)
            loss = criterion(prediction, targets)
        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"训练 loss 为 {loss.item()}；"
                f"{nonfinite_batch_message(batch, prediction, targets)}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        if not amp_enabled and not torch.isfinite(gradient_norm).item():
            raise RuntimeError(f"梯度含 NaN/Inf：{nonfinite_gradient_names(model)}")
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = amp_enabled and scaler.get_scale() < scale_before
        if skipped:
            skipped_steps += 1
            consecutive_skips += 1
            print(
                f"  [WARN] step={step} AMP overflow，"
                f"scale {scale_before:g}->{scaler.get_scale():g}",
                flush=True,
            )
            if consecutive_skips >= max_consecutive_amp_skips:
                raise RuntimeError(
                    f"连续 {consecutive_skips} 个 batch 发生 AMP overflow"
                )
        else:
            consecutive_skips = 0
            scheduler.step()
        batch_size = targets.shape[0]
        loss_sum += loss.detach().item() * batch_size
        sample_count += batch_size
        if step == 1 or step % print_every == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(
                f"  [TRAIN] {step:4d}/{total} | loss={loss.item():.6f} | "
                f"avg={loss_sum / sample_count:.6f} "
                f"| lr={optimizer.param_groups[0]['lr']:.2e} | skipped={skipped_steps} "
                f"| {sample_count / elapsed:.1f} samples/s | "
                f"gpu_mem={gpu_memory(device)}",
                flush=True,
            )
    if sample_count == 0:
        raise RuntimeError("训练 DataLoader 没有产生样本")
    return {
        "loss": loss_sum / sample_count,
        "samples": sample_count,
        "seconds": time.perf_counter() - started,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "amp_skipped_steps": skipped_steps,
    }


@torch.no_grad()
def evaluate(
    *,
    stage: str,
    model,
    loader,
    criterion,
    task,
    device,
    amp_enabled: bool,
    max_batches: int | None,
    print_every: int,
) -> tuple[dict, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    total = min(len(loader), max_batches) if max_batches else len(loader)
    loss_sum = sample_count = 0
    predictions, targets = [], []
    metadata = {key: [] for key in task.metadata_keys}
    started = time.perf_counter()
    print(f"  [{stage}] 共 {total} batches", flush=True)
    for step, batch in enumerate(loader, 1):
        if step > total:
            break
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            prediction, target = forward_task(task, model, batch, device)
            loss = criterion(prediction, target)
        if (
            not torch.isfinite(loss).item()
            or not torch.isfinite(prediction).all().item()
        ):
            raise RuntimeError(
                f"{stage} 出现 NaN/Inf；"
                f"{nonfinite_batch_message(batch, prediction, target)}"
            )
        batch_size = target.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size
        predictions.append(prediction.float().cpu())
        targets.append(target.float().cpu())
        _append_metadata(metadata, batch, task.metadata_keys)
        if step == 1 or step % print_every == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(
                f"  [{stage}] {step:4d}/{total} | "
                f"avg_loss={loss_sum / sample_count:.6f} | "
                f"{sample_count / elapsed:.1f} samples/s",
                flush=True,
            )
    if not predictions:
        raise RuntimeError(f"{stage} DataLoader 没有产生样本")
    prediction_array, target_array = (
        torch.cat(predictions).numpy(),
        torch.cat(targets).numpy(),
    )
    metadata_arrays = {key: np.asarray(values) for key, values in metadata.items()}
    metrics = task.metric_fn(prediction_array, target_array, metadata_arrays)
    metrics.update(loss=loss_sum / sample_count, samples=sample_count)
    return metrics, prediction_array, target_array, metadata_arrays
