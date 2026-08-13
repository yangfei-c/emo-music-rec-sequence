"""文本 VA 模型、损失、指标、反向和 optimizer step 检查。"""

import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.data import build_text_dataloader, build_text_datasets
from msmmr.metrics import compute_va_metrics
from msmmr.training.utilities import get_device, load_yaml, set_seed
from msmmr.training.factories import (
    build_text_model,
    build_text_optimizer,
    build_va_loss,
)
from validation.helpers import require


def main() -> None:
    dataset_config = load_yaml(ROOT / "configs/datasets.yaml")
    train_config = load_yaml(ROOT / "configs/text_training.yaml")
    set_seed(int(train_config["seed"]))
    device = get_device(str(train_config["device"]))
    dataset = build_text_datasets(dataset_config["text_va"])["train"]
    loader = build_text_dataloader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    batch = next(iter(loader))

    model = build_text_model(train_config, device)
    criterion = build_va_loss(train_config)
    optimizer = build_text_optimizer(model, train_config)
    model.train()
    output = model(batch["texts"], device=device)
    targets = batch["targets"].to(device)
    predictions, embeddings = output["va_predictions"], output["text_embedding"]
    require(
        predictions.shape == (2, 2),
        f"VA 输出应为 [2,2]，实际为 {tuple(predictions.shape)}",
    )
    require(
        embeddings.shape == (2, 256),
        f"embedding 应为 [2,256]，实际为 {tuple(embeddings.shape)}",
    )
    require(torch.isfinite(predictions).all().item(), "预测含 NaN/Inf")

    loss = criterion(predictions, targets)
    require(loss.ndim == 0 and torch.isfinite(loss).item(), f"loss 非有限标量：{loss}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    frozen = [
        parameter for parameter in model.parameters() if not parameter.requires_grad
    ]
    require(
        any(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for _, parameter in trainable
        ),
        "可训练参数没有有限梯度",
    )
    require(all(parameter.grad is None for parameter in frozen), "冻结参数意外收到梯度")
    optimizer.step()

    metrics = compute_va_metrics(
        predictions.detach().float().cpu().numpy(),
        targets.detach().float().cpu().numpy(),
    )
    require(
        all(np.isfinite(value) for value in metrics.values()), "文本 VA 指标含 NaN/Inf"
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for _, parameter in trainable)
    print(
        f"device={device} | texts={len(batch['texts'])} | targets={tuple(targets.shape)}"
    )
    print(
        f"embedding={tuple(embeddings.shape)} | VA={tuple(predictions.shape)} | "
        f"loss={loss.item():.6f}"
    )
    print(f"parameters total={total:,} | trainable={trainable_count:,}")
    print(
        f"metrics CCC={metrics['va_score']:.4f} | "
        f"PCC={metrics['mean_pearson']:.4f} | "
        f"RMSE={metrics['mean_rmse']:.4f}"
    )
    print("[PASS] 文本 VA 前向、损失、指标、反向、冻结路由和 optimizer step 全部通过")


if __name__ == "__main__":
    main()
