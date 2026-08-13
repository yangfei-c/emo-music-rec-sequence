"""音乐损失、梯度路由与 optimizer step 检查。"""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.data import MTGTaggingDataset, MusicVADataset, build_music_dataloader
from msmmr.losses import GroupBalancedASL, TagSubsetWeightedBCE, VALoss
from msmmr.models import MusicMultiTaskModel
from msmmr.training.utilities import load_yaml
from validation.helpers import require, resolve_device


DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
TRAIN_CONFIG = ROOT / "configs" / "music_training.yaml"


def check_gradients(model: MusicMultiTaskModel, mtg: bool) -> None:
    routes = {
        "segment_projection": True,
        "song_fusion": True,
        "genre_head": mtg,
        "instrument_head": mtg,
        "mood_head": mtg,
        "va_head": not mtg,
    }

    for name, expected in routes.items():
        gradients = [parameter.grad for parameter in getattr(model, name).parameters()]

        if not expected:
            require(
                all(gradient is None for gradient in gradients), f"{name} 不应产生梯度"
            )
            continue

        require(all(gradient is not None for gradient in gradients), f"{name} 缺少梯度")
        require(
            all(torch.isfinite(gradient).all().item() for gradient in gradients),
            f"{name} 梯度包含 NaN 或 Inf",
        )
        require(
            sum(gradient.float().square().sum() for gradient in gradients).item() > 0,
            f"{name} 梯度为 0",
        )


def check_batch(
    name: str,
    loader,
    model: MusicMultiTaskModel,
    tag_loss: GroupBalancedASL,
    va_loss: VALoss,
    optimizer,
    device: torch.device,
) -> None:
    batch = next(iter(loader))
    features, mask, targets = (
        batch["features"].to(device),
        batch["segment_mask"].to(device),
        batch["targets"].to(device),
    )
    outputs = model(features, mask)
    mtg = name == "mtg"

    components = (
        tag_loss(outputs["tag_logits"], targets, True)
        if mtg
        else va_loss(outputs["va_predictions"], targets, True)
    )
    loss = components["total"]
    require(loss.ndim == 0 and torch.isfinite(loss).item(), f"{name} loss 不是有限标量")

    before = model.segment_projection[1].weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    check_gradients(model, mtg)
    optimizer.step()
    require(
        not torch.equal(before, model.segment_projection[1].weight.detach()),
        f"{name} 参数未更新",
    )

    if mtg:
        require(
            torch.allclose(
                loss,
                torch.stack(
                    (components["genre"], components["instrument"], components["mood"])
                ).mean(),
            ),
            "MTG 分组平均错误",
        )
        detail = (
            f"genre={components['genre'].item():.6f} | "
            f"instrument={components['instrument'].item():.6f} | "
            f"mood={components['mood'].item():.6f}"
        )
    else:
        require(
            torch.allclose(
                loss, torch.stack((components["valence"], components["arousal"])).mean()
            ),
            f"{name} 两轴平均错误",
        )
        detail = (
            f"valence={components['valence'].item():.6f} | "
            f"arousal={components['arousal'].item():.6f}"
        )

    print(
        f"[PASS] {name.upper()} | input={tuple(features.shape)} | "
        f"total={loss.item():.6f} | {detail}"
    )


def check_subset_route(
    group: str, loader, model: MusicMultiTaskModel, device: torch.device
) -> None:
    groups = {
        "genre": (slice(0, 87), "genre_head"),
        "instrument": (slice(87, 127), "instrument_head"),
        "mood": (slice(127, 183), "mood_head"),
    }
    label_slice, selected_head = groups[group]
    batch = next(iter(loader))
    features, mask, targets = (
        batch["features"].to(device),
        batch["segment_mask"].to(device),
        batch["targets"].to(device),
    )
    model.zero_grad(set_to_none=True)
    weights = torch.ones(label_slice.stop - label_slice.start, device=device)
    loss = TagSubsetWeightedBCE(label_slice, weights)(
        model(features, mask)["tag_logits"], targets
    )
    loss.backward()
    for head in ("genre_head", "instrument_head", "mood_head"):
        gradients = [parameter.grad for parameter in getattr(model, head).parameters()]
        if head == selected_head:
            require(
                all(
                    gradient is not None and torch.count_nonzero(gradient).item() > 0
                    for gradient in gradients
                ),
                f"{group}-only 未更新 {head}",
            )
        else:
            require(
                all(
                    gradient is None or torch.count_nonzero(gradient).item() == 0
                    for gradient in gradients
                ),
                f"{group}-only 不应更新 {head}",
            )
    print(
        f"[PASS] {group}-only 梯度路由 | loss={loss.item():.6f} | selected={selected_head}"
    )


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config, model_config = train_config["data"], train_config["model"]
    tag_config, va_config = train_config["loss"]["tag"], train_config["loss"]["va"]
    device = resolve_device(str(train_config.get("device", "cpu")))

    model = (
        MusicMultiTaskModel(
            layer_indices=train_config["feature"]["layer_indices"],
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            pooling=str(train_config["feature"].get("pooling", "mean_std")),
            pooling_eps=float(model_config.get("pooling_eps", 1.0e-5)),
        )
        .to(device)
        .train()
    )

    tag_loss = GroupBalancedASL(
        float(tag_config["gamma_pos"]),
        float(tag_config["gamma_neg"]),
        float(tag_config["clip"]),
        float(tag_config["eps"]),
    )
    va_loss = VALoss(float(va_config["beta"]), float(va_config.get("ccc_weight", 0.0)))

    optimizer_config = train_config["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )

    datasets = {
        "mtg": MTGTaggingDataset(dataset_config["mtg"], "train"),
        "genre": MTGTaggingDataset(dataset_config["mtg"], "train", tag_set="genre"),
        "instrument": MTGTaggingDataset(
            dataset_config["mtg"], "train", tag_set="instrument"
        ),
        "mood": MTGTaggingDataset(dataset_config["mtg"], "train", tag_set="moodtheme"),
        "deam": MusicVADataset(dataset_config["deam"], "deam", "train"),
        "pmemo": MusicVADataset(dataset_config["pmemo"], "pmemo", "train"),
    }
    loaders = {
        name: build_music_dataloader(
            dataset,
            int(data_config["batch_size"]),
            True,
            int(data_config["num_workers"]),
            bool(data_config["pin_memory"]),
        )
        for name, dataset in datasets.items()
    }

    print(
        f"device={device} | "
        f"ASL=({tag_loss.gamma_pos},{tag_loss.gamma_neg},{tag_loss.clip}) | "
        f"SmoothL1={va_loss.beta} | CCC weight={va_loss.ccc_weight}"
    )
    for name in ("mtg", "deam", "pmemo"):
        check_batch(name, loaders[name], model, tag_loss, va_loss, optimizer, device)
    for group in ("genre", "instrument", "mood"):
        check_subset_route(group, loaders[group], model, device)
    print("[PASS] loss、梯度路由和参数更新检查全部通过")


if __name__ == "__main__":
    main()
