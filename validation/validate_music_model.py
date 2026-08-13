"""音乐模型输出、维度与 padding mask 检查。"""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.data import MTGTaggingDataset, MusicVADataset, build_music_dataloader
from msmmr.models import MusicMultiTaskModel
from msmmr.training.utilities import load_yaml
from validation.helpers import require, resolve_device


DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
TRAIN_CONFIG = ROOT / "configs" / "music_training.yaml"


@torch.no_grad()
def check_batch(
    name: str, loader, model: MusicMultiTaskModel, device: torch.device
) -> None:
    batch = next(iter(loader))
    features, mask, targets = (
        batch["features"].to(device),
        batch["segment_mask"].to(device),
        batch["targets"].to(device),
    )
    outputs = model(features, mask)
    batch_size, hidden_dim = features.shape[0], model.hidden_dim

    expected = {
        "song_embedding": (batch_size, hidden_dim),
        "genre_logits": (batch_size, 87),
        "instrument_logits": (batch_size, 40),
        "mood_logits": (batch_size, 56),
        "tag_logits": (batch_size, 183),
        "va_predictions": (batch_size, 2),
    }

    require(
        targets.shape == (batch_size, 183 if name == "mtg" else 2),
        f"{name} targets 形状错误：{tuple(targets.shape)}",
    )
    require(set(outputs) == set(expected), f"{name} 输出键错误：{sorted(outputs)}")

    for key, shape in expected.items():
        require(
            outputs[key].shape == shape,
            f"{name} {key} 形状为 {tuple(outputs[key].shape)}，预期为 {shape}",
        )
        require(
            torch.isfinite(outputs[key]).all().item(), f"{name} {key} 包含 NaN 或 Inf"
        )

    joined = torch.cat(
        (outputs["genre_logits"], outputs["instrument_logits"], outputs["mood_logits"]),
        dim=-1,
    )
    require(torch.equal(outputs["tag_logits"], joined), f"{name} tag_logits 拼接错误")

    padding = ~mask
    padding_status = "无 padding"

    if padding.any().item():
        changed = features.clone()
        changed[padding] = torch.randn_like(changed[padding]) * 1000
        changed_embedding = model(changed, mask)["song_embedding"]
        require(
            torch.allclose(
                outputs["song_embedding"], changed_embedding, rtol=1.0e-4, atol=1.0e-5
            ),
            f"{name} padding 影响了输出",
        )
        padding_status = "mask 检查通过"

    print(
        f"[PASS] {name.upper()} | input={tuple(features.shape)} | "
        f"target={tuple(targets.shape)} | "
        f"embedding={tuple(outputs['song_embedding'].shape)} | "
        f"tag={tuple(outputs['tag_logits'].shape)} | "
        f"va={tuple(outputs['va_predictions'].shape)} | {padding_status}"
    )


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config, model_config = train_config["data"], train_config["model"]
    device = resolve_device(str(train_config.get("device", "cpu")))

    model = (
        MusicMultiTaskModel(
            layer_indices=train_config["feature"]["layer_indices"],
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config["dropout"]),
            pooling_eps=float(model_config.get("pooling_eps", 1.0e-5)),
        )
        .to(device)
        .eval()
    )

    datasets = {
        "mtg": MTGTaggingDataset(dataset_config["mtg"], "train"),
        "deam": MusicVADataset(dataset_config["deam"], "deam", "train"),
        "pmemo": MusicVADataset(dataset_config["pmemo"], "pmemo", "train"),
    }

    loaders = {
        name: build_music_dataloader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=True,
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )
        for name, dataset in datasets.items()
    }

    parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"device={device} | "
        f"layers={tuple(train_config['feature']['layer_indices'])} | "
        f"hidden={model.hidden_dim} | parameters={parameters:,}"
    )

    for name in ("mtg", "deam", "pmemo"):
        check_batch(name, loaders[name], model, device)

    print("[PASS] 三个数据集前向传播检查全部通过")


if __name__ == "__main__":
    main()
