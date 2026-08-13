"""音乐 Dataset 与 DataLoader 检查。"""

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.data import MTGTaggingDataset, MusicVADataset, build_music_dataloader
from msmmr.training.utilities import load_yaml
from validation.helpers import require


DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
TRAIN_CONFIG = ROOT / "configs" / "music_training.yaml"
EXPECTED_LENGTHS = {
    "mtg_full183": 32859,
    "mtg_genre87": 32572,
    "mtg_instrument40": 14395,
    "mtg_mood56": 9949,
    "deam": 1261,
    "pmemo": 536,
}
SUBSET_SLICES = {
    "mtg_genre87": slice(0, 87),
    "mtg_instrument40": slice(87, 127),
    "mtg_mood56": slice(127, 183),
}


def check_batch(name: str, dataset, dataloader, target_dim: int) -> None:
    print(f"\n================ {name.upper()} ================")
    batch = next(iter(dataloader))
    features, mask, lengths, targets = (
        batch["features"],
        batch["segment_mask"],
        batch["lengths"],
        batch["targets"],
    )

    require(
        len(dataset) == EXPECTED_LENGTHS[name],
        f"数据集长度为 {len(dataset)}，预期为 {EXPECTED_LENGTHS[name]}",
    )
    require(
        features.ndim == 4 and tuple(features.shape[2:]) == (12, 768),
        f"features 形状错误：{tuple(features.shape)}",
    )
    require(
        mask.shape == features.shape[:2], f"segment_mask 形状错误：{tuple(mask.shape)}"
    )
    require(
        targets.ndim == 2 and targets.shape[1] == target_dim,
        f"targets 形状错误：{tuple(targets.shape)}",
    )
    require(
        lengths.shape[0] == features.shape[0],
        f"lengths 形状错误：{tuple(lengths.shape)}",
    )

    require(features.dtype == torch.float32, f"features dtype 错误：{features.dtype}")
    require(targets.dtype == torch.float32, f"targets dtype 错误：{targets.dtype}")
    require(mask.dtype == torch.bool, f"segment_mask dtype 错误：{mask.dtype}")
    require(lengths.dtype == torch.long, f"lengths dtype 错误：{lengths.dtype}")

    require(torch.isfinite(features).all().item(), "features 包含 NaN 或 Inf")
    require(torch.isfinite(targets).all().item(), "targets 包含 NaN 或 Inf")
    require((lengths > 0).all().item(), "存在没有有效分段的歌曲")
    require(torch.equal(mask.sum(dim=1), lengths), "segment_mask 与 lengths 不一致")

    if (~mask).any().item():
        require(
            torch.count_nonzero(features[~mask]).item() == 0, "padding 位置存在非零特征"
        )

    if name.startswith("mtg"):
        require(
            ((targets == 0) | (targets == 1)).all().item(), "MTG target 不是二值标签"
        )
        require((targets.sum(dim=1) >= 1).all().item(), "MTG 中存在没有正标签的样本")
        if name in SUBSET_SLICES:
            label_slice = SUBSET_SLICES[name]
            outside = torch.cat(
                (targets[:, : label_slice.start], targets[:, label_slice.stop :]), dim=1
            )
            require(
                torch.count_nonzero(outside).item() == 0,
                f"{name} manifest 包含组外正标签",
            )
    else:
        require(
            targets.min().item() >= -1.0 - 1e-6 and targets.max().item() <= 1.0 + 1e-6,
            f"VA 超出 [-1,1]：{targets.min().item()}～{targets.max().item()}",
        )

    print(f"dataset length：{len(dataset)}")
    print(f"sample ids：{batch['ids']}")
    print(f"features：{tuple(features.shape)}")
    print(f"segment_mask：{tuple(mask.shape)}")
    print(f"targets：{tuple(targets.shape)}")
    print(f"valid segments：{lengths.tolist()}")
    print(f"target range：{targets.min().item():.4f} ～ {targets.max().item():.4f}")
    print(
        f"dtype：features={features.dtype}，targets={targets.dtype}，mask={mask.dtype}"
    )
    print(f"[PASS] {name.upper()} DataLoader")


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    data_config = train_config["data"]

    datasets = {
        "mtg_full183": MTGTaggingDataset(dataset_config["mtg"], "train"),
        "mtg_genre87": MTGTaggingDataset(
            dataset_config["mtg"], "train", tag_set="genre"
        ),
        "mtg_instrument40": MTGTaggingDataset(
            dataset_config["mtg"], "train", tag_set="instrument"
        ),
        "mtg_mood56": MTGTaggingDataset(
            dataset_config["mtg"], "train", tag_set="moodtheme"
        ),
        "deam": MusicVADataset(dataset_config["deam"], "deam", "train"),
        "pmemo": MusicVADataset(dataset_config["pmemo"], "pmemo", "train"),
    }

    loaders = {
        name: build_music_dataloader(
            dataset,
            batch_size=int(data_config["batch_size"]),
            shuffle=False,
            num_workers=int(data_config["num_workers"]),
            pin_memory=bool(data_config["pin_memory"]),
        )
        for name, dataset in datasets.items()
    }

    print(f"项目根目录：{ROOT}")
    print(f"数据配置：{DATASET_CONFIG}")
    print(f"训练配置：{TRAIN_CONFIG}")
    print(f"batch_size：{data_config['batch_size']}")
    print(f"num_workers：{data_config['num_workers']}")
    print(f"pin_memory：{data_config['pin_memory']}")
    print("检查模式：只读取每个数据集的一个 batch，不训练模型")

    for name, dataset in datasets.items():
        check_batch(name, dataset, loaders[name], 183 if name.startswith("mtg") else 2)

    print("\n================ 最终结果 ================")
    print("[PASS] MTG 四种 manifest 与两个 VA Dataset/DataLoader 检查全部通过")


if __name__ == "__main__":
    main()
