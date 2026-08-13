import csv
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SPLITS = ("train", "validation", "test")
MTG_FILES = {
    "all": {split: f"autotagging-{split}.tsv" for split in SPLITS},
    "genre": {split: f"autotagging_genre-{split}.tsv" for split in SPLITS},
    "instrument": {
        split: f"autotagging_instrument-{split}.tsv" for split in SPLITS
    },
    "moodtheme": {
        split: f"autotagging_moodtheme-{split}.tsv" for split in SPLITS
    },
}


def segment_index(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"分段文件名错误：{path}") from error


def load_song_features(feature_dir: Path) -> torch.Tensor:
    """读取一首音乐的所有 MERT 分段并统一为 [S,12,768]。"""
    files = sorted(feature_dir.glob("segment_*.npy"), key=segment_index)
    if not files:
        raise FileNotFoundError(f"没有找到 MERT 特征：{feature_dir}")
    segments = []
    for path in files:
        feature = np.load(path, allow_pickle=False)
        if feature.shape == (1, 12, 768):
            feature = feature[0]
        if feature.shape != (12, 768):
            raise ValueError(f"MERT 形状错误：{path}，实际为 {feature.shape}")
        if not np.isfinite(feature).all():
            raise ValueError(f"MERT 特征含 NaN/Inf：{path}")
        segments.append(torch.from_numpy(feature.astype(np.float32, copy=False)))
    return torch.stack(segments)


class MTGTaggingDataset(Dataset):
    """MTG-Jamendo official split-0；读取 full183 或单一标签组 manifest。"""

    def __init__(self, config: dict, split: str, tag_set: str = "all") -> None:
        if tag_set not in MTG_FILES or split not in MTG_FILES[tag_set]:
            raise ValueError(f"未知 MTG tag_set/split：{tag_set}/{split}")
        feature_root = Path(config["feature_dir"])
        split_file = Path(config["split_dir"]) / MTG_FILES[tag_set][split]
        tags = (
            np.load(config["tag_vocabulary_file"], allow_pickle=False)
            .reshape(-1)
            .astype(str)
        )
        if len(tags) != 183 or len(set(tags)) != 183:
            raise ValueError("MTG tag vocabulary 必须包含 183 个唯一标签")
        tag_to_index = {tag: index for index, tag in enumerate(tags)}

        self.tag_set, self.samples = tag_set, []
        with split_file.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader)
            for row in reader:
                relative_path = Path(row[3].replace("\\", "/")).with_suffix("")
                tag_indices = [tag_to_index[tag] for tag in row[5:]]
                self.samples.append((row[0], feature_root / relative_path, tag_indices))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        song_id, feature_dir, tag_indices = self.samples[index]
        target = torch.zeros(183, dtype=torch.float32)
        target[tag_indices] = 1.0
        return {
            "id": song_id,
            "features": load_song_features(feature_dir),
            "target": target,
            "dataset": "mtg",
        }


class MusicVADataset(Dataset):
    """DEAM 或 PMEmo 静态 VA；内部统一到 [-1,1]。"""

    def __init__(self, config: dict, dataset_name: str, split: str) -> None:
        feature_root = Path(config["feature_dir"])
        split_file = Path(config["split_dir"]) / f"{split}.txt"
        annotations = {}
        annotation_file = Path(config["static_annotation_file"])
        with annotation_file.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                song_id = str(int(float(row["song_id"])))
                annotations[song_id] = (
                    float(row["valence_mean"]),
                    float(row["arousal_mean"]),
                )
        song_ids = [
            str(int(float(line)))
            for line in split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.dataset_name, self.samples = dataset_name, []
        for song_id in song_ids:
            if song_id not in annotations:
                raise KeyError(f"{dataset_name}/{split} 缺少标注：{song_id}")
            valence, arousal = annotations[song_id]
            normalized_va = ((valence - 5.0) / 4.0, (arousal - 5.0) / 4.0)
            self.samples.append((song_id, feature_root / song_id, normalized_va))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        song_id, feature_dir, target = self.samples[index]
        return {
            "id": song_id,
            "features": load_song_features(feature_dir),
            "target": torch.tensor(target, dtype=torch.float32),
            "dataset": self.dataset_name,
        }


def collate_music_batch(samples: list[dict]) -> dict:
    feature_list = [sample["features"] for sample in samples]
    lengths = torch.tensor(
        [feature.shape[0] for feature in feature_list], dtype=torch.long
    )
    features = pad_sequence(feature_list, batch_first=True, padding_value=0.0)
    segment_mask = torch.arange(features.shape[1])[None, :] < lengths[:, None]
    return {
        "ids": [sample["id"] for sample in samples],
        "features": features,
        "segment_mask": segment_mask,
        "lengths": lengths,
        "targets": torch.stack([sample["target"] for sample in samples]),
        "dataset": [sample["dataset"] for sample in samples],
    }


def build_music_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    sampler=None,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if sampler is not None and shuffle:
        raise ValueError("sampler 与 shuffle=True 不能同时使用")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_music_batch,
        generator=generator,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def domain_balanced_sampler(
    lengths: list[int] | tuple[int, ...], seed: int
) -> WeightedRandomSampler:
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError(f"域样本数必须均为正数：{lengths}")
    domains = len(lengths)
    weights = torch.tensor(
        [1.0 / (domains * length) for length in lengths for _ in range(length)],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=domains * max(lengths),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
