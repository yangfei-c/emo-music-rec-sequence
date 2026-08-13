import csv
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.training.utilities import load_yaml
from validation.helpers import require


CONFIG_FILE = ROOT / "configs" / "datasets.yaml"

MTG_SPLITS = {
    "train": ("autotagging-train.tsv", 32859),
    "validation": ("autotagging-validation.tsv", 11101),
    "test": ("autotagging-test.tsv", 11565),
}
MTG_SUBSET_SPLITS = {
    "genre": ({"train": 32572, "validation": 11043, "test": 11479}, "genre---"),
    "instrument": ({"train": 14395, "validation": 5466, "test": 5115}, "instrument---"),
    "moodtheme": ({"train": 9949, "validation": 3802, "test": 4231}, "mood/theme---"),
}

VA_INFO = {
    "deam": {
        "official_audio": 1802,
        "annotations": 1802,
        "splits": {"train": 1261, "validation": 271, "test": 270},
    },
    "pmemo": {
        "official_audio": 794,
        "annotations": 767,
        "splits": {"train": 536, "validation": 116, "test": 115},
    },
}

EXPECTED_TAGS = {
    "genre---": 87,
    "instrument---": 40,
    "mood/theme---": 56,
}

VA_COLUMNS = {
    "song_id",
    "valence_mean",
    "valence_std",
    "arousal_mean",
    "arousal_std",
}

VALID_MERT_SHAPES = {(12, 768), (1, 12, 768)}
RANDOM_SEED = 42


def load_config() -> dict:
    return load_yaml(CONFIG_FILE)


def get_audio_ids(audio_dir: Path) -> set[str]:
    """以相对路径且去掉扩展名后的字符串作为歌曲 ID。"""
    return {
        path.relative_to(audio_dir).with_suffix("").as_posix()
        for path in audio_dir.rglob("*.mp3")
    }


def get_feature_dirs(feature_dir: Path) -> dict[str, Path]:
    """通过 segment_0.npy 确定每首歌曲的特征目录。"""
    return {
        path.parent.relative_to(feature_dir).as_posix(): path.parent
        for path in feature_dir.rglob("segment_0.npy")
    }


def check_mert(name: str, config: dict, expected_audio: int) -> None:
    """检查音频—特征目录对应，并随机读取一个 MERT 分段。"""
    audio_dir = Path(config["audio_dir"])
    feature_dir = Path(config["feature_dir"])

    require(audio_dir.is_dir(), f"{name} 音频目录不存在：{audio_dir}")
    require(feature_dir.is_dir(), f"{name} 特征目录不存在：{feature_dir}")

    audio_ids = get_audio_ids(audio_dir)
    feature_dirs = get_feature_dirs(feature_dir)

    require(
        len(audio_ids) == expected_audio,
        f"{name} 音频数量为 {len(audio_ids)}，预期为 {expected_audio}",
    )
    require(
        set(feature_dirs) == audio_ids,
        f"{name} 音频与 MERT 特征目录不完全对应",
    )

    rng = random.Random(f"{RANDOM_SEED}-{name}")
    sample_id = rng.choice(sorted(feature_dirs))
    segments = sorted(feature_dirs[sample_id].glob("segment_*.npy"))

    require(segments, f"{name} 抽样特征目录为空：{sample_id}")

    sample_file = rng.choice(segments)
    feature = np.load(sample_file, mmap_mode="r", allow_pickle=False)

    require(
        feature.shape in VALID_MERT_SHAPES,
        f"{name} 抽样形状错误：{feature.shape}",
    )
    require(
        feature.dtype == np.float32,
        f"{name} 抽样 dtype 错误：{feature.dtype}",
    )

    print(
        f"[PASS] 音频/MERT：{len(audio_ids)}/{len(feature_dirs)}；"
        f"抽样 {sample_id}/{sample_file.name}，"
        f"shape={feature.shape}，dtype={feature.dtype}"
    )


def check_mtg(config: dict) -> None:
    print("\n================ MTG ================")

    split_dir = Path(config["split_dir"])
    tag_file = Path(config["tag_vocabulary_file"])

    require(split_dir.is_dir(), f"MTG split_dir 不存在：{split_dir}")
    require(tag_file.is_file(), f"MTG 标签词表不存在：{tag_file}")

    rows = []

    for split, (filename, expected) in MTG_SPLITS.items():
        path = split_dir / filename
        require(path.is_file(), f"缺少 {path}")

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file, delimiter="\t")
            next(reader, None)
            split_rows = list(reader)

        require(
            len(split_rows) == expected,
            f"MTG {split} 数量为 {len(split_rows)}，预期为 {expected}",
        )
        rows.extend(split_rows)

    track_ids = {row[0] for row in rows}
    require(
        len(track_ids) == 55525,
        f"MTG 正式划分歌曲数为 {len(track_ids)}，预期为 55525",
    )

    tags = {tag for row in rows for tag in row[5:]}
    counts = {
        prefix: sum(tag.startswith(prefix) for tag in tags) for prefix in EXPECTED_TAGS
    }

    for prefix, expected in EXPECTED_TAGS.items():
        require(
            counts[prefix] == expected,
            f"{prefix} 标签数为 {counts[prefix]}，预期为 {expected}",
        )

    require(len(tags) == 183, f"MTG 总标签数为 {len(tags)}，预期为 183")

    tag_list = (
        np.load(
            tag_file,
            allow_pickle=False,
        )
        .reshape(-1)
        .astype(str)
    )

    require(len(tag_list) == 183, "tag_list.npy 数量不是 183")
    require(set(tag_list) == tags, "tag_list.npy 与 TSV 标签不一致")

    split_counts = "/".join(
        str(MTG_SPLITS[name][1]) for name in ("train", "validation", "test")
    )

    print(f"[PASS] 正式划分：{split_counts}，" f"合计 {len(track_ids)}")
    print(
        "[PASS] 标签："
        f"genre={counts['genre---']}，"
        f"instrument={counts['instrument---']}，"
        f"mood/theme={counts['mood/theme---']}，"
        f"total={len(tags)}"
    )
    print("[PASS] tag_list.npy 与正式 TSV 一致")

    for group, (expected_splits, prefix) in MTG_SUBSET_SPLITS.items():
        actual = []
        for split, expected in expected_splits.items():
            path = split_dir / f"autotagging_{group}-{split}.tsv"
            require(path.is_file(), f"缺少 {path}")
            with path.open("r", encoding="utf-8", newline="") as file:
                reader = csv.reader(file, delimiter="\t")
                next(reader, None)
                subset_rows = list(reader)
            require(
                len(subset_rows) == expected,
                f"MTG {group}/{split} 数量为 {len(subset_rows)}，预期为 {expected}",
            )
            require(
                all(tag.startswith(prefix) for row in subset_rows for tag in row[5:]),
                f"MTG {group}/{split} 含组外标签",
            )
            require(
                {row[0] for row in subset_rows} <= track_ids,
                f"MTG {group}/{split} 含 full183 manifest 之外的歌曲",
            )
            actual.append(len(subset_rows))
        print(f"[PASS] {group} 子任务划分：{'/'.join(map(str, actual))}")

    # MTG 本地完整预处理集合为 55,609，
    # 正式训练仍只使用上面的 55,525 首。
    check_mert("mtg", config, expected_audio=55609)


def read_split_ids(split_dir: Path, expected: dict) -> set[int]:
    """读取当前参考划分；不打印划分之间的重复检查。"""
    split_sets = {}

    for split, count in expected.items():
        path = split_dir / f"{split}.txt"
        require(path.is_file(), f"缺少 {path}")

        ids = {
            int(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        require(
            len(ids) == count,
            f"{path.name} 数量为 {len(ids)}，预期为 {count}",
        )
        split_sets[split] = ids

    union = set().union(*split_sets.values())

    # 静默检查，不输出每一对划分的 overlap。
    require(
        sum(map(len, split_sets.values())) == len(union),
        "参考划分之间存在重复 song_id",
    )

    return union


def read_va_annotations(path: Path) -> tuple[set[int], dict[str, tuple[float, float]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])

        require(VA_COLUMNS <= columns, f"{path.name} 字段不完整")

        ids = set()
        values = {
            "valence_mean": [],
            "arousal_mean": [],
            "valence_std": [],
            "arousal_std": [],
        }

        for row in reader:
            ids.add(int(float(row["song_id"])))

            for column in values:
                values[column].append(float(row[column]))

    require(
        all(
            value >= 0
            for column in ("valence_std", "arousal_std")
            for value in values[column]
        ),
        f"{path.name} 中存在负标准差",
    )

    ranges = {
        column: (min(column_values), max(column_values))
        for column, column_values in values.items()
    }

    return ids, ranges


def check_va_dataset(name: str, config: dict) -> None:
    info = VA_INFO[name]
    print(f"\n================ {name.upper()} ================")

    split_dir = Path(config["split_dir"])
    annotation_file = Path(config["static_annotation_file"])

    require(split_dir.is_dir(), f"{name} split_dir 不存在：{split_dir}")
    require(
        annotation_file.is_file(),
        f"{name} 静态标注文件不存在：{annotation_file}",
    )

    annotation_ids, ranges = read_va_annotations(annotation_file)

    require(
        len(annotation_ids) == info["annotations"],
        f"{name} 静态标注数为 {len(annotation_ids)}，" f"预期为 {info['annotations']}",
    )

    split_ids = read_split_ids(split_dir, info["splits"])
    require(
        split_ids == annotation_ids,
        f"{name} 参考划分 ID 与静态标注 ID 不一致",
    )

    split_counts = "/".join(
        str(info["splits"][split]) for split in ("train", "validation", "test")
    )

    print(f"[PASS] 当前静态标注：{len(annotation_ids)}；" f"参考划分：{split_counts}")
    print(
        "[PASS] VA 范围："
        f"Valence={ranges['valence_mean'][0]:.4f}"
        f"～{ranges['valence_mean'][1]:.4f}；"
        f"Arousal={ranges['arousal_mean'][0]:.4f}"
        f"～{ranges['arousal_mean'][1]:.4f}"
    )
    print(
        "       标准差范围："
        f"Valence={ranges['valence_std'][0]:.4f}"
        f"～{ranges['valence_std'][1]:.4f}；"
        f"Arousal={ranges['arousal_std'][0]:.4f}"
        f"～{ranges['arousal_std'][1]:.4f}"
    )

    check_mert(
        name,
        config,
        expected_audio=info["official_audio"],
    )


def main() -> None:
    config = load_config()

    print(f"项目根目录：{ROOT}")
    print(f"配置文件：{CONFIG_FILE}")
    print("检查模式：只读，不修改任何数据")

    for name in ("mtg", "deam", "pmemo"):
        require(name in config, f"datasets.yaml 中缺少 {name}")

    check_mtg(config["mtg"])
    check_va_dataset("deam", config["deam"])
    check_va_dataset("pmemo", config["pmemo"])

    print("\n================ 最终结果 ================")
    print("[PASS] 三个数据集的数量、标签、VA 标注和 MERT 检查全部通过")


if __name__ == "__main__":
    main()
