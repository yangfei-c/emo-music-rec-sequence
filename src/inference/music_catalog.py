from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data import MTGDataset, build_dataloader
from src.training.checkpoint import load_model_state, resolve_checkpoint_path
from src.training.common import get_device, load_yaml
from src.training.components import build_music_model


ROOT = Path(__file__).resolve().parents[2]

DATASET_CONFIG_PATH = ROOT / "configs" / "dataset.yaml"
DEFAULT_CATALOG_CONFIG_PATH = ROOT / "configs" / "local_catalog.yaml"

TAG_OUTPUT_ROOT = ROOT / "outputs" / "music" / "tagging" / "mtg" / "full183"
VA_OUTPUT_ROOT = ROOT / "outputs" / "music" / "va" / "deam_pmemo_joint"

SPLITS = ("train", "validation", "test")

EXPECTED_TAG_TASK = "music_mtg_full183"
EXPECTED_VA_TASK = "music_va_joint"

GENRE_SLICE = slice(0, 87)
INSTRUMENT_SLICE = slice(87, 127)
MOOD_SLICE = slice(127, 183)

VA_DIM = 2
GENRE_DIM = 87
INSTRUMENT_DIM = 40
MOOD_DIM = 56
TAG_DIM = 183
EMBEDDING_DIM = 256


def _resolve_checkpoint(
    value: str | Path,
    output_root: Path,
) -> Path:
    """
    支持：
    1. latest
    2. best.pt 完整路径
    3. 某次运行目录路径
    4. 相对于仓库根目录的路径
    """
    if str(value).lower() == "latest":
        path = resolve_checkpoint_path(
            "latest",
            output_root,
            "best.pt",
        )
        if path is None:
            raise FileNotFoundError(
                f"没有找到 checkpoint：{output_root}/*/best.pt"
            )
        return path

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if path.is_dir():
        path = path / "best.pt"

    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint 不存在：{path}"
        )

    return path


def _load_frozen_model(
    checkpoint_path: Path,
    expected_task: str,
    device: torch.device,
):
    """
    使用 checkpoint 自己保存的 train_config 重建模型，
    再加载对应 best.pt 权重。
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    actual_task = checkpoint.get("task_name")

    if actual_task != expected_task:
        raise ValueError(
            "checkpoint 任务不匹配："
            f"expected={expected_task}, "
            f"actual={actual_task}, "
            f"path={checkpoint_path}"
        )

    train_config = checkpoint.get("train_config")

    if not isinstance(train_config, dict):
        raise ValueError(
            f"checkpoint 缺少有效 train_config：{checkpoint_path}"
        )

    model = build_music_model(
        train_config,
        device,
    )

    load_model_state(
        model,
        checkpoint,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def _build_mtg_loaders(
    dataset_config: dict,
    inference_config: dict,
) -> tuple[dict, dict[str, int]]:
    """
    使用现有 MTGDataset 和 build_dataloader，
    读取 full183 对应的 official split-0。
    """
    if "mtg" not in dataset_config:
        raise KeyError(
            "configs/dataset.yaml 中缺少 mtg 配置"
        )

    batch_size = int(
        inference_config.get("batch_size", 32)
    )
    num_workers = int(
        inference_config.get("num_workers", 4)
    )
    pin_memory = bool(
        inference_config.get("pin_memory", True)
    )

    if batch_size <= 0:
        raise ValueError(
            f"batch_size 必须大于 0，实际为 {batch_size}"
        )

    if num_workers < 0:
        raise ValueError(
            f"num_workers 不能小于 0，实际为 {num_workers}"
        )

    datasets = {
        split: MTGDataset(
            dataset_config["mtg"],
            split=split,
            tag_set="all",
        )
        for split in SPLITS
    }

    loaders = {
        split: build_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler=None,
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }

    lengths = {
        split: len(dataset)
        for split, dataset in datasets.items()
    }

    return loaders, lengths


def _infer_mtg(
    loaders: dict,
    va_model,
    tag_model,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    同一个 MTG batch 分别通过：

    VA joint checkpoint:
        -> pred_va
        -> va_embedding

    MTG full183 checkpoint:
        -> tag_prob
        -> tag_embedding

    注意：
    不使用 VA 模型的 tag head；
    不使用 Tag 模型的 VA head。
    """
    song_ids: list[str] = []
    split_names: list[str] = []

    pred_va_batches: list[np.ndarray] = []
    va_embedding_batches: list[np.ndarray] = []

    tag_prob_batches: list[np.ndarray] = []
    tag_embedding_batches: list[np.ndarray] = []

    non_blocking = device.type == "cuda"

    with torch.inference_mode():

        for split in SPLITS:

            loader = loaders[split]

            print(
                f"[INFERENCE] split={split} | "
                f"songs={len(loader.dataset)} | "
                f"batches={len(loader)}",
                flush=True,
            )

            for batch_index, batch in enumerate(
                loader,
                start=1,
            ):

                features = batch["features"].to(
                    device,
                    non_blocking=non_blocking,
                )

                segment_mask = batch["segment_mask"].to(
                    device,
                    non_blocking=non_blocking,
                )

                # -------------------------
                # 1. Joint VA model
                # -------------------------

                va_output = va_model(
                    features,
                    segment_mask,
                )

                va_predictions = (
                    va_output["va_predictions"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                va_embedding = (
                    va_output["song_embedding"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                # -------------------------
                # 2. MTG full183 model
                # -------------------------

                tag_output = tag_model(
                    features,
                    segment_mask,
                )

                tag_prob = (
                    torch.sigmoid(
                        tag_output["tag_logits"]
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                tag_embedding = (
                    tag_output["song_embedding"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                # -------------------------
                # 3. Metadata
                # -------------------------

                batch_ids = [
                    str(song_id)
                    for song_id in batch["ids"]
                ]

                song_ids.extend(batch_ids)

                split_names.extend(
                    [split] * len(batch_ids)
                )

                pred_va_batches.append(
                    va_predictions
                )

                va_embedding_batches.append(
                    va_embedding
                )

                tag_prob_batches.append(
                    tag_prob
                )

                tag_embedding_batches.append(
                    tag_embedding
                )

                if (
                    batch_index == 1
                    or batch_index % 100 == 0
                    or batch_index == len(loader)
                ):
                    print(
                        f"[{split.upper()}] "
                        f"batch={batch_index}/{len(loader)}",
                        flush=True,
                    )

    pred_va = np.concatenate(
        pred_va_batches,
        axis=0,
    )

    va_embedding = np.concatenate(
        va_embedding_batches,
        axis=0,
    )

    tag_prob = np.concatenate(
        tag_prob_batches,
        axis=0,
    )

    tag_embedding = np.concatenate(
        tag_embedding_batches,
        axis=0,
    )

    genre_prob = tag_prob[:, GENRE_SLICE]
    instrument_prob = tag_prob[:, INSTRUMENT_SLICE]
    mood_prob = tag_prob[:, MOOD_SLICE]

    return {
        "song_id": np.asarray(
            song_ids,
            dtype=str,
        ),
        "split": np.asarray(
            split_names,
            dtype=str,
        ),

        "pred_va": pred_va,

        "genre_prob": genre_prob,
        "instrument_prob": instrument_prob,
        "mood_prob": mood_prob,
        "tag_prob": tag_prob,

        "va_embedding": va_embedding,
        "tag_embedding": tag_embedding,
    }


def _validate_catalog(
    catalog: dict[str, np.ndarray],
    expected_lengths: dict[str, int],
) -> dict[str, object]:
    """
    Catalog 最终验收。

    注意：
    pred_va 只检查范围，
    不做 clip。
    """
    required = {
        "song_id",
        "split",
        "pred_va",
        "genre_prob",
        "instrument_prob",
        "mood_prob",
        "tag_prob",
        "va_embedding",
        "tag_embedding",
    }

    missing = required - catalog.keys()

    if missing:
        raise ValueError(
            "Catalog 缺少字段："
            + ", ".join(sorted(missing))
        )

    n = len(catalog["song_id"])

    expected_total = sum(
        expected_lengths.values()
    )

    if n != expected_total:
        raise ValueError(
            "歌曲总数错误："
            f"catalog={n}, "
            f"dataset={expected_total}"
        )

    if catalog["split"].shape != (n,):
        raise ValueError(
            f"split shape 错误：{catalog['split'].shape}"
        )

    expected_shapes = {
        "pred_va": (n, VA_DIM),
        "genre_prob": (n, GENRE_DIM),
        "instrument_prob": (
            n,
            INSTRUMENT_DIM,
        ),
        "mood_prob": (n, MOOD_DIM),
        "tag_prob": (n, TAG_DIM),
        "va_embedding": (
            n,
            EMBEDDING_DIM,
        ),
        "tag_embedding": (
            n,
            EMBEDDING_DIM,
        ),
    }

    for name, expected_shape in expected_shapes.items():

        actual_shape = catalog[name].shape

        if actual_shape != expected_shape:
            raise ValueError(
                f"{name} shape 错误："
                f"expected={expected_shape}, "
                f"actual={actual_shape}"
            )

    # -------------------------
    # Song ID
    # -------------------------

    unique_ids = len(
        set(catalog["song_id"].tolist())
    )

    duplicate_count = n - unique_ids

    if duplicate_count != 0:
        raise ValueError(
            f"存在重复 song_id：{duplicate_count}"
        )

    if np.any(
        np.char.str_len(catalog["song_id"]) == 0
    ):
        raise ValueError(
            "存在空 song_id"
        )

    # -------------------------
    # Split counts
    # -------------------------

    actual_split_counts = {}

    for split in SPLITS:

        count = int(
            np.sum(
                catalog["split"] == split
            )
        )

        actual_split_counts[split] = count

        expected_count = int(
            expected_lengths[split]
        )

        if count != expected_count:
            raise ValueError(
                f"{split} 数量错误："
                f"expected={expected_count}, "
                f"actual={count}"
            )

    # -------------------------
    # Finite values
    # -------------------------

    numeric_fields = (
        "pred_va",
        "genre_prob",
        "instrument_prob",
        "mood_prob",
        "tag_prob",
        "va_embedding",
        "tag_embedding",
    )

    for name in numeric_fields:

        if not np.isfinite(
            catalog[name]
        ).all():

            raise ValueError(
                f"{name} 含 NaN/Inf"
            )

    # -------------------------
    # Tag probability
    # -------------------------

    tag_prob = catalog["tag_prob"]

    if np.any(tag_prob < 0.0) or np.any(
        tag_prob > 1.0
    ):
        raise ValueError(
            "tag_prob 超出 [0,1]"
        )

    # -------------------------
    # VA range
    # 这里只统计，不裁剪
    # -------------------------

    pred_va = catalog["pred_va"]

    va_min = pred_va.min(
        axis=0
    )

    va_max = pred_va.max(
        axis=0
    )

    outside_mask = (
        (pred_va < -1.0)
        | (pred_va > 1.0)
    )

    outside_value_count = int(
        outside_mask.sum()
    )

    outside_song_count = int(
        outside_mask.any(axis=1).sum()
    )

    return {
        "songs": n,
        "duplicate_song_id": duplicate_count,
        "split_counts": actual_split_counts,
        "valence_min": float(
            va_min[0]
        ),
        "valence_max": float(
            va_max[0]
        ),
        "arousal_min": float(
            va_min[1]
        ),
        "arousal_max": float(
            va_max[1]
        ),
        "outside_va_value_count":
            outside_value_count,
        "outside_va_song_count":
            outside_song_count,
    }


def _resolve_output_path(
    value: str | Path,
) -> Path:
    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if path.suffix.lower() != ".npz":
        raise ValueError(
            f"Catalog 输出必须是 .npz：{path}"
        )

    return path


def _save_catalog(
    catalog: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,

        song_id=catalog["song_id"],
        split=catalog["split"],

        pred_va=catalog["pred_va"],

        genre_prob=catalog["genre_prob"],
        instrument_prob=catalog[
            "instrument_prob"
        ],
        mood_prob=catalog["mood_prob"],
        tag_prob=catalog["tag_prob"],

        va_embedding=catalog[
            "va_embedding"
        ],
        tag_embedding=catalog[
            "tag_embedding"
        ],
    )


def _print_summary(
    *,
    tag_checkpoint: Path,
    va_checkpoint: Path,
    output_path: Path,
    summary: dict[str, object],
) -> None:

    print()
    print(
        "========== MTG MUSIC CATALOG =========="
    )

    print(
        f"Tag checkpoint : {tag_checkpoint}"
    )

    print(
        f"VA checkpoint  : {va_checkpoint}"
    )

    print(
        f"Songs          : {summary['songs']}"
    )

    for split, count in summary[
        "split_counts"
    ].items():

        print(
            f"{split:<15}: {count}"
        )

    print(
        "pred_va        : "
        f"({summary['songs']}, 2)"
    )

    print(
        "tag_prob       : "
        f"({summary['songs']}, 183)"
    )

    print(
        "va_embedding   : "
        f"({summary['songs']}, 256)"
    )

    print(
        "tag_embedding  : "
        f"({summary['songs']}, 256)"
    )

    print(
        "Valence range  : "
        f"[{summary['valence_min']:.6f}, "
        f"{summary['valence_max']:.6f}]"
    )

    print(
        "Arousal range  : "
        f"[{summary['arousal_min']:.6f}, "
        f"{summary['arousal_max']:.6f}]"
    )

    print(
        "VA values outside [-1,1] : "
        f"{summary['outside_va_value_count']}"
    )

    print(
        "Songs outside [-1,1]     : "
        f"{summary['outside_va_song_count']}"
    )

    print(
        "Duplicate song_id        : "
        f"{summary['duplicate_song_id']}"
    )

    print(
        f"Saved          : {output_path}"
    )

    print(
        "======================================="
    )


def build_mtg_music_catalog(
    config_path: str | Path
    = DEFAULT_CATALOG_CONFIG_PATH,
) -> Path:
    """
    构建完整 MTG Music Catalog。

    最终保存字段：

        song_id
        split

        pred_va

        genre_prob
        instrument_prob
        mood_prob
        tag_prob

        va_embedding
        tag_embedding

    不保存 true_va。
    不对 pred_va 做 clip。
    """
    config_path = Path(
        config_path
    ).expanduser()

    if not config_path.is_absolute():
        config_path = (
            ROOT / config_path
        ).resolve()

    catalog_config = load_yaml(
        config_path
    )

    dataset_config = load_yaml(
        DATASET_CONFIG_PATH
    )

    device = get_device(
        str(
            catalog_config.get(
                "device",
                "cuda",
            )
        )
    )

    checkpoint_config = catalog_config.get(
        "checkpoints",
        {},
    )

    tag_checkpoint_value = (
        checkpoint_config.get(
            "tag",
            "latest",
        )
    )

    va_checkpoint_value = (
        checkpoint_config.get(
            "va",
            "latest",
        )
    )

    tag_checkpoint = _resolve_checkpoint(
        tag_checkpoint_value,
        TAG_OUTPUT_ROOT,
    )

    va_checkpoint = _resolve_checkpoint(
        va_checkpoint_value,
        VA_OUTPUT_ROOT,
    )

    print(
        f"[DEVICE] {device}",
        flush=True,
    )

    print(
        f"[TAG] {tag_checkpoint}",
        flush=True,
    )

    print(
        f"[VA] {va_checkpoint}",
        flush=True,
    )

    # -------------------------
    # Frozen models
    # -------------------------

    tag_model = _load_frozen_model(
        checkpoint_path=tag_checkpoint,
        expected_task=EXPECTED_TAG_TASK,
        device=device,
    )

    va_model = _load_frozen_model(
        checkpoint_path=va_checkpoint,
        expected_task=EXPECTED_VA_TASK,
        device=device,
    )

    # -------------------------
    # MTG full183 loaders
    # -------------------------

    loaders, dataset_lengths = (
        _build_mtg_loaders(
            dataset_config,
            catalog_config.get(
                "inference",
                {},
            ),
        )
    )

    print(
        "[MTG] "
        + " | ".join(
            f"{split}={dataset_lengths[split]}"
            for split in SPLITS
        ),
        flush=True,
    )

    # -------------------------
    # Inference
    # -------------------------

    catalog = _infer_mtg(
        loaders=loaders,
        va_model=va_model,
        tag_model=tag_model,
        device=device,
    )

    # -------------------------
    # Validation
    # -------------------------

    summary = _validate_catalog(
        catalog,
        dataset_lengths,
    )

    # -------------------------
    # Save
    # -------------------------

    output_config = catalog_config.get(
        "output",
        {},
    )

    output_path = _resolve_output_path(
        output_config.get(
            "path",
            "outputs/inference/mtg/"
            "mtg_music_catalog.npz",
        )
    )

    _save_catalog(
        catalog,
        output_path,
    )

    _print_summary(
        tag_checkpoint=tag_checkpoint,
        va_checkpoint=va_checkpoint,
        output_path=output_path,
        summary=summary,
    )

    return output_path