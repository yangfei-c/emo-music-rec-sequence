"""
音乐实验配置
MTG 标签与 DEAM/PMEmo VA任务
"""

from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from msmmr.data import (
    MTGTaggingDataset,
    MusicVADataset,
    build_music_dataloader,
    domain_balanced_sampler,
)
from msmmr.losses import TagSubsetWeightedBCE
from msmmr.metrics import compute_va_metrics
from msmmr.training.utilities import get_device, load_yaml, set_seed
from msmmr.training.factories import (
    build_music_model,
    build_optimizer,
    build_tag_loss,
    build_va_loss,
)
from msmmr.training.engine import RunOptions, execute_experiment
from msmmr.training.tasks import (
    MTG_TASK,
    make_domain_balanced_va_metrics,
    make_mtg_subset_task,
    make_va_task,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG = ROOT / "configs/datasets.yaml"
TRAIN_CONFIG = ROOT / "configs/music_training.yaml"
SPLITS = ("train", "validation", "test")
DOMAINS = ("deam", "pmemo")
TAG_MODES = {
    "full183": {"tag_set": "all", "group": None, "output": "full183"},
    "genre87": {"tag_set": "genre", "group": "genre", "output": "genre87"},
    "instrument40": {
        "tag_set": "instrument",
        "group": "instrument",
        "output": "instrument40",
    },
    "mood56": {"tag_set": "moodtheme", "group": "mood", "output": "mood56"},
}
TAG_SLICES = {
    "genre": slice(0, 87),
    "instrument": slice(87, 127),
    "mood": slice(127, 183),
}


def _context() -> tuple[dict, dict, int, torch.device]:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    return (
        dataset_config,
        train_config,
        seed,
        get_device(str(train_config.get("device", "cpu"))),
    )


def _loaders(
    datasets: dict, data: dict, batch_size: int, sampler=None, drop_last: bool = False
) -> dict:
    return {
        split: build_music_dataloader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler if split == "train" else None,
            num_workers=int(data["num_workers"]),
            pin_memory=bool(data["pin_memory"]),
            drop_last=split == "train" and drop_last,
        )
        for split, dataset in datasets.items()
    }


def _subset_loss(
    dataset: MTGTaggingDataset, group: str, train_config: dict, device: torch.device
) -> TagSubsetWeightedBCE:
    label_slice = TAG_SLICES[group]
    positives = torch.zeros(label_slice.stop - label_slice.start)
    for _, _, indices in dataset.samples:
        for index in indices:
            if index not in range(label_slice.start, label_slice.stop):
                raise ValueError(f"{group} manifest 出现组外标签索引：{index}")
            positives[index - label_slice.start] += 1
    maximum = float(train_config["loss"]["tag_subset"].get("max_pos_weight", 20.0))
    weights = ((len(dataset) - positives) / positives.clamp_min(1)).clamp(1.0, maximum)
    return TagSubsetWeightedBCE(label_slice, weights).to(device)


def run_music_tagging(tag_mode: str, options: RunOptions):
    if tag_mode not in TAG_MODES:
        raise ValueError(f"未知 tag_mode：{tag_mode}；可选项：{', '.join(TAG_MODES)}")
    dataset_config, train_config, seed, device = _context()
    mode = TAG_MODES[tag_mode]
    tag_set, group = mode["tag_set"], mode["group"]
    datasets = {
        split: MTGTaggingDataset(dataset_config["mtg"], split, tag_set=tag_set)
        for split in SPLITS
    }
    data = train_config["data"]
    batch_size = int(data["tag_subset_batch_size"] if group else data["batch_size"])
    loaders = _loaders(datasets, data, batch_size)
    model = build_music_model(train_config, device)
    if group is None:
        task, criterion = MTG_TASK, build_tag_loss(train_config)
    else:
        task, criterion = make_mtg_subset_task(group), _subset_loss(
            datasets["train"], group, train_config, device
        )
    output_root = ROOT / "outputs/music/tagging/mtg" / mode["output"]
    return execute_experiment(
        output_root=output_root,
        options=options,
        seed=seed,
        task=task,
        dataset_config=dataset_config,
        train_config=train_config,
        run_config={
            "tag_mode": tag_mode,
            "tag_set": tag_set,
            "label_group": group,
            "dataset": dataset_config,
            "train": train_config,
        },
        datasets=datasets,
        loaders=loaders,
        model=model,
        criterion=criterion,
        optimizer=build_optimizer(model, train_config),
        device=device,
        setup_lines=(
            f"split=official split-0 | tag_mode={tag_mode} | tag_set={tag_set}",
        ),
    )


def _va_datasets(dataset_config: dict, experiment: str, seed: int):
    if experiment not in (*DOMAINS, "joint"):
        raise ValueError(f"未知 VA 实验：{experiment}")
    names = DOMAINS if experiment == "joint" else (experiment,)
    domain_sets = {
        name: {
            split: MusicVADataset(dataset_config[name], name, split) for split in SPLITS
        }
        for name in names
    }
    if experiment in DOMAINS:
        return (
            domain_sets[experiment],
            None,
            make_va_task(experiment),
            f"in-domain={experiment}",
        )
    datasets = {
        split: ConcatDataset([domain_sets[name][split] for name in DOMAINS])
        for split in SPLITS
    }
    evaluation_lengths = {
        sum(len(domain_sets[name][split]) for name in DOMAINS): tuple(
            len(domain_sets[name][split]) for name in DOMAINS
        )
        for split in ("validation", "test")
    }

    def joint_metrics(predictions, targets):
        lengths = evaluation_lengths.get(len(predictions))
        return (
            compute_va_metrics(predictions, targets)
            if lengths is None
            else make_domain_balanced_va_metrics(lengths, DOMAINS)(predictions, targets)
        )

    train_lengths = [len(domain_sets[name]["train"]) for name in DOMAINS]
    protocol = "joint=domain-balanced training sampler and equal-domain validation CCC"
    return (
        datasets,
        domain_balanced_sampler(train_lengths, seed),
        make_va_task("joint", joint_metrics),
        protocol,
    )


def run_music_emotion(
    experiment: str, options: RunOptions, run_name: str | None = None
):
    dataset_config, train_config, seed, device = _context()
    datasets, sampler, task, protocol = _va_datasets(dataset_config, experiment, seed)
    data = train_config["data"]
    batch_size = int(data.get("va_batch_size", data["batch_size"]))
    use_batch_ccc = float(train_config["loss"]["va"].get("ccc_weight", 0.0)) > 0
    loaders = _loaders(datasets, data, batch_size, sampler, use_batch_ccc)
    model = build_music_model(train_config, device)
    output_name = run_name or (
        "deam_pmemo_joint" if experiment == "joint" else experiment
    )
    return execute_experiment(
        output_root=ROOT / "outputs/music/va" / output_name,
        options=options,
        seed=seed,
        task=task,
        dataset_config=dataset_config,
        train_config=train_config,
        run_config={
            "experiment": experiment,
            "run_name": run_name,
            "protocol": protocol,
            "dataset": dataset_config,
            "train": train_config,
        },
        datasets=datasets,
        loaders=loaders,
        model=model,
        criterion=build_va_loss(train_config),
        optimizer=build_optimizer(model, train_config),
        device=device,
        setup_lines=(
            f"VA: y_norm=(y-5)/4 | {protocol}",
            f"batch={batch_size} | loss={train_config['loss']['va']['name']}",
        ),
    )
