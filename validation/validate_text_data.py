import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.data import build_text_dataloader, build_text_datasets, load_text_va_records
from msmmr.training.utilities import load_yaml
from validation.helpers import require


DATASET_CONFIG = ROOT / "configs" / "datasets.yaml"
TRAIN_CONFIG = ROOT / "configs" / "text_training.yaml"
EXPECTED_SOURCES = {
    "CVAS": 2583,
    "CVAT": 2971,
    "CVAI": 1465,
    "EmoBank": 10062,
    "Facebook": 2895,
}
EXPECTED_SPLITS = {"train": 15807, "validation": 1974, "test": 1980}


def main() -> None:
    dataset_config, train_config = load_yaml(DATASET_CONFIG), load_yaml(TRAIN_CONFIG)
    require("text_va" in dataset_config, "datasets.yaml 缺少 text_va")
    config = dataset_config["text_va"]
    records, audit = load_text_va_records(config)

    require(
        audit["raw_records"] == 19976,
        f"原始记录数为 {audit['raw_records']}，预期 19976",
    )
    require(
        audit["raw_by_source"] == EXPECTED_SOURCES,
        f"来源计数不一致：{audit['raw_by_source']}",
    )
    require(
        audit["invalid_records"] == 2,
        f"无效记录数为 {audit['invalid_records']}，预期 2",
    )
    require(
        audit["unique_records"] == 19761,
        f"去重后记录数为 {audit['unique_records']}，预期 19761",
    )
    require(
        audit["conflicting_groups"] == 103,
        f"冲突标注组为 {audit['conflicting_groups']}，预期 103",
    )

    datasets = build_text_datasets(config)
    id_sets = {
        split: {sample["id"] for sample in dataset.samples}
        for split, dataset in datasets.items()
    }
    require(
        all(len(datasets[split]) == count for split, count in EXPECTED_SPLITS.items()),
        "local fixed split 数量不一致",
    )
    require(
        not (
            id_sets["train"] & id_sets["validation"]
            or id_sets["train"] & id_sets["test"]
            or id_sets["validation"] & id_sets["test"]
        ),
        "划分间存在重复文本 ID",
    )
    require(
        len(set().union(*id_sets.values())) == len(records),
        "三个划分未完整覆盖去重后的记录",
    )

    for split, dataset in datasets.items():
        languages, sources = Counter(
            sample["language"] for sample in dataset.samples
        ), Counter(sample["source"] for sample in dataset.samples)
        require(set(languages) == {"zh", "en"}, f"{split} 语言不完整：{languages}")
        require(set(sources) == set(EXPECTED_SOURCES), f"{split} 来源不完整：{sources}")
        targets = torch.tensor(
            [sample["target"] for sample in dataset.samples], dtype=torch.float32
        )
        require(torch.isfinite(targets).all().item(), f"{split} target 含 NaN/Inf")
        require(
            targets.min().item() >= -1.0 and targets.max().item() <= 1.0,
            f"{split} target 超出 [-1,1]",
        )

    data_config = train_config["data"]
    loader = build_text_dataloader(
        datasets["train"],
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=int(data_config["num_workers"]),
        pin_memory=bool(data_config["pin_memory"]),
    )
    batch = next(iter(loader))
    require(
        len(batch["texts"]) == int(data_config["batch_size"]), "文本 batch size 错误"
    )
    require(
        all(isinstance(text, str) and text for text in batch["texts"]),
        "batch 中存在无效文本",
    )
    require(
        batch["targets"].shape == (int(data_config["batch_size"]), 2),
        f"target 形状错误：{tuple(batch['targets'].shape)}",
    )
    require(
        batch["targets"].dtype == torch.float32,
        f"target dtype 错误：{batch['targets'].dtype}",
    )
    require(
        batch["duplicate_counts"].dtype == torch.long, "duplicate_counts dtype 错误"
    )

    print(f"项目根目录：{ROOT}")
    print(
        "划分说明：local fixed split；按 language/source 分层、文本去重后固定哈希排序，不是 official split"
    )
    print(
        f"原始记录：{audit['raw_records']}；有效：{audit['valid_raw_records']}；"
        f"无效：{audit['invalid_records']}"
    )
    print(
        f"去重后：{audit['unique_records']}；"
        f"合并重复行：{audit['merged_duplicate_rows']}；"
        f"冲突标注组：{audit['conflicting_groups']}"
    )
    print(f"来源：{audit['raw_by_source']}")
    print(
        f"划分：{', '.join(f'{name}={len(dataset)}' for name, dataset in datasets.items())}"
    )
    print(
        f"batch texts：{len(batch['texts'])}；"
        f"targets：{tuple(batch['targets'].shape)}；"
        f"范围：{batch['targets'].min().item():.4f}～"
        f"{batch['targets'].max().item():.4f}"
    )
    print(
        f"batch languages：{Counter(batch['languages'])}；"
        f"sources：{Counter(batch['sources'])}"
    )
    print("[PASS] 文本 VA 配置、清洗、固定划分、Dataset 和 DataLoader 检查全部通过")


if __name__ == "__main__":
    main()
