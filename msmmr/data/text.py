import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset


SPLITS = ("train", "validation", "test")


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha1(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def _record_id(language: str, text: str) -> str:
    return f"{language}-{hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]}"


def load_text_va_records(config: dict) -> tuple[list[dict], dict]:
    """读取、校验并按文本合并重复项；原始标签 [0,1] 映射到 [-1,1]。"""
    files = config.get("json_files", {})
    if not files:
        raise ValueError("text_va.json_files 不能为空")
    invalid_policy = str(config.get("invalid_text_policy", "error")).lower()
    if invalid_policy not in {"error", "skip"}:
        raise ValueError("invalid_text_policy 只支持 error 或 skip")

    grouped = defaultdict(list)
    raw_by_language = defaultdict(int)
    raw_by_source = defaultdict(int)
    invalid = []
    for language, filename in files.items():
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"文本 VA 文件不存在：{path}")
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"文本 VA 顶层必须是 list：{path}")
        for row_index, row in enumerate(rows):
            raw_by_language[str(language)] += 1
            if not isinstance(row, dict):
                invalid.append((str(path), row_index, "record 不是 object"))
                continue
            source, text = str(row.get("source", "unknown")).strip(), row.get("text")
            raw_by_source[source] += 1
            if not isinstance(text, str) or not text.strip():
                invalid.append((str(path), row_index, "text 不是非空字符串"))
                continue
            try:
                valence, arousal = float(row["valence"]), float(row["arousal"])
            except (KeyError, TypeError, ValueError):
                invalid.append((str(path), row_index, "VA 缺失或不可转换"))
                continue
            valid_targets = (
                math.isfinite(valence)
                and math.isfinite(arousal)
                and 0.0 <= valence <= 1.0
                and 0.0 <= arousal <= 1.0
            )
            if not valid_targets:
                invalid.append((str(path), row_index, "VA 非有限或超出 [0,1]"))
                continue
            text = text.strip()
            grouped[(str(language), text)].append((source, valence, arousal))

    if invalid and invalid_policy == "error":
        path, index, reason = invalid[0]
        raise ValueError(f"发现 {len(invalid)} 条无效文本记录；首条：{path}[{index}] {reason}")

    records, conflicting_groups = [], 0
    for (language, text), values in grouped.items():
        valences = [value[1] for value in values]
        arousals = [value[2] for value in values]
        has_conflict = (
            max(valences) - min(valences) > 1.0e-8
            or max(arousals) - min(arousals) > 1.0e-8
        )
        if has_conflict:
            conflicting_groups += 1
        sources = sorted({value[0] for value in values})
        valence, arousal = sum(valences) / len(valences), sum(arousals) / len(arousals)
        records.append(
            {
                "id": _record_id(language, text),
                "text": text,
                "language": language,
                "source": "+".join(sources),
                "target": (2.0 * valence - 1.0, 2.0 * arousal - 1.0),
                "duplicate_count": len(values),
            }
        )

    records.sort(key=lambda record: record["id"])
    valid_raw = sum(raw_by_language.values()) - len(invalid)
    audit = {
        "raw_records": sum(raw_by_language.values()),
        "valid_raw_records": valid_raw,
        "invalid_records": len(invalid),
        "unique_records": len(records),
        "merged_duplicate_rows": valid_raw - len(records),
        "conflicting_groups": conflicting_groups,
        "raw_by_language": dict(sorted(raw_by_language.items())),
        "raw_by_source": dict(sorted(raw_by_source.items())),
        "invalid_examples": invalid[:5],
    }
    return records, audit


def split_text_va_records(records: list[dict], config: dict) -> dict[str, list[dict]]:
    """按 language/source 分层并以稳定哈希排序，生成非官方的本地固定划分。"""
    split_config = config.get("split", {})
    split_name = split_config.get("name", "local_source_stratified_group")
    if split_name != "local_source_stratified_group":
        raise ValueError("当前文本数据仅支持 local_source_stratified_group 划分")
    ratios = {
        name: float(split_config.get(name, value))
        for name, value in zip(SPLITS, (0.8, 0.1, 0.1))
    }
    valid_ratios = all(value > 0.0 for value in ratios.values()) and math.isclose(
        sum(ratios.values()), 1.0, abs_tol=1.0e-8
    )
    if not valid_ratios:
        raise ValueError(f"文本划分比例必须为正且总和为 1：{ratios}")
    seed = int(split_config.get("seed", 42))

    strata = defaultdict(list)
    for record in records:
        strata[(record["language"], record["source"])].append(record)

    result = {name: [] for name in SPLITS}
    for stratum, items in sorted(strata.items()):
        items = sorted(
            items,
            key=lambda record: _stable_key(seed, f"{stratum}\0{record['id']}"),
        )
        size = len(items)
        train_end = int(size * ratios["train"])
        validation_end = train_end + int(size * ratios["validation"])
        result["train"].extend(items[:train_end])
        result["validation"].extend(items[train_end:validation_end])
        result["test"].extend(items[validation_end:])

    for split in SPLITS:
        result[split].sort(key=lambda record: record["id"])
    return result


class TextVADataset(Dataset):
    """中英文文本 VA；样本由 :func:`build_text_datasets` 一次清洗并固定划分。"""

    def __init__(self, samples: list[dict], split: str, audit: dict) -> None:
        if split not in SPLITS:
            raise ValueError(f"未知文本 split：{split}")
        self.samples, self.split, self.audit = samples, split, audit

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        return {
            "id": sample["id"],
            "text": sample["text"],
            "target": torch.tensor(sample["target"], dtype=torch.float32),
            "language": sample["language"],
            "source": sample["source"],
            "duplicate_count": sample["duplicate_count"],
        }


def build_text_datasets(config: dict) -> dict[str, TextVADataset]:
    """只读取和清洗一次 JSON，再构造三个互斥的本地固定划分。"""
    records, audit = load_text_va_records(config)
    splits = split_text_va_records(records, config)
    return {split: TextVADataset(splits[split], split, audit) for split in SPLITS}


def collate_text_batch(samples: list[dict]) -> dict:
    """保留原始文本；tokenization 在文本模型/训练层完成。"""
    return {
        "ids": [sample["id"] for sample in samples],
        "texts": [sample["text"] for sample in samples],
        "targets": torch.stack([sample["target"] for sample in samples]),
        "languages": [sample["language"] for sample in samples],
        "sources": [sample["source"] for sample in samples],
        "duplicate_counts": torch.tensor(
            [sample["duplicate_count"] for sample in samples], dtype=torch.long
        ),
    }


def build_text_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_text_batch,
        generator=generator,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
