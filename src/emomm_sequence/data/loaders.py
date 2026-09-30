"""Load fixed feature archives and create reproducible training batches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def concatenate_npz(paths: list[Path], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    parts = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                parts[key].append(data[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def load_pair_data(paths: list[Path]) -> dict[str, np.ndarray]:
    return concatenate_npz(
        paths,
        (
            "text_embeddings",
            "audio_embeddings",
            "splits",
            "datasets",
            "global_identities",
        ),
    )


def load_va_data(paths: list[Path], embedding_key: str) -> dict[str, np.ndarray]:
    return concatenate_npz(
        paths,
        (embedding_key, "va_targets", "splits", "datasets", "global_identities"),
    )


def make_pair_loader(
    data: dict[str, np.ndarray], split: str, batch_size: int, seed: int
) -> DataLoader:
    mask = data["splits"].astype(str) == split
    identities = data["global_identities"][mask].astype(str)
    _, identity_ids = np.unique(identities, return_inverse=True)
    domains = data["datasets"][mask].astype(str)
    _, domain_ids = np.unique(domains, return_inverse=True)
    dataset = TensorDataset(
        torch.from_numpy(data["text_embeddings"][mask].astype(np.float32)),
        torch.from_numpy(data["audio_embeddings"][mask].astype(np.float32)),
        torch.from_numpy(identity_ids.astype(np.int64)),
        torch.from_numpy(domain_ids.astype(np.int64)),
    )
    sampler = None
    if split == "train":
        counts = np.bincount(domain_ids)
        weights = torch.from_numpy((1.0 / counts[domain_ids]).astype(np.float64))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(counts.max() * len(counts)),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=split == "train",
    )


def make_va_loader(
    data: dict[str, np.ndarray],
    embedding_key: str,
    split: str,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, list[str]]:
    mask = data["splits"].astype(str) == split
    domains = data["datasets"][mask].astype(str)
    domain_names = sorted(set(domains.tolist()))
    domain_index = {name: index for index, name in enumerate(domain_names)}
    domain_ids = np.asarray([domain_index[value] for value in domains], dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(data[embedding_key][mask].astype(np.float32)),
        torch.from_numpy(data["va_targets"][mask].astype(np.float32)),
        torch.from_numpy(domain_ids),
    )
    sampler = None
    if split == "train" and len(domain_names) > 1:
        counts = np.bincount(domain_ids)
        weights = torch.from_numpy((1.0 / counts[domain_ids]).astype(np.float64))
        sampler = WeightedRandomSampler(
            weights,
            num_samples=int(counts.max() * len(counts)),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train" and sampler is None,
        sampler=sampler,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=split == "train",
    )
    return loader, domain_names


def make_mtg_loader(
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    split: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    mask = mtg["splits"].astype(str) == split
    embeddings = music_index[mtg["index_positions"][mask]].astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(mtg["preference_targets"][mask].astype(bool)),
        torch.from_numpy(mtg["affect_targets"][mask].astype(bool)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=split == "train",
    )


