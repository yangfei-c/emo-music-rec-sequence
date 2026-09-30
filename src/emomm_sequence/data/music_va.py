"""Load frozen music VA features and enforce five disjoint data partitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from emomm_sequence.training.records import check_identity_splits, sha256

ROOT = Path(__file__).resolve().parents[3]

def validation_partition(groups: np.ndarray, seed: int) -> dict[str, str]:
    """Match the fixed five-way split used in the existing music-VA run."""
    ordered = sorted(set(groups.astype(str)), key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    n_model, n_cal = round(len(ordered) * 0.4), round(len(ordered) * 0.3)
    return {
        group: "validation_model" if i < n_model else "calibration" if i < n_model + n_cal else "policy"
        for i, group in enumerate(ordered)
    }


def _identity_hash(values: np.ndarray) -> str:
    return hashlib.sha256("\n".join(sorted(values.astype(str))).encode()).hexdigest()


def prepare_data(config: dict) -> tuple[dict[str, np.ndarray], dict]:
    blocks: list[dict[str, np.ndarray]] = []
    source_audit: dict[str, dict] = {}
    for source, relative in config["data"]["sources"].items():
        path = ROOT / relative
        with np.load(path, allow_pickle=False) as loaded:
            embeddings = loaded["audio_embeddings"].astype(np.float32)
            targets = ((loaded["va_targets"].astype(np.float32) + 1.0) / 2.0).astype(np.float32)
            splits = loaded["splits"].astype(str)
            identities = loaded["global_identities"].astype(str)
            groups = loaded["content_group_ids"].astype(str)
            source_audit[source] = {
                "input_path": relative,
                "input_sha256": sha256(path),
                "feature_manifest_sha256": str(loaded["manifest_sha256"].item()),
                "feature_checkpoint_sha256": str(loaded["checkpoint_sha256"].item()),
                "n_rows": int(len(splits)),
            }
        if embeddings.ndim != 2 or embeddings.shape[1] != int(config["model"]["embedding_dim"]):
            raise ValueError(f"Unexpected {source} feature shape: {embeddings.shape}")
        if targets.shape != (len(embeddings), 2):
            raise ValueError(f"Unexpected {source} VA shape: {targets.shape}")
        blocks.append({
            "embeddings": embeddings,
            "targets": targets,
            "splits": splits,
            "identities": identities,
            "groups": groups,
            "sources": np.repeat(source, len(splits)),
        })
    data = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    check_identity_splits(data["groups"], data["splits"], "music VA content groups")
    if len(set(data["groups"])) != len(data["groups"]):
        raise ValueError("Clean music VA features must contain one row per content group")
    if not np.isfinite(data["embeddings"]).all() or not np.isfinite(data["targets"]).all():
        raise ValueError("Nonfinite features or targets")
    if np.any(data["targets"] < 0) or np.any(data["targets"] > 1):
        raise ValueError("Music-VA targets must be in [0,1]")

    partitions = data["splits"].astype("<U20")
    for source in sorted(set(data["sources"])):
        selected = (data["sources"] == source) & (data["splits"] == "validation")
        mapping = validation_partition(data["groups"][selected], int(config["data"]["subsplit_seed"]))
        partitions[selected] = [mapping[group] for group in data["groups"][selected]]
    data["partitions"] = partitions
    expected_counts = config["data"].get("expected_counts", {})
    counts: dict[str, dict] = {}
    partition_hashes: dict[str, dict] = {}
    for source in sorted(set(data["sources"])):
        counts[source] = {}
        partition_hashes[source] = {}
        for partition in ("train", "validation_model", "calibration", "policy", "test"):
            selected = (data["sources"] == source) & (partitions == partition)
            counts[source][partition] = int(selected.sum())
            partition_hashes[source][partition] = _identity_hash(data["groups"][selected])
        if source in expected_counts and counts[source] != expected_counts[source]:
            raise ValueError(f"{source} partitions differ from frozen manifest: {counts[source]}")
    audit = {
        "target_transform": "(clean_npz_va_targets_in_minus1_plus1 + 1) / 2",
        "subsplit_seed": int(config["data"]["subsplit_seed"]),
        "content_groups_unique": True,
        "source_assets": source_audit,
        "counts": counts,
        "partition_group_sha256": partition_hashes,
        "test_used_for_training_validation_or_calibration": False,
    }
    return data, audit


