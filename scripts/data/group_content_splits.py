"""Repair train/validation/test splits by grouping near-identical audio content.

Existing non-conflicting assignments are preserved.  If one content cluster spans
multiple splits, every member is deterministically assigned together, without
using VA labels or retrieval outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size)

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.9999)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--tag-supervision", type=Path)
    parser.add_argument("--deduplicate-music", action="store_true")
    args = parser.parse_args()
    paths = [path if path.is_absolute() else ROOT / path for path in args.input]
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    loaded, embeddings, identities, original_splits = [], [], [], []
    offsets = [0]
    for path in paths:
        with np.load(path, allow_pickle=False) as source:
            values = {key: source[key] for key in source.files}
        if "audio_embeddings" not in values:
            raise KeyError(f"{path} lacks audio_embeddings")
        loaded.append(values)
        embeddings.append(values["audio_embeddings"].astype(np.float32))
        identity = values.get("global_identities")
        if identity is None:
            identity = np.char.add(np.char.add(values["datasets"].astype(str), ":"), values["ids"].astype(str))
        identities.append(identity.astype(str))
        original_splits.append(values["splits"].astype(str))
        offsets.append(offsets[-1] + len(identity))
    if bool(args.catalog) != bool(args.tag_supervision):
        parser.error("--catalog and --tag-supervision must be supplied together")
    tag_values = None
    if args.catalog:
        with np.load(args.catalog, allow_pickle=False) as catalog:
            catalog_embeddings = catalog["embeddings"].astype(np.float32)
            catalog_ids = catalog["track_ids"].astype(str)
        with np.load(args.tag_supervision, allow_pickle=False) as tags:
            tag_values = {key: tags[key] for key in tags.files}
        positions = tag_values["index_positions"]
        if not np.array_equal(catalog_ids[positions], tag_values["track_ids"]):
            raise ValueError("Catalog/tag row alignment mismatch")
        embeddings.append(catalog_embeddings[positions])
        identities.append(np.asarray([f"mtg:{int(value.removeprefix('track_'))}" for value in tag_values["track_ids"]]))
        original_splits.append(tag_values["splits"].astype(str))
    embedding = np.concatenate(embeddings)
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    identity = np.concatenate(identities)
    split = np.concatenate(original_splits)
    union_find = UnionFind(len(embedding))
    identity_first: dict[str, int] = {}
    for index, value in enumerate(identity):
        if value in identity_first:
            union_find.union(index, identity_first[value])
        else:
            identity_first[value] = index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(4)
    tensor = torch.from_numpy(embedding).to(device)
    for start in range(0, len(embedding), args.block_size):
        stop = min(start + args.block_size, len(embedding))
        similarities = tensor[start:stop] @ tensor.T
        matched = torch.nonzero(similarities >= args.threshold).cpu().numpy()
        for left_offset, right in matched:
            left = start + int(left_offset)
            if left < int(right):
                union_find.union(left, int(right))
        if start % (args.block_size * 20) == 0:
            print(f"Content audit {stop}/{len(embedding)}", flush=True)
    del tensor, similarities
    groups: dict[int, list[int]] = {}
    for index in range(len(embedding)):
        groups.setdefault(union_find.find(index), []).append(index)
    repaired = split.copy()
    content_group_ids = np.empty(len(split), dtype="<U64")
    conflicting_groups = []
    for members in groups.values():
        member_identities = sorted(set(identity[members].tolist()))
        group_key = hashlib.sha256("|".join(member_identities).encode("utf-8")).hexdigest()
        content_group_ids[members] = group_key
        member_splits = sorted(set(split[members].tolist()))
        if len(member_splits) > 1:
            # A held-out recording must never be moved into model training.
            destination = max(member_splits, key={"train": 0, "validation": 1, "test": 2}.__getitem__)
            repaired[members] = destination
            conflicting_groups.append(
                {
                    "content_group": group_key,
                    "members": member_identities,
                    "original_splits": member_splits,
                    "repaired_split": destination,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    exclusions = []
    # Music-VA uses one representative per recording. Caption pairs deliberately
    # retain multiple texts for the same music (multi-positive retrieval).
    music_members = {}
    if args.deduplicate_music:
        for file_index, (path, values) in enumerate(zip(paths, loaded)):
            if "text_embeddings" in values:
                continue
            for row in range(len(values["splits"])):
                index = offsets[file_index] + row
                music_members.setdefault(content_group_ids[index], []).append(index)
    excluded_indices = set()
    for group, members in music_members.items():
        chosen = max(members, key=lambda i: ({"train": 0, "validation": 1, "test": 2}[split[i]], -i))
        for index in members:
            if index != chosen:
                excluded_indices.add(index)
                exclusions.append({"identity": str(identity[index]), "kept_identity": str(identity[chosen]), "content_group": str(group)})
    for file_index, (path, values) in enumerate(zip(paths, loaded)):
        start, stop = offsets[file_index], offsets[file_index + 1]
        values["original_splits"] = values["splits"].astype(str)
        values["splits"] = repaired[start:stop]
        values["content_group_ids"] = content_group_ids[start:stop]
        keep = np.asarray([i not in excluded_indices for i in range(start, stop)], dtype=bool)
        values = {key: value[keep] if value.ndim and len(value) == stop - start else value for key, value in values.items()}
        output = output_dir / f"{path.stem}_clean.npz"
        np.savez_compressed(output, **values)
        output_files.append(
            {
                "source": path.as_posix(),
                "output": output.as_posix(),
                "rows": int(keep.sum()),
                "deduplicated_rows": int((~keep).sum()),
                "moved_rows": int(np.sum(split[start:stop] != repaired[start:stop])),
                "split_counts": {
                    name: int(np.sum(values["splits"] == name))
                    for name in ("train", "validation", "test")
                },
            }
        )
    if tag_values is not None:
        begin = offsets[-1]
        tag_values["original_splits"] = tag_values["splits"].copy()
        tag_values["splits"] = repaired[begin:]
        tag_values["content_group_ids"] = content_group_ids[begin:]
        # Eligibility uses the repaired training partition only.
        eligible_counts = {}
        train_mask = tag_values["splits"] == "train"
        for factor in ("preference", "affect"):
            targets = tag_values[f"{factor}_targets"]
            keep = np.asarray([int((train_mask & targets[:, j]).sum()) >= 50 and len(set(tag_values["artist_ids"][train_mask & targets[:, j]])) >= 20 for j in range(targets.shape[1])])
            for suffix in ("labels", "prompts", "label_embeddings"):
                tag_values[f"{factor}_{suffix}"] = tag_values[f"{factor}_{suffix}"][keep]
            tag_values[f"{factor}_targets"] = targets[:, keep]
            eligible_counts[factor] = int(keep.sum())
        tag_output = output_dir / "mtg_factor_supervision_clean.npz"
        np.savez_compressed(tag_output, **tag_values)
        output_files.append({"source": str(args.tag_supervision), "output": str(tag_output), "rows": len(tag_values["splits"]), "moved_rows": int((split[begin:] != repaired[begin:]).sum()), "eligible_labels": eligible_counts, "split_counts": {name: int((tag_values["splits"] == name).sum()) for name in ("train", "validation", "test")}})
    report = {
        "method": "content_group_split_repair",
        "cosine_threshold": args.threshold,
        "rows": len(split),
        "content_groups": len(groups),
        "conflicting_groups_repaired": len(conflicting_groups),
        "conflicting_groups": conflicting_groups,
        "excluded_music_duplicates": exclusions,
        "files": output_files,
    }
    (output_dir / "content_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key not in {"conflicting_groups", "excluded_music_duplicates"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
