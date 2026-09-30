"""Audit the language-facing quality of the supported MTG-Jamendo labels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.data.tags import (
    MTG_PROMPT_OVERRIDES,
    MTG_SEMANTICALLY_AMBIGUOUS,
    legacy_tag_prompt,
    split_mtg_tag,
    tag_display_name,
    tag_prompt,
    tag_query_views,
)
from emomm_sequence.training.records import sha256, write_json


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def issue_types(label: str) -> list[str]:
    _, value = split_mtg_tag(label)
    issues: list[str] = []
    display = tag_display_name(label)
    if value.replace("_", " ").replace("-", " ") != display:
        issues.append("identifier_needs_lexical_expansion")
    if label in MTG_PROMPT_OVERRIDES:
        issues.append("prompt_needs_musical_context")
    if label in MTG_SEMANTICALLY_AMBIGUOUS:
        issues.append("semantically_ambiguous_without_context")
    if legacy_tag_prompt(label) != tag_prompt(label) and not issues:
        issues.append("surface_form_normalization")
    return issues


def top_cooccurrences(
    targets: np.ndarray, labels: np.ndarray, limit: int = 30
) -> list[dict]:
    binary = targets.astype(np.int64)
    counts = binary.sum(axis=0)
    intersections = binary.T @ binary
    unions = counts[:, None] + counts[None, :] - intersections
    jaccard = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    np.fill_diagonal(jaccard, 0.0)
    pairs: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for flat in np.argsort(jaccard, axis=None)[::-1]:
        left, right = np.unravel_index(int(flat), jaccard.shape)
        key = tuple(sorted((int(left), int(right))))
        if left == right or key in seen:
            continue
        seen.add(key)
        pairs.append(
            {
                "left": str(labels[left]),
                "right": str(labels[right]),
                "jaccard": float(jaccard[left, right]),
                "intersection": int(intersections[left, right]),
                "left_train_count": int(counts[left]),
                "right_train_count": int(counts[right]),
            }
        )
        if len(pairs) == limit:
            break
    return pairs


def audit_factor(
    factor: str, data: dict[str, np.ndarray], train_mask: np.ndarray
) -> tuple[list[dict], dict]:
    labels = data[f"{factor}_labels"].astype(str)
    stored_prompts = data[f"{factor}_prompts"].astype(str)
    targets = data[f"{factor}_targets"].astype(bool)
    counts = targets[train_mask].sum(axis=0)
    artists = data["artist_ids"].astype(str)
    rows: list[dict] = []
    canonical_groups: dict[str, list[str]] = defaultdict(list)
    for index, (label, stored_prompt) in enumerate(
        zip(labels, stored_prompts, strict=True)
    ):
        canonical = tag_prompt(label)
        canonical_groups[canonical.casefold()].append(label)
        positive_artists = np.unique(artists[train_mask][targets[train_mask, index]])
        rows.append(
            {
                "factor": factor,
                "index": index,
                "official_label_id": label,
                "namespace": split_mtg_tag(label)[0],
                "display_name": tag_display_name(label),
                "stored_legacy_prompt": stored_prompt,
                "reconstructed_legacy_prompt": legacy_tag_prompt(label),
                "canonical_prompt": canonical,
                "query_views": list(tag_query_views(label)),
                "issues": issue_types(label),
                "train_tracks": int(counts[index]),
                "train_artists": int(len(positive_artists)),
                "target_action": "keep_official_label_id_and_targets",
            }
        )
    duplicate_prompts = [
        labels for labels in canonical_groups.values() if len(labels) > 1
    ]
    summary = {
        "labels": len(labels),
        "changed_prompts": sum(
            row["stored_legacy_prompt"] != row["canonical_prompt"] for row in rows
        ),
        "labels_with_issues": sum(bool(row["issues"]) for row in rows),
        "issue_counts": {
            issue: sum(issue in row["issues"] for row in rows)
            for issue in sorted({issue for row in rows for issue in row["issues"]})
        },
        "duplicate_canonical_prompt_groups": duplicate_prompts,
        "top_train_cooccurrences": top_cooccurrences(targets[train_mask], labels),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/features/clamp3/clean/mtg_factor_supervision_clean.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/mtg_label_audit_20260914/evaluation")
    )
    args = parser.parse_args()
    source = resolve(args.input)
    output_dir = resolve(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    train_mask = data["splits"].astype(str) == "train"
    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for factor in ("preference", "affect"):
        factor_rows, factor_summary = audit_factor(factor, data, train_mask)
        rows.extend(factor_rows)
        summaries[factor] = factor_summary
    payload = {
        "scope": "language-facing mapping audit; no track target was changed",
        "mapping_decision_partition": "label strings and official documentation only",
        "model_metrics_or_test_labels_used_for_mapping": False,
        "official_preprocessing": {
            "status": "the public autotagging splits are already officially cleaned",
            "tag_map_url": "https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/tag_map.json",
            "dataset_repository": "https://github.com/MTG/mtg-jamendo-dataset",
            "remaining_risk": (
                "Uploader-provided weak labels can contain missing or incorrect track-level "
                "annotations; this audit does not relabel tracks without independent human evidence."
            ),
        },
        "source": {"path": str(source), "sha256": sha256(source)},
        "train_tracks": int(train_mask.sum()),
        "summary": summaries,
        "mapping": rows,
    }
    write_json(output_dir / "audit.json", payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mapping.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "factor",
                "index",
                "official_label_id",
                "namespace",
                "display_name",
                "stored_legacy_prompt",
                "canonical_prompt",
                "query_views",
                "issues",
                "train_tracks",
                "train_artists",
                "target_action",
            ),
        )
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["query_views"] = " | ".join(row["query_views"])
            formatted["issues"] = " | ".join(row["issues"])
            formatted.pop("reconstructed_legacy_prompt")
            writer.writerow(formatted)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
