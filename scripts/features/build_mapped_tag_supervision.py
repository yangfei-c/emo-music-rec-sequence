"""Build mapped MTG supervision without changing official track targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.data.tags import tag_prompt, tag_query_views
from emomm_sequence.encoding.text import FrozenTextEncoder
from emomm_sequence.training.records import sha256, write_json


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


@torch.inference_mode()
def encode_texts(
    texts: list[str], config: dict, batch_size: int, device: torch.device
) -> np.ndarray:
    backbone = config["backbone"]
    encoder = FrozenTextEncoder(
        ROOT / backbone["checkpoint"], config.get("text_model", "FacebookAI/xlm-roberta-base")
    ).to(device)
    tokenizer = encoder.tokenizer()
    outputs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        outputs.append(
            encoder.encode(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            )
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(outputs).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("artifacts/features/clamp3/clean/mtg_factor_supervision_clean.npz"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/training/retriever.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz"),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    source, config_path, output = map(resolve, (args.source, args.config, args.output))
    report = output.with_suffix(".json")
    if output.exists() or report.exists():
        raise FileExistsError(output)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    source_targets = {
        factor: payload[f"{factor}_targets"].copy()
        for factor in ("preference", "affect")
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factor_reports: dict[str, dict] = {}
    for factor in ("preference", "affect"):
        labels = payload[f"{factor}_labels"].astype(str)
        legacy_prompts = payload[f"{factor}_prompts"].astype(str)
        legacy_embeddings = payload[f"{factor}_label_embeddings"].astype(np.float32)
        canonical_prompts = np.asarray([tag_prompt(label) for label in labels], dtype=str)
        views = np.asarray([tag_query_views(label) for label in labels], dtype=str)
        if views.ndim != 2 or views.shape[1] != 3:
            raise ValueError(f"{factor} query views must have exactly three forms")
        unique_texts = list(dict.fromkeys(views.reshape(-1).tolist()))
        encoded = encode_texts(unique_texts, config, args.batch_size, device)
        lookup = {text: embedding for text, embedding in zip(unique_texts, encoded, strict=True)}
        view_embeddings = np.stack(
            [[lookup[text] for text in row] for row in views], axis=0
        ).astype(np.float32)
        payload[f"{factor}_legacy_prompts"] = legacy_prompts
        payload[f"{factor}_legacy_label_embeddings"] = legacy_embeddings
        payload[f"{factor}_prompts"] = canonical_prompts
        payload[f"{factor}_label_embeddings"] = view_embeddings[:, 0]
        payload[f"{factor}_query_prompts"] = views
        payload[f"{factor}_query_embeddings"] = view_embeddings
        factor_reports[factor] = {
            "labels": int(len(labels)),
            "changed_primary_prompts": int((legacy_prompts != canonical_prompts).sum()),
            "query_views_per_label": 3,
            "unique_query_texts": len(unique_texts),
            "targets_unchanged": bool(
                np.array_equal(source_targets[factor], payload[f"{factor}_targets"])
            ),
        }
    payload["mapping_version"] = np.asarray("mtg_language_mapping_v1")
    payload["mapping_source_sha256"] = np.asarray(sha256(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    result = {
        "scope": "prompt remapping and query views only",
        "official_label_ids_unchanged": True,
        "track_targets_unchanged": all(
            item["targets_unchanged"] for item in factor_reports.values()
        ),
        "source": {"path": str(source), "sha256": sha256(source)},
        "config": {"path": str(config_path), "sha256": sha256(config_path)},
        "output": {"path": str(output), "sha256": sha256(output)},
        "factors": factor_reports,
    }
    write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
