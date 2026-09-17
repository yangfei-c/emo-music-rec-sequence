"""Compare frozen CLaMP3 and Ours on identical held-out recordings/prompts."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.evaluation.pairs import grouped_retrieval_metrics
from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.utilities import load_yaml


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def normalized(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


@torch.inference_mode()
def project(model, values: np.ndarray, modality: str, factor: str) -> np.ndarray:
    if model is None:
        return normalized(values.astype(np.float32))
    return np.concatenate([
        model.encode_factor(
            modality,
            factor,
            torch.from_numpy(values[i:i+512].astype(np.float32)),
        ).numpy()
        for i in range(0, len(values), 512)
    ])


def evaluate(model, pairs: list[dict], tags: dict, catalog: dict) -> dict:
    semantic = {}
    for data in pairs:
        for dataset in np.unique(data["datasets"]):
            mask = (data["splits"] == "test") & (data["datasets"] == dataset)
            if not mask.any():
                continue
            semantic[str(dataset)] = grouped_retrieval_metrics(
                project(model, data["text_embeddings"][mask], "text", "semantic"),
                project(model, data["audio_embeddings"][mask], "music", "semantic"),
                data["global_identities"][mask].astype(str), (1, 5, 10))
    result = {"semantic_pairs": semantic}
    mask = tags["splits"] == "test"
    music = catalog["embeddings"][tags["index_positions"][mask]]
    for factor in ("preference", "affect"):
        queries = project(model, tags[f"{factor}_label_embeddings"], "text", factor)
        embeddings = project(model, music, "music", factor)
        metrics = label_to_track_metrics(queries @ embeddings.T, tags[f"{factor}_targets"][mask].T)
        metrics["prompts"] = tags[f"{factor}_prompts"].astype(str).tolist()
        result[f"{factor}_tags"] = metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/current_retriever.yaml")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_yaml(args.config)
    supervision = config["supervision"]
    pairs = [load_npz(ROOT / value) for value in supervision["semantic_pairs"]]
    tags = load_npz(ROOT / supervision["mtg_factors"])
    catalog = load_npz(ROOT / supervision["music_index"])
    aggregate = json.loads((args.runs / "aggregate.json").read_text(encoding="utf-8"))
    selected = max(aggregate["runs"], key=lambda r: (r["best_validation_score"], -r["seed"]))
    with threadpool_limits(limits=4):
        baseline = evaluate(None, pairs, tags, catalog)
        ours = []
        for seed in aggregate["seeds"]:
            checkpoint = args.runs / f"seed_{seed}" / "best.pt"
            model = FrozenRetriever(checkpoint).eval()
            ours.append({"seed": seed, "checkpoint_sha256": sha256(checkpoint), "results": evaluate(model, pairs, tags, catalog)})
            print(f"Retrieval evaluation seed {seed} complete", flush=True)
    write_json(args.output, {"baseline": {"model": "CLaMP3", "results": baseline}, "ours": {"model": "Ours", "runs": ours}, "selected_seed": selected["seed"],
        "protocol": {"same_music_candidates": True, "same_prompts": True, "same_official_backbone": True, "candidate_partition": "test_after_content_group_repair", "preference_labels": "MTG genre/instrument, eligible from training only", "affect_labels": "MTG mood/theme, eligible from training only", "baseline_definition": "L2-normalized original CLaMP3 text/music vectors; cosine similarity; no trained Adapters", "ours_definition": "same frozen CLaMP3 vectors through newly trained matching factor Adapters", "song_describer_role": "external_test_only_never_training", "qwen_in_benchmark": False, "seed_selection": "highest_validation_composite_only", "test_status": "re_evaluation_not_new_unseen_holdout", "uncertainty": "three-seed SD describes initialization variation, not subject-level uncertainty"}})
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
