"""Assemble independently selected semantic, affect, and preference heads.

The six adapters are independent by construction.  This utility combines
factor states selected on validation data and re-evaluates the assembled model
on validation only.  It never reads a test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/training"))

from emomm_sequence.evaluation.retrieval import counterfactual_factor_metrics
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, state_sha256, write_json
from emomm_sequence.training.utilities import load_yaml
from train_retriever import (
    load_pair_data,
    load_va_data,
    make_va_loader,
    pair_evaluation,
    resolve,
    tag_evaluation,
    va_evaluation,
)


FACTORS = ("semantic", "affect", "preference")
TRAINING_ONLY_MODEL_FIELDS = {
    "adapter_drop_probability",
    "adapter_drop_probabilities",
}


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "model_config", "seed", "run_id"}
    if not required.issubset(payload):
        raise ValueError(f"checkpoint lacks training metadata: {path}")
    return payload


def factor_keys(state: dict[str, torch.Tensor], factor: str) -> list[str]:
    prefixes = (
        f"text_adapters.{factor}.",
        f"music_adapters.{factor}.",
        f"logit_scales.{factor}",
    )
    return [key for key in state if key.startswith(prefixes)]


def inference_architecture(config: dict) -> dict:
    """Remove stochastic training controls that do not alter checkpoint shape."""

    return {
        key: value
        for key, value in config.items()
        if key not in TRAINING_ONLY_MODEL_FIELDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for factor in FACTORS:
        parser.add_argument(f"--{factor}", type=Path, required=True)
    parser.add_argument("--semantic-alpha", type=float, default=0.4)
    parser.add_argument("--affect-alpha", type=float, default=1.0)
    parser.add_argument("--preference-alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/retriever.yaml")
    args = parser.parse_args()
    blends = {
        factor: float(getattr(args, f"{factor}_alpha")) for factor in FACTORS
    }
    if any(not 0.0 <= value <= 1.0 for value in blends.values()):
        parser.error("all output blends must be in [0, 1]")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    paths = {
        factor: (
            getattr(args, factor)
            if getattr(args, factor).is_absolute()
            else ROOT / getattr(args, factor)
        )
        for factor in FACTORS
    }
    payloads = {factor: load_checkpoint(path) for factor, path in paths.items()}
    source_model_configs = {
        factor: payloads[factor]["model_config"] for factor in FACTORS
    }
    architecture_configs = [
        inference_architecture(source_model_configs[factor]) for factor in FACTORS
    ]
    if any(config != architecture_configs[0] for config in architecture_configs[1:]):
        raise ValueError("factor checkpoints use different model architectures")
    seeds = {int(payloads[factor]["seed"]) for factor in FACTORS}
    if len(seeds) != 1:
        raise ValueError("factor checkpoints must use the same random seed")

    state = {
        key: value.detach().cpu().clone()
        for key, value in payloads["semantic"]["model"].items()
    }
    for factor in FACTORS:
        source_state = payloads[factor]["model"]
        keys = factor_keys(source_state, factor)
        if not keys:
            raise ValueError(f"no {factor} parameters found")
        for key in keys:
            state[key] = source_state[key].detach().cpu().clone()
    for key in state:
        if key.startswith("text_va_head.") or key.startswith("music_va_head."):
            state[key] = payloads["affect"]["model"][key].detach().cpu().clone()

    model_config = dict(architecture_configs[0])
    model_config["adapter_drop_probability"] = 0.0
    model_config["adapter_drop_probabilities"] = {
        factor: 0.0 for factor in FACTORS
    }
    for factor, alpha in blends.items():
        model_config[f"{factor}_output_blend"] = alpha
    selection_scores = {
        factor: float(
            payloads[factor].get("factor_best_validation_scores", {}).get(
                factor, float("nan")
            )
        )
        for factor in FACTORS
    }
    checkpoint = output / "best.pt"
    torch.save(
        {
            "model": state,
            "model_config": model_config,
            "embedding_dim": int(model_config.get("embedding_dim", 768)),
            "selection_metric": "modular_factorwise_validation",
            "factor_source_validation_scores_before_blending": selection_scores,
            "seed": seeds.pop(),
            "run_id": output.name,
            "source_checkpoints": {
                factor: str(path.relative_to(ROOT)) for factor, path in paths.items()
            },
        },
        checkpoint,
    )

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    supervision = config["supervision"]
    pair_data = load_pair_data([resolve(path) for path in supervision["semantic_pairs"]])
    text_va_data = load_va_data([resolve(supervision["text_va"])], "text_embeddings")
    music_va_data = load_va_data(
        [resolve(path) for path in supervision["music_va"]], "audio_embeddings"
    )
    with np.load(resolve(supervision["mtg_factors"]), allow_pickle=False) as data:
        mtg = {key: data[key] for key in data.files}
    with np.load(resolve(supervision["music_index"]), allow_pickle=False) as data:
        music_index = data["embeddings"].astype(np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenRetriever(checkpoint).to(device).eval()
    batch_pairs = int(config["training"]["batch_size_pairs"])
    batch_mtg = int(config["training"]["batch_size_mtg"])
    batch_va = int(config["training"]["batch_size_va"])
    retrieval_ks = tuple(int(k) for k in config["evaluation"]["retrieval_ks"])
    tag_ks = tuple(int(k) for k in config["evaluation"]["tag_ks"])
    text_loader, text_domains = make_va_loader(
        text_va_data, "text_embeddings", "validation", batch_va, int(model.seed if hasattr(model, "seed") else 42)
    )
    music_loader, music_domains = make_va_loader(
        music_va_data, "audio_embeddings", "validation", batch_va, 42
    )
    metrics = {
        "partition": "validation",
        "test_evaluated": False,
        "semantic_pairs": pair_evaluation(
            model, pair_data, "validation", device, batch_pairs, retrieval_ks
        ),
        "preference_tags": tag_evaluation(
            model, "preference", mtg, music_index, "validation", device, batch_mtg, tag_ks
        ),
        "affect_tags": tag_evaluation(
            model, "affect", mtg, music_index, "validation", device, batch_mtg, tag_ks
        ),
        "text_va": va_evaluation(model, "text", text_loader, device, text_domains),
        "music_va": va_evaluation(model, "music", music_loader, device, music_domains),
        "counterfactual": counterfactual_factor_metrics(
            model,
            mtg,
            music_index,
            "validation",
            device,
            batch_mtg,
            int(config["evaluation"]["counterfactual_queries"]),
            int(config["evaluation"]["counterfactual_seed"]),
        ),
    }
    write_json(output / "metrics.json", metrics)
    write_json(
        output / "provenance.json",
        {
            "test_evaluated": False,
            "seed": int(next(iter({payloads[f]["seed"] for f in FACTORS}))),
            "blends": blends,
            "sources": {
                factor: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "run_id": payloads[factor]["run_id"],
                    "factor_best_epoch": payloads[factor].get("factor_best_epochs", {}).get(factor),
                    "training_adapter_drop_probability": source_model_configs[
                        factor
                    ].get("adapter_drop_probability", 0.0),
                    "training_adapter_drop_probabilities": source_model_configs[
                        factor
                    ].get("adapter_drop_probabilities"),
                }
                for factor, path in paths.items()
            },
            "checkpoint_sha256": sha256(checkpoint),
            "assembled_state_sha256": state_sha256(model),
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "semantic_mrr": metrics["semantic_pairs"]["selection_score"],
                "preference_ndcg_at_10": metrics["preference_tags"]["macro"]["ndcg_at_10"],
                "affect_ndcg_at_10": metrics["affect_tags"]["macro"]["ndcg_at_10"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
