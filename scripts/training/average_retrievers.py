"""Average compatible Ours retriever checkpoints and evaluate validation only.

This implements a deployment-neutral model soup: several retrievers that start
from the same initialization are averaged into one parameter state, so inference
uses exactly one model.  Test partitions are deliberately never evaluated here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

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


def load_checkpoint(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "model_config"}
    if not required.issubset(payload):
        raise ValueError(f"checkpoint lacks model metadata: {path}")
    return payload


def average_states(
    states: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return a dtype-preserving weighted average of compatible states."""

    if not states:
        raise ValueError("at least one state is required")
    reference_keys = tuple(states[0].keys())
    for index, state in enumerate(states[1:], start=1):
        if tuple(state.keys()) != reference_keys:
            raise ValueError(f"state {index} has different parameter keys/order")
    if weights is None:
        weights = [1.0 / len(states)] * len(states)
    if len(weights) != len(states):
        raise ValueError("one weight is required per state")
    coefficients = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any():
        raise ValueError("weights must be finite and non-negative")
    total = float(coefficients.sum())
    if total <= 0:
        raise ValueError("weights must have positive sum")
    coefficients /= total

    result: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        tensors = [state[key].detach().cpu() for state in states]
        shape, dtype = tensors[0].shape, tensors[0].dtype
        if any(tensor.shape != shape or tensor.dtype != dtype for tensor in tensors[1:]):
            raise ValueError(f"incompatible tensor for {key}")
        if tensors[0].is_floating_point() or tensors[0].is_complex():
            accumulator = torch.zeros_like(tensors[0], dtype=torch.float64)
            for coefficient, tensor in zip(coefficients, tensors, strict=True):
                accumulator.add_(tensor.to(torch.float64), alpha=float(coefficient))
            result[key] = accumulator.to(dtype)
        else:
            if any(not torch.equal(tensor, tensors[0]) for tensor in tensors[1:]):
                raise ValueError(f"non-floating state differs for {key}")
            result[key] = tensors[0].clone()
    return result


def evaluate_validation(checkpoint: Path, config_path: Path) -> dict:
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
        text_va_data, "text_embeddings", "validation", batch_va, 42
    )
    music_loader, music_domains = make_va_loader(
        music_va_data, "audio_embeddings", "validation", batch_va, 42
    )
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--weights", type=float, nargs="+")
    parser.add_argument("--semantic-alpha", type=float, default=0.4)
    parser.add_argument("--affect-alpha", type=float, default=1.0)
    parser.add_argument("--preference-alpha", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/training/retriever.yaml"
    )
    args = parser.parse_args()
    blends = {
        "semantic": float(args.semantic_alpha),
        "affect": float(args.affect_alpha),
        "preference": float(args.preference_alpha),
    }
    if any(not 0.0 <= value <= 1.0 for value in blends.values()):
        parser.error("all output blends must be in [0, 1]")
    paths = [path if path.is_absolute() else ROOT / path for path in args.checkpoints]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if output.exists():
        raise FileExistsError(output)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("duplicate checkpoint in soup")
    payloads = [load_checkpoint(path) for path in paths]
    architecture = payloads[0]["model_config"]
    if any(payload["model_config"] != architecture for payload in payloads[1:]):
        raise ValueError("checkpoints use different model architectures")
    initial_hashes = {payload.get("initial_state_sha256") for payload in payloads}
    if None in initial_hashes or len(initial_hashes) != 1:
        raise ValueError(
            "model soup requires checkpoints with the same recorded initialization"
        )
    normalized_weights = np.asarray(
        args.weights if args.weights is not None else np.ones(len(payloads)),
        dtype=np.float64,
    )
    if len(normalized_weights) != len(payloads):
        parser.error("--weights must match --checkpoints")
    normalized_weights /= normalized_weights.sum()
    state = average_states([payload["model"] for payload in payloads], normalized_weights)
    model_config = dict(architecture)
    for factor, alpha in blends.items():
        model_config[f"{factor}_output_blend"] = alpha

    output.mkdir(parents=True)
    checkpoint = output / "best.pt"
    torch.save(
        {
            "model": state,
            "model_config": model_config,
            "embedding_dim": int(model_config.get("embedding_dim", 768)),
            "selection_metric": "validation_neutral_weight_average",
            "seed": int(payloads[0].get("seed", -1)),
            "run_id": output.name,
            "initial_state_sha256": next(iter(initial_hashes)),
            "source_checkpoints": [str(path.relative_to(ROOT)) for path in paths],
            "source_weights": normalized_weights.tolist(),
        },
        checkpoint,
    )
    metrics = evaluate_validation(checkpoint, config_path)
    write_json(output / "metrics.json", metrics)
    model = FrozenRetriever(checkpoint)
    write_json(
        output / "provenance.json",
        {
            "method": "uniform_model_soup" if args.weights is None else "weighted_model_soup",
            "test_evaluated": False,
            "same_initialization_verified": True,
            "initial_state_sha256": next(iter(initial_hashes)),
            "blends": blends,
            "sources": [
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256(path),
                    "run_id": payload.get("run_id"),
                    "seed": payload.get("seed"),
                    "weight": float(weight),
                }
                for path, payload, weight in zip(
                    paths, payloads, normalized_weights, strict=True
                )
            ],
            "checkpoint_sha256": sha256(checkpoint),
            "averaged_state_sha256": state_sha256(model),
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
