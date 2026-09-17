"""Interpolate one independently trained factor between compatible checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.training.records import sha256, write_json


FACTORS = ("semantic", "affect", "preference")


def factor_keys(state: dict[str, torch.Tensor], factor: str) -> list[str]:
    prefixes = (
        f"text_adapters.{factor}.",
        f"music_adapters.{factor}.",
        f"logit_scales.{factor}",
    )
    keys = [key for key in state if key.startswith(prefixes)]
    if factor == "affect":
        keys.extend(
            key
            for key in state
            if key.startswith("text_va_head.") or key.startswith("music_va_head.")
        )
    return sorted(set(keys))


def load(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model", "model_config", "seed", "initial_state_sha256"}
    if not required.issubset(payload):
        raise ValueError(f"checkpoint lacks compatibility metadata: {path}")
    return payload


def interpolate_state(
    base: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    factor: str,
    candidate_weight: float,
) -> dict[str, torch.Tensor]:
    if set(base) != set(candidate):
        raise ValueError("checkpoint state dictionaries have different keys")
    if factor not in FACTORS:
        raise ValueError(f"unknown factor: {factor}")
    if not 0.0 <= candidate_weight <= 1.0:
        raise ValueError("candidate weight must be in [0, 1]")
    state = {key: value.detach().cpu().clone() for key, value in base.items()}
    keys = factor_keys(state, factor)
    if not keys:
        raise ValueError(f"no {factor} parameters found")
    for key in keys:
        left = base[key].detach().cpu()
        right = candidate[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"incompatible tensor for interpolation: {key}")
        if not torch.is_floating_point(left):
            if not torch.equal(left, right):
                raise ValueError(f"non-floating factor state differs: {key}")
            state[key] = left.clone()
        else:
            state[key] = torch.lerp(left, right, float(candidate_weight))
    return state


def tensor_state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--factor", choices=FACTORS, required=True)
    parser.add_argument("--candidate-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_path = args.base if args.base.is_absolute() else ROOT / args.base
    candidate_path = (
        args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    base, candidate = load(base_path), load(candidate_path)
    if base["model_config"] != candidate["model_config"]:
        raise ValueError("checkpoints use different model architectures")
    if int(base["seed"]) != int(candidate["seed"]):
        raise ValueError("checkpoints use different seeds")
    if base["initial_state_sha256"] != candidate["initial_state_sha256"]:
        raise ValueError("checkpoints do not share the same initialization")
    state = interpolate_state(
        base["model"],
        candidate["model"],
        args.factor,
        float(args.candidate_weight),
    )
    checkpoint = output / "best.pt"
    payload = {
        **base,
        "model": state,
        "run_id": output.name,
        "selection_metric": "validation_selected_factor_weight_interpolation",
        "interpolation": {
            "factor": args.factor,
            "candidate_weight": float(args.candidate_weight),
            "base": str(base_path.relative_to(ROOT)),
            "candidate": str(candidate_path.relative_to(ROOT)),
        },
    }
    torch.save(payload, checkpoint)
    write_json(
        output / "provenance.json",
        {
            "test_evaluated": False,
            "factor": args.factor,
            "candidate_weight": float(args.candidate_weight),
            "same_initialization": True,
            "base_sha256": sha256(base_path),
            "candidate_sha256": sha256(candidate_path),
            "checkpoint_sha256": sha256(checkpoint),
            "state_sha256": tensor_state_sha256(state),
        },
    )
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "factor": args.factor,
                "candidate_weight": float(args.candidate_weight),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
