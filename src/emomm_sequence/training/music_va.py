"""Train ordered VA quantiles and fit source-specific song-level joint CQR.

This command uses train for optimization, validation_model for early stopping,
and calibration for CQR. It deliberately never evaluates policy or test labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.data.music_va import prepare_data
from emomm_sequence.training.records import check_identity_splits, environment, sha256, state_sha256, write_json
from emomm_sequence.training.utilities import load_yaml, set_seed
from emomm_sequence.models.music_va import VAQuantileRegressor
from emomm_sequence.losses.va import pinball_loss
from emomm_sequence.calibration.cqr import apply_joint_cqr, fit_joint_cqr


@torch.inference_mode()
def predict_quantiles(model: VAQuantileRegressor, features: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate([
        model(torch.from_numpy(features[i:i + 512]).to(device)).cpu().numpy()
        for i in range(0, len(features), 512)
    ]).astype(np.float32)


def _ccc(target: np.ndarray, prediction: np.ndarray) -> float:
    mean_target, mean_prediction = float(target.mean()), float(prediction.mean())
    variance = float(target.var() + prediction.var() + (mean_target - mean_prediction) ** 2)
    if variance <= 1e-12:
        return 1.0 if np.allclose(target, prediction) else 0.0
    covariance = float(np.mean((target - mean_target) * (prediction - mean_prediction)))
    return float(np.clip(2.0 * covariance / variance, -1.0, 1.0))


def _validation_metrics(predictions: np.ndarray, targets: np.ndarray, sources: np.ndarray) -> dict:
    values = {}
    for source in sorted(set(sources)):
        selected = sources == source
        p, y = predictions[selected], targets[selected]
        pinball = float(pinball_loss(torch.from_numpy(p), torch.from_numpy(y)).item())
        median = p[:, :, 1]
        values[source] = {
            "n_songs": int(len(y)),
            "pinball": pinball,
            "median_mae_v": float(np.mean(np.abs(median[:, 0] - y[:, 0]))),
            "median_mae_a": float(np.mean(np.abs(median[:, 1] - y[:, 1]))),
            "median_ccc_v": _ccc(y[:, 0], median[:, 0]),
            "median_ccc_a": _ccc(y[:, 1], median[:, 1]),
        }
    values["source_equal_pinball"] = float(np.mean([values[source]["pinball"] for source in sorted(set(sources))]))
    return values


def _calibration_diagnostics(predictions: np.ndarray, targets: np.ndarray, sources: np.ndarray, calibration: dict) -> dict:
    _, lower, upper = apply_joint_cqr(predictions, calibration, sources)
    diagnostics = {}
    for source in sorted(set(sources)):
        selected = sources == source
        covered = (targets[selected] >= lower[selected]) & (targets[selected] <= upper[selected])
        diagnostics[source] = {
            "n_songs": int(selected.sum()),
            "empirical_joint_coverage_on_fitted_calibration": float(covered.all(axis=1).mean()),
            "empirical_v_coverage_on_fitted_calibration": float(covered[:, 0].mean()),
            "empirical_a_coverage_on_fitted_calibration": float(covered[:, 1].mean()),
            "mean_width_v": float((upper[selected, 0] - lower[selected, 0]).mean()),
            "mean_width_a": float((upper[selected, 1] - lower[selected, 1]).mean()),
        }
    return diagnostics


def train_seed(config: dict, data: dict, audit: dict, seed: int, output: Path, device: torch.device) -> dict:
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    model = VAQuantileRegressor(**config["model"]).to(device)
    initial_state = state_sha256(model)
    torch.save({"model": model.state_dict(), "seed": seed, "model_config": config["model"]}, output / "initial.pt")
    cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))
    train_mask = data["partitions"] == "train"
    validation_mask = data["partitions"] == "validation_model"
    calibration_mask = data["partitions"] == "calibration"
    _, domain_ids = np.unique(data["sources"][train_mask], return_inverse=True)
    dataset = TensorDataset(torch.from_numpy(data["embeddings"][train_mask]), torch.from_numpy(data["targets"][train_mask]))
    if cfg["source_balanced_sampling"]:
        weights = torch.as_tensor(1.0 / np.bincount(domain_ids)[domain_ids], dtype=torch.float64)
    else:
        weights = torch.ones(len(dataset), dtype=torch.float64)
    best_score, reference, bad_epochs, best_epoch = float("inf"), float("inf"), 0, 0
    history = []
    for epoch in range(1, int(cfg["max_epochs"]) + 1):
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=torch.Generator().manual_seed(seed + epoch))
        loader = DataLoader(dataset, sampler=sampler, batch_size=int(cfg["batch_size"]), num_workers=0)
        model.train()
        epoch_losses = []
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(features), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite quantile loss: seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["gradient_clip_norm"]))
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        val_predictions = predict_quantiles(model, data["embeddings"][validation_mask], device)
        validation = _validation_metrics(val_predictions, data["targets"][validation_mask], data["sources"][validation_mask])
        score = validation["source_equal_pinball"]
        history.append({"epoch": epoch, "train_pinball": float(np.mean(epoch_losses)), "validation_source_equal_pinball": score})
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"VA quantile seed={seed} epoch={epoch} train={history[-1]['train_pinball']:.5f} val={score:.5f}", flush=True)
        if score < best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "model": model.state_dict(), "model_config": config["model"],
                "seed": seed, "epoch": epoch, "selection_score": score,
                "run_id": config["run_id"], "initial_state_sha256": initial_state,
            }, output / "best.pt")
        if score < reference - float(cfg["early_stopping_min_delta"]):
            reference, bad_epochs = score, 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(cfg["early_stopping_patience"]):
            break

    checkpoint_path = output / "best.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model"], strict=True)
    final_state = state_sha256(model)
    if final_state == initial_state:
        raise RuntimeError("Quantile head parameters were not trained")
    val_predictions = predict_quantiles(model, data["embeddings"][validation_mask], device)
    validation = _validation_metrics(val_predictions, data["targets"][validation_mask], data["sources"][validation_mask])
    cal_predictions = predict_quantiles(model, data["embeddings"][calibration_mask], device)
    cal_targets = data["targets"][calibration_mask]
    cal_sources = data["sources"][calibration_mask]
    calibration = fit_joint_cqr(cal_predictions, cal_targets, cal_sources, float(config["calibration"]["alpha"]))
    calibration.update({
        "run_id": config["run_id"], "seed": seed, "fitted_partition": "calibration",
        "checkpoint_sha256": sha256(checkpoint_path),
        "calibration_group_sha256": {source: audit["partition_group_sha256"][source]["calibration"] for source in sorted(set(cal_sources))},
        "label_scope": "aggregate song-level music-expression VA; not individual listener emotion",
        "coverage_scope": "exchangeable new song from same source only; no selection-conditional or cross-domain guarantee",
    })
    write_json(output / "calibration.json", calibration)
    metrics = {
        "seed": seed, "best_epoch": best_epoch,
        "best_validation_source_equal_pinball": best_score,
        "validation_model": validation,
        "calibration_fit_diagnostics": _calibration_diagnostics(cal_predictions, cal_targets, cal_sources, calibration),
        "policy_and_test_evaluated": False,
    }
    write_json(output / "metrics.json", metrics)
    write_json(output / "provenance.json", {
        **environment(), "run_id": config["run_id"], "seed": seed,
        "initialization": "fresh_random_no_checkpoint", "initial_state_sha256": initial_state,
        "final_state_sha256": final_state, "checkpoint_sha256": sha256(checkpoint_path),
        "best_epoch": best_epoch, "validation_selection_only": True,
        "source_assets": audit["source_assets"],
        "partition_group_sha256": audit["partition_group_sha256"],
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/va_quantiles.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_yaml(args.config)
    device_choice = args.device or config["device"]
    if device_choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; pass --device cpu explicitly")
    device = torch.device(device_choice)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    seeds = args.seeds or config["seeds"]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Seed list contains duplicates")
    data, audit = prepare_data(config)
    output.mkdir(parents=True)
    write_json(output / "effective_config.json", config)
    write_json(output / "data_audit.json", audit)
    np.savez_compressed(
        output / "partition_manifest.npz",
        identities=data["identities"],
        content_group_ids=data["groups"],
        sources=data["sources"],
        partitions=data["partitions"],
    )
    results = [train_seed(config, data, audit, seed, output / f"seed_{seed}", device) for seed in seeds]
    selected = min(results, key=lambda result: (result["best_validation_source_equal_pinball"], result["seed"]))
    write_json(output / "aggregate.json", {
        "model": "Ours-Route2", "component": "music_va_quantiles", "run_id": config["run_id"],
        "seeds": seeds, "selected_seed_by_validation_only": selected["seed"],
        "selected_checkpoint": f"seed_{selected['seed']}/best.pt",
        "selected_calibration": f"seed_{selected['seed']}/calibration.json",
        "policy_and_test_evaluated": False, "runs": results,
    })
    print(f"VA quantile training complete: {len(results)} seeds; selected seed={selected['seed']} by validation", flush=True)


if __name__ == "__main__":
    main()
