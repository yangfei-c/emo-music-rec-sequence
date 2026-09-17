"""Train the deployed music-VA head from scratch, then calibrate music changes.

The same RegressionRelation class is used here and in production. No Ours
checkpoint initializes this head. Test recordings are never used for training,
early stopping, temperature fitting, or preference-neighborhood construction.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.relations import relation_metrics
from emomm_sequence.losses import VALoss
from emomm_sequence.metrics import compute_va_metrics
from emomm_sequence.relation import RegressionRelation, relation_log_probabilities
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.calibration import fit_axis_temperatures
from emomm_sequence.training.pairs import build_pair_batch, training_thresholds
from emomm_sequence.training.records import check_identity_splits, sha256, state_sha256, training_provenance, write_json
from emomm_sequence.training.utilities import load_yaml, set_seed


def validation_partition(groups: np.ndarray, seed: int) -> dict[str, str]:
    ordered = sorted(set(groups.astype(str)), key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    n_model, n_cal = round(len(ordered) * 0.4), round(len(ordered) * 0.3)
    return {g: "validation_model" if i < n_model else "calibration" if i < n_model + n_cal else "policy" for i, g in enumerate(ordered)}


def prepare_data(config: dict, output: Path) -> dict:
    import json
    blocks = []
    for source, relative in config["data"]["sources"].items():
        with np.load(ROOT / relative, allow_pickle=False) as data:
            blocks.append({"audio_embeddings": data["audio_embeddings"], "targets": (data["va_targets"] + 1.0) / 2.0,
                           "ids": data["global_identities"].astype(str), "groups": data["content_group_ids"].astype(str),
                           "splits": data["splits"].astype(str), "sources": np.repeat(source, len(data["splits"]))})
    merged = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    check_identity_splits(merged["groups"], merged["splits"], "music VA content groups")
    if len(set(merged["groups"])) != len(merged["groups"]):
        raise ValueError("Music features must be deduplicated before both training stages")
    if not np.isfinite(merged["targets"]).all() or not ((merged["targets"] >= 0) & (merged["targets"] <= 1)).all():
        raise ValueError("Music-VA targets outside [0,1]")
    partitions = merged["splits"].astype("<U20")
    for source in np.unique(merged["sources"]):
        mask = (merged["sources"] == source) & (merged["splits"] == "validation")
        mapping = validation_partition(merged["groups"][mask], config["data"]["subsplit_seed"])
        partitions[mask] = [mapping[g] for g in merged["groups"][mask]]
    merged["partitions"] = partitions
    runs_root = ROOT / config["data"]["retriever_runs"]
    runs = json.loads((runs_root / "aggregate.json").read_text(encoding="utf-8"))["runs"]
    selected = max(runs, key=lambda run: (run["best_validation_score"], -run["seed"]))
    checkpoint = runs_root / f"seed_{selected['seed']}" / "best.pt"
    # Neighbors are sampled separately within each partition, without using VA.
    model = FrozenRetriever(checkpoint).eval()
    with torch.inference_mode():
        merged["preference_embeddings"] = np.concatenate([model.encode_factor("music", "preference", torch.from_numpy(merged["audio_embeddings"][i:i+512])).numpy() for i in range(0, len(partitions), 512)])
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "prepared_data.npz", **merged)
    write_json(output / "data_audit.json", {
        "target_transform": "(feature_target_minus1_plus1 + 1) / 2; DEAM and PMEmo to [0,1]",
        "retriever_seed": selected["seed"], "retriever_checkpoint_sha256": sha256(checkpoint),
        "selection_uses_test": False, "content_groups_unique": True,
        "source_hashes": {relative: sha256(ROOT / relative) for relative in config["data"]["sources"].values()},
        "counts": {s: {p: int(((merged["sources"] == s) & (partitions == p)).sum()) for p in np.unique(partitions)} for s in np.unique(merged["sources"])}})
    return merged


@torch.inference_mode()
def coordinates(model: RegressionRelation, data: dict, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate([model.coordinates(torch.from_numpy(data["audio_embeddings"][i:i+512]).to(device)).cpu().numpy() for i in range(0, len(data["ids"]), 512)])


def pair_probabilities(prediction: np.ndarray, pairs, small, big, temperature) -> np.ndarray:
    delta = torch.from_numpy(prediction[pairs.right] - prediction[pairs.left])
    with torch.inference_mode():
        return relation_log_probabilities(delta, torch.as_tensor(small, dtype=torch.float32), torch.as_tensor(big, dtype=torch.float32), torch.as_tensor(temperature, dtype=torch.float32)).exp().numpy()


def train_seed(config: dict, data: dict, pairs: dict, small: np.ndarray, big: np.ndarray, seed: int, output: Path) -> dict:
    set_seed(seed)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = RegressionRelation(**config["model"]).to(device)
    output.mkdir(parents=True, exist_ok=True)
    provenance = training_provenance(model, seed, config)
    provenance["initialization"] = "fresh_random_no_checkpoint"
    torch.save({"model": model.state_dict(), "seed": seed}, output / "initial.pt")
    write_json(output / "provenance.json", provenance)
    cfg = config["training"]
    loss_fn = VALoss(beta=cfg["huber_beta"], ccc_weight=cfg["ccc_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    mask = data["partitions"] == "train"
    _, domains = np.unique(data["sources"][mask], return_inverse=True)
    dataset = TensorDataset(torch.from_numpy(data["audio_embeddings"][mask]), torch.from_numpy(data["targets"][mask]), torch.from_numpy(domains))
    weights = torch.as_tensor(1.0 / np.bincount(domains)[domains], dtype=torch.float64)
    best, reference, bad, best_epoch = float("inf"), float("inf"), 0, 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=torch.Generator().manual_seed(seed + epoch))
        loader = DataLoader(dataset, sampler=sampler, batch_size=cfg["batch_size"], num_workers=0)
        model.train()
        losses = []
        for x, y, source in loader:
            x, y, source = x.to(device), y.to(device), source.to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted = model.coordinates(x)
            loss = torch.stack([loss_fn(predicted[source == value], y[source == value]) for value in source.unique()]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite VA loss seed={seed} epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.item()))
        predicted = coordinates(model, data, device)
        val = pairs["validation_model"]
        selection = relation_metrics(pair_probabilities(predicted, val, small, big, np.ones(2)), val.labels5, val.source, val.mode)["source_axis_equal_nll"]
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_relation_nll": selection})
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print(f"Music VA seed={seed} epoch={epoch} loss={np.mean(losses):.5f} validation NLL={selection:.5f}", flush=True)
        if selection < best:
            best, best_epoch = selection, epoch
            torch.save({"model": model.state_dict(), "seed": seed, "epoch": epoch, "selection_score": selection, "run_id": config["run_id"], "initial_state_sha256": provenance["initial_state_sha256"]}, output / "best.pt")
        if selection < reference - cfg["early_stopping_min_delta"]:
            reference, bad = selection, 0
        else:
            bad += 1
        if bad >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"], strict=True)
    predicted = coordinates(model, data, device)
    cal = pairs["calibration"]
    delta = torch.from_numpy(predicted[cal.right] - predicted[cal.left])
    temperature = fit_axis_temperatures(lambda values, temp: relation_log_probabilities(values, torch.as_tensor(small), torch.as_tensor(big), temp), delta, cal.labels5,
        grid=np.exp(np.linspace(np.log(config["calibration"]["temperature_min"]), np.log(config["calibration"]["temperature_max"]), config["calibration"]["grid_size"])))
    calibration = {"temperature_v": float(temperature[0]), "temperature_a": float(temperature[1]), "fixed_small_bounds": small.tolist(), "fixed_big_bounds": big.tolist(), "fitted_partition": "calibration", "run_id": config["run_id"]}
    write_json(output / "calibration.json", calibration)
    results = {}
    for partition, batch in pairs.items():
        probabilities = pair_probabilities(predicted, batch, small, big, temperature)
        chosen = data["partitions"] == partition
        results[partition] = {
            "relations": relation_metrics(probabilities, batch.labels5, batch.source, batch.mode),
            "va": {source: compute_va_metrics(predicted[chosen & (data["sources"] == source)], data["targets"][chosen & (data["sources"] == source)]) for source in np.unique(data["sources"])}}
        if partition == "test":
            np.savez_compressed(output / "test_predictions.npz", ids=data["ids"][chosen], sources=data["sources"][chosen], predictions=predicted[chosen], targets=data["targets"][chosen], relation_probabilities=probabilities, relation_labels=batch.labels5, relation_sources=batch.source)
    provenance.update({"best_epoch": best_epoch, "best_validation_nll": best, "checkpoint_sha256": sha256(output / "best.pt"), "final_state_sha256": state_sha256(model)})
    if provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
        raise RuntimeError("VA training left parameters unchanged")
    write_json(output / "provenance.json", provenance)
    result = {"model": "Ours", "component": "music_va_relation", "seed": seed, "best_epoch": best_epoch, "best_validation_nll": best, "results": results}
    write_json(output / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/va_relation.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_yaml(args.config)
    output = args.output.resolve()
    seeds = args.seeds or config["seeds"]
    if any((output / f"seed_{seed}" / "best.pt").exists() for seed in seeds):
        raise FileExistsError("Existing VA run must not be overwritten; use a new --output")
    data = prepare_data(config, output)
    thresholds = training_thresholds(data["targets"], data["sources"], data["partitions"])
    small = np.mean([value["small"] for value in thresholds.values()], axis=0).astype(np.float32)
    big = np.mean([value["big"] for value in thresholds.values()], axis=0).astype(np.float32)
    cfg = config["pairs"]
    pairs = {partition: build_pair_batch(data["targets"], data["sources"], data["partitions"], data["preference_embeddings"], thresholds, partition=partition, seed=seed, max_pairs=cfg["max_pairs_per_source"], max_pairs_per_song=cfg["max_pairs_per_song"], neighbor_fraction=cfg["neighbor_fraction"]) for partition, seed in cfg["seeds"].items()}
    for partition, batch in pairs.items():
        np.savez_compressed(output / f"pairs_{partition}.npz", left=batch.left, right=batch.right, sources=batch.source, modes=batch.mode, labels=batch.labels5)
    write_json(output / "effective_config.json", config)
    write_json(output / "training_thresholds.json", thresholds)
    runs = [train_seed(config, data, pairs, small, big, seed, output / f"seed_{seed}") for seed in seeds]
    write_json(output / "aggregate.json", {"model": "Ours", "component": "music_va_relation", "seeds": seeds, "test_used_for_selection": False, "runs": runs})
    print(f"Music VA training complete: {len(runs)} fresh seeds", flush=True)


if __name__ == "__main__":
    main()
