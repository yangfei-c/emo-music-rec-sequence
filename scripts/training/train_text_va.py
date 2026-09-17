"""Fit EmoBank VA after retrieval training, without altering any Adapter."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.losses import VALoss
from emomm_sequence.metrics import compute_va_metrics
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.text_va import TextVAHead, load_text_va
from emomm_sequence.training.records import check_identity_splits, sha256, state_sha256, training_provenance, write_json
from emomm_sequence.training.utilities import load_yaml, set_seed


def prepare_data(config: dict, output: Path) -> tuple[dict, str]:
    path = ROOT / config["data"]["features"]
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    check_identity_splits(data["global_identities"], data["splits"], "EmoBank")
    target = data["va_targets"]
    if not np.isfinite(target).all() or not ((target >= -1) & (target <= 1)).all():
        raise ValueError("Expected EmoBank targets normalized to [-1,1]")
    runs_root = ROOT / config["data"]["retriever_runs"]
    runs = json.loads((runs_root / "aggregate.json").read_text(encoding="utf-8"))["runs"]
    selected = max(runs, key=lambda r: (r["best_validation_score"], -r["seed"]))
    checkpoint = runs_root / f"seed_{selected['seed']}" / "best.pt"
    retriever = FrozenRetriever(checkpoint).eval()
    before = state_sha256(retriever)
    with torch.inference_mode():
        data["features"] = torch.cat([
            retriever.encode_text("affect", torch.from_numpy(data["text_embeddings"][i:i+512].astype(np.float32)))
            for i in range(0, len(target), 512)
        ]).numpy()
    assert state_sha256(retriever) == before
    digest = sha256(checkpoint)
    write_json(output / "data_audit.json", {
        "source": str(path.relative_to(ROOT)), "source_sha256": sha256(path),
        "retriever_seed": selected["seed"], "retriever_checkpoint_sha256": digest,
        "input": "frozen CLaMP3 -> frozen selected text affect Adapter -> trainable Linear(768,2) -> tanh",
        "target_scale": [-1, 1], "adapter_optimized": False,
        "counts": {s: int((data["splits"] == s).sum()) for s in ("train", "validation", "test")},
    })
    return data, digest


def train_seed(config: dict, data: dict, digest: str, seed: int, output: Path) -> dict:
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = TextVAHead(**config["model"]).to(device)
    provenance = training_provenance(model, seed, config)
    provenance.update(initialization="fresh_random_linear_head", retriever_checkpoint_sha256=digest,
                      frozen_input_transform="selected text affect Adapter", adapter_optimized=False)
    torch.save({"model": model.state_dict(), "seed": seed}, output / "initial.pt")
    cfg = config["training"]
    criterion = VALoss(beta=cfg["huber_beta"], ccc_weight=cfg["ccc_weight"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    x = torch.from_numpy(data["features"].astype(np.float32))
    y = torch.from_numpy(data["va_targets"].astype(np.float32))
    train = data["splits"] == "train"
    validation = data["splits"] == "validation"
    loader = DataLoader(TensorDataset(x[train], y[train]), batch_size=cfg["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, reference, bad, best_epoch = -float("inf"), -float("inf"), 0, 0
    history = []
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb.to(device)), yb.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite text VA loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.item()))
        model.eval()
        with torch.inference_mode():
            prediction = model(x[validation].to(device)).cpu()
            val_loss = float(criterion(prediction, y[validation]))
        metrics = compute_va_metrics(prediction, y[validation])
        score = metrics["va_score"]
        history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_loss": val_loss,
                        "validation_mean_ccc": score})
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "model_config": config["model"], "seed": seed,
                        "epoch": epoch, "retriever_checkpoint_sha256": digest, "target_scale": [-1, 1]}, output / "best.pt")
        if score > reference + cfg["early_stopping_min_delta"]:
            reference, bad = score, 0
        else:
            bad += 1
        print(f"Text VA seed={seed} epoch={epoch} loss={np.mean(losses):.5f} validation CCC={score:.5f}", flush=True)
        if bad >= cfg["early_stopping_patience"]:
            break
    loaded = load_text_va(output / "best.pt", digest)
    with torch.inference_mode():
        prediction = loaded(x).numpy()
    results = {s: compute_va_metrics(prediction[data["splits"] == s], y[data["splits"] == s])
               for s in ("train", "validation", "test")}
    provenance.update(best_epoch=best_epoch, checkpoint_sha256=sha256(output / "best.pt"),
                      final_state_sha256=state_sha256(loaded), save_reload_verified=True)
    if provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
        raise RuntimeError("Text VA parameters did not change")
    write_json(output / "provenance.json", provenance)
    mask = data["splits"] == "test"
    np.savez_compressed(output / "test_predictions.npz", ids=data["global_identities"][mask],
                        predictions=prediction[mask], targets=y[mask].numpy())
    result = {"seed": seed, "best_epoch": best_epoch, "best_validation_score": best, "results": results}
    write_json(output / "metrics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training/text_va.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_yaml(args.config)
    seeds = args.seeds or config["seeds"]
    if any((args.output / f"seed_{seed}").exists() for seed in seeds):
        raise FileExistsError("Use a new output directory; do not overwrite a text VA run")
    data, digest = prepare_data(config, args.output)
    write_json(args.output / "effective_config.json", config)
    runs = [train_seed(config, data, digest, seed, args.output / f"seed_{seed}") for seed in seeds]
    write_json(args.output / "aggregate.json", {"component": "text_va", "seeds": seeds, "runs": runs,
               "selected_seed": max(runs, key=lambda r: (r["best_validation_score"], -r["seed"]))["seed"],
               "test_used_for_selection": False})


if __name__ == "__main__":
    main()
