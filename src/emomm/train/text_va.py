"""Fit EmoBank VA directly on frozen CLaMP3 text features, independently of S/A/P."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from emomm.train.losses import VALoss
from emomm.train import compute_va_metrics
from emomm import require_device
from emomm.data import read_features
from emomm.models.text_va import INPUT_REPRESENTATION, TextVAHead, load_text_va
from emomm.train import check_identity_splits, sha256, state_sha256, training_provenance, write_json
from emomm.train import set_seed


def prepare_data(config: dict) -> tuple[dict, str]:
    path = ROOT / config["data"]["features"]
    data = read_features(path)
    check_identity_splits(data["global_identities"], data["splits"], "EmoBank")
    target = data["va_targets"]
    if not np.isfinite(target).all() or not ((target >= -1) & (target <= 1)).all():
        raise ValueError("Expected EmoBank targets normalized to [-1,1]")
    if target.shape != (len(data["text_embeddings"]), 2) or not data["has_va"].all():
        raise ValueError("EmoBank features and VA labels must be complete and aligned")
    for split in ("train", "validation"):
        if (data["splits"].astype(str) == split).sum() < 2:
            raise ValueError(f"EmoBank/{split} needs at least two examples")
    data["features"] = data["text_embeddings"].astype(np.float32)
    digest = str(data["checkpoint_sha256"])
    return data, digest


def train_seed(config: dict, data: dict, digest: str, seed: int, output: Path) -> dict:
    set_seed(seed)
    output.mkdir(parents=True, exist_ok=False)
    device = require_device(config["device"])
    model = TextVAHead(**config["model"]).to(device)
    provenance = training_provenance(model, seed, config)
    provenance.update(initialization="fresh_random_linear_head", backbone_checkpoint_sha256=digest,
                      input_representation=INPUT_REPRESENTATION, adapter_optimized=False)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
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
                        "epoch": epoch, "backbone_checkpoint_sha256": digest,
                        "input_representation": INPUT_REPRESENTATION, "target_scale": [-1, 1],
                        "run_id": config["run_id"], "feature_id": config["feature_id"]}, output / "best.pt")
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
               for s in ("train", "validation")}
    provenance.update(best_epoch=best_epoch, checkpoint_sha256=sha256(output / "best.pt"),
                      final_state_sha256=state_sha256(loaded), save_reload_verified=True)
    if provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
        raise RuntimeError("Text VA parameters did not change")
    result = {"seed": seed, "best_epoch": best_epoch, "best_validation_score": best, "results": results, "training": provenance}
    write_json(output / "metrics.json", result)
    return result

