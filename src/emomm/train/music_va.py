"""Train fresh ordered music VA quantiles on the supplied train/validation splits."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from ..data import read_features
from ..models.music_va import VAQuantileRegressor
from .. import require_device
from . import check_identity_splits, compute_va_metrics, set_seed, sha256, state_sha256, write_json
from .losses import pinball_loss


def prepare_data(paths, excluded=None):
    blocks = []
    excluded = excluded or {}
    for source, path in paths.items():
        values = read_features(path)
        keep = ~np.isin(values['ids'].astype(str), excluded.get(f'music_{source}', []))
        for key in ('audio_embeddings', 'has_va', 'va_targets', 'splits', 'global_identities'):
            values[key] = values[key][keep]
        if not values["has_va"].all():
            raise ValueError(f"{source} 存在缺失 VA 标签")
        targets = values["va_targets"].astype(np.float32)
        if not np.isfinite(targets).all() or np.any(np.abs(targets) > 1):
            raise ValueError(f"{source} VA 标签应为 [-1,1]")
        blocks.append(dict(features=values["audio_embeddings"].astype(np.float32),
                           targets=(targets + 1) / 2, splits=values["splits"],
                           ids=values["global_identities"], sources=np.repeat(source, len(targets))))
    result = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    check_identity_splits(result["ids"], result["splits"], "MusicVA")
    for source in paths:
        for split in ("train", "validation"):
            if ((result["sources"] == source) & (result["splits"] == split)).sum() < 2:
                raise ValueError(f"{source}/{split} 样本不足")
    return result


@torch.inference_mode()
def predict(model, features, device):
    model.eval()
    return np.concatenate([model(torch.from_numpy(features[i:i + 512]).to(device)).cpu().numpy()
                           for i in range(0, len(features), 512)])


def validation_metrics(quantiles, targets, sources):
    result = {}
    for source in sorted(set(sources)):
        mask = sources == source
        q, y = quantiles[mask], targets[mask]
        result[source] = {"pinball": float(pinball_loss(torch.from_numpy(q), torch.from_numpy(y))),
                          "n_songs": int(mask.sum()),
                          "median_va": compute_va_metrics(2 * q[:, :, 1] - 1, 2 * y - 1)}
    result["source_equal_pinball"] = float(np.mean([v["pinball"] for v in result.values()]))
    return result


def train_seed(config, data, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    device = require_device(config["device"])
    model = VAQuantileRegressor(**config["model"]).to(device)
    initial = state_sha256(model)
    cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    train, validation = data["splits"] == "train", data["splits"] == "validation"
    dataset = TensorDataset(torch.from_numpy(data["features"][train]), torch.from_numpy(data["targets"][train]))
    _, domains = np.unique(data["sources"][train], return_inverse=True)
    weights = torch.from_numpy(1.0 / np.bincount(domains)[domains])
    best, reference, bad, best_epoch, history = float("inf"), float("inf"), 0, 0, []
    for epoch in range(1, cfg["max_epochs"] + 1):
        generator = torch.Generator().manual_seed(seed + epoch)
        sampler = WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator) if cfg["source_balanced_sampling"] else None
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], sampler=sampler,
                            shuffle=sampler is None, generator=generator)
        model.train()
        total, count = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = pinball_loss(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("MusicVA 损失含非有限值")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(x)
            count += len(x)
        metrics = validation_metrics(predict(model, data["features"][validation], device),
                                     data["targets"][validation], data["sources"][validation])
        score = metrics["source_equal_pinball"]
        history.append(dict(epoch=epoch, train_pinball=total / count, validation_source_equal_pinball=score))
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(model=model.state_dict(), model_config=config["model"], seed=seed,
                            epoch=epoch, run_id=config["run_id"], feature_id=config["feature_id"],
                            selection_score=score, target_scale=[0, 1], quantile_levels=[0.05, 0.5, 0.95]), output / "best.pt")
        if score < reference - cfg["early_stopping_min_delta"]:
            reference, bad = score, 0
        else:
            bad += 1
        print(f"MusicVA seed={seed} epoch={epoch} train={total/count:.5f} validation={score:.5f}", flush=True)
        if bad >= cfg["early_stopping_patience"]:
            break
    model.load_state_dict(torch.load(output / "best.pt", weights_only=True, map_location=device)["model"], strict=True)
    final = state_sha256(model)
    if final == initial:
        raise RuntimeError("MusicVA 参数未发生更新")
    result = dict(seed=seed, best_epoch=best_epoch, best_validation_score=best,
                  results={"validation": validation_metrics(predict(model, data["features"][validation], device),
                           data["targets"][validation], data["sources"][validation])},
                  initialization="fresh_random", initial_state_sha256=initial, final_state_sha256=final,
                  checkpoint_sha256=sha256(output / "best.pt"), calibration_fitted=False, test_used=False)
    write_json(output / "metrics.json", result)
    return result
