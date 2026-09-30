"""Fast validation-only localization of the MTG preference tower bottleneck."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.models.retriever import RetrieverModel
from emomm_sequence.losses.retrieval import multi_positive_cross_entropy
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.selection import tag_balanced_selection_score
from emomm_sequence.training.utilities import set_seed


RUN_ID = "diagnosis_20260914_tower_freeze"
REPORT = ROOT / "artifacts/runs" / RUN_ID / "evaluation"
RUNS = ROOT / "artifacts/runs" / RUN_ID
SEEDS = (133, 134, 135)
VARIANTS = {
    "both_frozen": {"train_text": False, "train_music": False},
    "music_only": {"train_text": False, "train_music": True},
    "text_only": {"train_text": True, "train_music": False},
    "both_train": {"train_text": True, "train_music": True},
}
METRIC_KEYS = ("ndcg_at_10", "average_precision", "precision_at_10", "recall_at_10")


def load_inputs(config: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    mtg_path = ROOT / config["supervision"]["mtg_factors"]
    music_path = ROOT / config["supervision"]["music_index"]
    with np.load(mtg_path, allow_pickle=False) as data:
        mtg = {key: data[key] for key in data.files}
    with np.load(music_path, allow_pickle=False) as data:
        music = data["embeddings"].astype(np.float32)
    return mtg, music


def make_train_loader(
    mtg: dict[str, np.ndarray], music: np.ndarray, batch_size: int, seed: int
) -> DataLoader:
    mask = mtg["splits"].astype(str) == "train"
    dataset = TensorDataset(
        torch.from_numpy(music[mtg["index_positions"][mask]].astype(np.float32)),
        torch.from_numpy(mtg["preference_targets"][mask].astype(bool)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=True,
    )


def build_model(config: dict, controls: dict[str, bool], device: torch.device) -> RetrieverModel:
    values = config["model"]
    model = RetrieverModel(
        embedding_dim=int(config["backbone"]["embedding_dim"]),
        bottleneck_dim=int(values["bottleneck_dim"]),
        dropout=float(values["dropout"]),
        adapter_drop_probability=0.0,
        adapter_drop_probabilities={factor: 0.0 for factor in ("semantic", "affect", "preference")},
        initial_adapter_scale=float(values["initial_adapter_scale"]),
        initial_factor_weights=tuple(float(value) for value in values["initial_factor_weights"]),
        initial_temperature=float(values["initial_temperature"]),
        train_factor_weights=False,
        adapter_sharing="none",
        adapter_kind="residual",
        semantic_output_blend=1.0,
        affect_output_blend=1.0,
        preference_output_blend=1.0,
    ).to(device)
    model.text_adapters["preference"].requires_grad_(controls["train_text"])
    model.music_adapters["preference"].requires_grad_(controls["train_music"])
    for factor in ("semantic", "affect"):
        model.text_adapters[factor].requires_grad_(False)
        model.music_adapters[factor].requires_grad_(False)
    model.text_va_head.requires_grad_(False)
    model.music_va_head.requires_grad_(False)
    model.factor_logits.requires_grad_(False)
    return model


@torch.inference_mode()
def evaluate(
    model: RetrieverModel,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...],
) -> dict:
    mask = mtg["splits"].astype(str) == "validation"
    text_raw = mtg["preference_label_embeddings"].astype(np.float32)
    music_raw = music[mtg["index_positions"][mask]].astype(np.float32)
    positive = mtg["preference_targets"][mask].astype(bool).T
    model.eval()
    text_parts, music_parts = [], []
    for start in range(0, len(text_raw), batch_size):
        batch = torch.from_numpy(text_raw[start : start + batch_size]).to(device)
        text_parts.append(model.encode_factor("text", "preference", batch).cpu().numpy())
    for start in range(0, len(music_raw), batch_size):
        batch = torch.from_numpy(music_raw[start : start + batch_size]).to(device)
        music_parts.append(model.encode_factor("music", "preference", batch).cpu().numpy())
    scores = np.concatenate(text_parts) @ np.concatenate(music_parts).T
    return label_to_track_metrics(scores, positive, ks)


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[str, list[int]]:
    train = mtg["splits"].astype(str) == "train"
    frequency = mtg["preference_targets"][train].sum(axis=0)
    labels = mtg["preference_labels"].astype(str)
    ordered = sorted(range(len(labels)), key=lambda index: (frequency[index], labels[index]))
    return {
        name: values.tolist()
        for name, values in zip(
            ("tail", "mid", "head"), np.array_split(np.asarray(ordered), 3)
        )
    }


def compact_metrics(metrics: dict, groups: dict[str, list[int]]) -> dict:
    rows = {int(row["query_index"]): row for row in metrics["per_query"]}
    grouped = {}
    for name, indices in groups.items():
        selected = [rows[index] for index in indices if index in rows]
        grouped[name] = {
            key: float(np.mean([row[key] for row in selected])) for key in METRIC_KEYS
        }
    return {
        "overall": {key: float(metrics["macro"][key]) for key in METRIC_KEYS},
        "frequency_groups": grouped,
    }


def train_one(
    config: dict,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    variant: str,
    controls: dict[str, bool],
    seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    batch_size = int(training["batch_size_mtg"])
    epochs = 6
    model = build_model(config, controls, device)
    labels = torch.from_numpy(mtg["preference_label_embeddings"].astype(np.float32)).to(device)
    loader = make_train_loader(mtg, music, batch_size, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(training["amp"]))
    ks = tuple(int(value) for value in config["evaluation"]["tag_ks"])
    groups = frequency_groups(mtg)
    run_dir = RUNS / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []

    initial_metrics = evaluate(model, mtg, music, device, batch_size, ks)
    best_score = tag_balanced_selection_score(
        initial_metrics["macro"]["ndcg_at_10"], initial_metrics["macro"]["average_precision"]
    )
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    if controls["train_text"] or controls["train_music"]:
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for music_batch, target in loader:
                music_batch, target = music_batch.to(device), target.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                    text = model.encode_factor("text", "preference", labels)
                    audio = model.encode_factor("music", "preference", music_batch)
                    logits = (text @ audio.T) * model.logit_scales["preference"].exp().clamp(max=100.0)
                    positive = target.T
                    loss = 0.5 * (
                        multi_positive_cross_entropy(logits, positive, dim=1)
                        + multi_positive_cross_entropy(logits, positive, dim=0)
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, float(training["gradient_clip_norm"]))
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.item())
            scheduler.step()
            metrics = evaluate(model, mtg, music, device, batch_size, ks)
            score = tag_balanced_selection_score(
                metrics["macro"]["ndcg_at_10"], metrics["macro"]["average_precision"]
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": total_loss / len(loader),
                    "selection_score": score,
                    **{key: float(metrics["macro"][key]) for key in METRIC_KEYS},
                }
            )
            if score > best_score:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"{variant} seed={seed} epoch={epoch} loss={total_loss / len(loader):.4f} "
                f"NDCG={metrics['macro']['ndcg_at_10']:.4f} AP={metrics['macro']['average_precision']:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    final_metrics = evaluate(model, mtg, music, device, batch_size, ks)
    model_config = copy.deepcopy(config["model"])
    model_config.update(
        adapter_drop_probability=0.0,
        adapter_drop_probabilities={factor: 0.0 for factor in ("semantic", "affect", "preference")},
        semantic_output_blend=1.0,
        affect_output_blend=1.0,
        preference_output_blend=1.0,
    )
    checkpoint = run_dir / "best.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in best_state.items()},
            "model_config": model_config,
            "embedding_dim": int(config["backbone"]["embedding_dim"]),
            "selection_metric": "preference_balanced_validation",
            "epoch": best_epoch,
            "selection_score": best_score,
            "seed": seed,
            "run_id": f"{RUN_ID}/{variant}/seed_{seed}",
        },
        checkpoint,
    )
    result = {
        "variant": variant,
        "controls": controls,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": compact_metrics(final_metrics, groups),
        "history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "test_evaluated": False,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def prepare() -> None:
    if REPORT.exists() or RUNS.exists():
        raise FileExistsError(f"diagnosis already exists: {RUN_ID}")
    config_path = ROOT / "configs/training/retriever.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    plan = {
        "run_id": RUN_ID,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "validation",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "variants": VARIANTS,
        "objective": "plain symmetric multi-positive contrastive loss only",
        "selection": "max over epoch 0..6 of sqrt(NDCG@10 * AP)",
        "frequency_groups": "equal label-count train-frequency thirds, frozen before evaluation",
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "mtg_sha256": sha256(ROOT / config["supervision"]["mtg_factors"]),
        "music_index_sha256": sha256(ROOT / config["supervision"]["music_index"]),
    }
    write_json(REPORT / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run() -> None:
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    config_path = ROOT / "configs/training/retriever.yaml"
    if sha256(config_path) != plan["config_sha256"] or sha256(Path(__file__)) != plan["script_sha256"]:
        raise RuntimeError("diagnosis code/config changed after protocol freeze")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mtg, music = load_inputs(config)
    rows = []
    for variant, controls in VARIANTS.items():
        for seed in SEEDS:
            rows.append(train_one(config, mtg, music, variant, controls, seed))
    write_json(REPORT / "execution.json", rows)


def summarize() -> None:
    rows = json.loads((REPORT / "execution.json").read_text(encoding="utf-8"))
    by_variant = {name: [row for row in rows if row["variant"] == name] for name in VARIANTS}
    aggregate = {}
    for name, variant_rows in by_variant.items():
        aggregate[name] = {"overall": {}, "frequency_groups": {}}
        for key in METRIC_KEYS:
            values = np.asarray([row["validation"]["overall"][key] for row in variant_rows])
            aggregate[name]["overall"][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        for group in ("tail", "mid", "head"):
            aggregate[name]["frequency_groups"][group] = {}
            for key in METRIC_KEYS:
                values = np.asarray([row["validation"]["frequency_groups"][group][key] for row in variant_rows])
                aggregate[name]["frequency_groups"][group][key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    baseline = {row["seed"]: row for row in by_variant["both_frozen"]}
    paired = {}
    for name in VARIANTS:
        if name == "both_frozen":
            continue
        paired[name] = {}
        for key in METRIC_KEYS:
            values = np.asarray([
                row["validation"]["overall"][key] - baseline[row["seed"]]["validation"]["overall"][key]
                for row in by_variant[name]
            ])
            paired[name][key] = {
                "mean_delta": float(values.mean()),
                "std_delta": float(values.std(ddof=1)),
                "positive_seeds": int((values > 1.0e-12).sum()),
            }
    conclusion = {
        name: all(paired[name][key]["mean_delta"] > 0 and paired[name][key]["positive_seeds"] >= 2 for key in ("ndcg_at_10", "average_precision"))
        for name in paired
    }
    summary = {
        "run_id": RUN_ID,
        "partition": "validation",
        "test_evaluated": False,
        "aggregate": aggregate,
        "paired_vs_both_frozen": paired,
        "improves_both_primary_metrics": conclusion,
    }
    write_json(REPORT / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "summarize"))
    args = parser.parse_args()
    {"prepare": prepare, "run": run, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
