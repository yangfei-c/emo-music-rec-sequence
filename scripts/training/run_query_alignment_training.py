"""Train MTG preference adapters against mapped and runtime query forms."""

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


RUN_ID = "diagnosis_20260914_query_alignment"
REPORT = ROOT / "artifacts/runs" / RUN_ID / "evaluation"
RUNS = ROOT / "artifacts" / "runs" / RUN_ID
MAPPED = ROOT / "artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz"
QWEN = ROOT / "artifacts/runs/diagnosis_20260914_tower_freeze/evaluation/query_form_embeddings.npz"
SEEDS = (133, 134, 135)
EPOCHS = 8
VARIANTS: dict[str, tuple[str, ...]] = {
    "legacy_only": ("legacy",),
    "canonical_only": ("canonical",),
    "mapped_three_view": ("canonical", "mapped_paraphrase_1", "mapped_paraphrase_2"),
    "mapped_plus_qwen": (
        "canonical",
        "mapped_paraphrase_1",
        "mapped_paraphrase_2",
        "qwen_runtime",
    ),
}
FORMS = (
    "legacy",
    "canonical",
    "mapped_paraphrase_1",
    "mapped_paraphrase_2",
    "qwen_runtime",
)
PRIMARY_FORMS = ("canonical", "qwen_runtime")
METRICS = ("ndcg_at_10", "average_precision", "precision_at_10", "recall_at_10")


def load_inputs(config: dict) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    with np.load(MAPPED, allow_pickle=False) as archive:
        mtg = {key: archive[key] for key in archive.files}
    with np.load(ROOT / config["supervision"]["music_index"], allow_pickle=False) as archive:
        music = archive["embeddings"].astype(np.float32)
    with np.load(QWEN, allow_pickle=False) as archive:
        qwen = {key: archive[key] for key in archive.files}
    labels = mtg["preference_labels"].astype(str)
    if not np.array_equal(labels, qwen["labels"].astype(str)):
        raise ValueError("Qwen query labels do not align with mapped MTG labels")
    views = mtg["preference_query_embeddings"].astype(np.float32)
    forms = {
        "legacy": mtg["preference_legacy_label_embeddings"].astype(np.float32),
        "canonical": views[:, 0],
        "mapped_paraphrase_1": views[:, 1],
        "mapped_paraphrase_2": views[:, 2],
        "qwen_runtime": qwen["qwen_preference"].astype(np.float32),
    }
    return mtg, music, forms


def make_train_loader(
    mtg: dict[str, np.ndarray], music: np.ndarray, batch_size: int, seed: int
) -> DataLoader:
    mask = mtg["splits"].astype(str) == "train"
    return DataLoader(
        TensorDataset(
            torch.from_numpy(music[mtg["index_positions"][mask]].astype(np.float32)),
            torch.from_numpy(mtg["preference_targets"][mask].astype(bool)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=True,
    )


def build_model(config: dict, device: torch.device) -> RetrieverModel:
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
    for factor in ("semantic", "affect"):
        model.text_adapters[factor].requires_grad_(False)
        model.music_adapters[factor].requires_grad_(False)
    model.text_va_head.requires_grad_(False)
    model.music_va_head.requires_grad_(False)
    model.factor_logits.requires_grad_(False)
    return model


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[str, list[int]]:
    train = mtg["splits"].astype(str) == "train"
    frequency = mtg["preference_targets"][train].sum(axis=0)
    labels = mtg["preference_labels"].astype(str)
    order = sorted(range(len(labels)), key=lambda index: (frequency[index], labels[index]))
    return {
        name: values.tolist()
        for name, values in zip(("tail", "mid", "head"), np.array_split(order, 3))
    }


def compact(metrics: dict, groups: dict[str, list[int]]) -> dict:
    by_index = {int(row["query_index"]): row for row in metrics["per_query"]}
    return {
        "overall": {key: float(metrics["macro"][key]) for key in METRICS},
        "frequency_groups": {
            group: {
                key: float(np.mean([by_index[index][key] for index in indices]))
                for key in METRICS
            }
            for group, indices in groups.items()
        },
    }


@torch.inference_mode()
def evaluate_forms(
    model: RetrieverModel,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    forms: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...],
    groups: dict[str, list[int]],
) -> dict[str, dict]:
    validation = mtg["splits"].astype(str) == "validation"
    raw_music = music[mtg["index_positions"][validation]].astype(np.float32)
    positives = mtg["preference_targets"][validation].astype(bool).T
    model.eval()
    music_parts = []
    for start in range(0, len(raw_music), batch_size):
        batch = torch.from_numpy(raw_music[start : start + batch_size]).to(device)
        music_parts.append(
            model.encode_factor("music", "preference", batch).float().cpu().numpy()
        )
    adapted_music = np.concatenate(music_parts)
    result = {}
    for name, raw_text in forms.items():
        text_parts = []
        for start in range(0, len(raw_text), batch_size):
            batch = torch.from_numpy(raw_text[start : start + batch_size]).to(device)
            text_parts.append(
                model.encode_factor("text", "preference", batch).float().cpu().numpy()
            )
        scores = np.concatenate(text_parts) @ adapted_music.T
        result[name] = compact(label_to_track_metrics(scores, positives, ks), groups)
    return result


def selection_score(evaluation: dict[str, dict]) -> float:
    return float(
        np.mean(
            [
                tag_balanced_selection_score(
                    evaluation[form]["overall"]["ndcg_at_10"],
                    evaluation[form]["overall"]["average_precision"],
                )
                for form in PRIMARY_FORMS
            ]
        )
    )


def checkpoint_config(config: dict) -> dict:
    values = copy.deepcopy(config["model"])
    values.update(
        adapter_drop_probability=0.0,
        adapter_drop_probabilities={factor: 0.0 for factor in ("semantic", "affect", "preference")},
        semantic_output_blend=1.0,
        affect_output_blend=1.0,
        preference_output_blend=1.0,
    )
    return values


def train_one(
    config: dict,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    forms: dict[str, np.ndarray],
    variant: str,
    active_forms: tuple[str, ...],
    seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training = config["training"]
    batch_size = int(training["batch_size_mtg"])
    model = build_model(config, device)
    train_text = {
        name: torch.from_numpy(forms[name]).to(device) for name in active_forms
    }
    loader = make_train_loader(mtg, music, batch_size, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and bool(training["amp"])
    )
    groups = frequency_groups(mtg)
    ks = tuple(int(value) for value in config["evaluation"]["tag_ks"])
    initial = evaluate_forms(model, mtg, music, forms, device, batch_size, ks, groups)
    best_score, best_epoch = selection_score(initial), 0
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict] = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for music_batch, target in loader:
            music_batch, target = music_batch.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                audio = model.encode_factor("music", "preference", music_batch)
                scale = model.logit_scales["preference"].exp().clamp(max=100.0)
                losses = []
                for query in train_text.values():
                    text = model.encode_factor("text", "preference", query)
                    logits = (text @ audio.T) * scale
                    positive = target.T
                    losses.append(
                        0.5
                        * (
                            multi_positive_cross_entropy(logits, positive, dim=1)
                            + multi_positive_cross_entropy(logits, positive, dim=0)
                        )
                    )
                loss = torch.stack(losses).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                parameters, float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
        scheduler.step()
        evaluation = evaluate_forms(
            model, mtg, music, forms, device, batch_size, ks, groups
        )
        score = selection_score(evaluation)
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(loader),
                "selection_score": score,
                "canonical": evaluation["canonical"]["overall"],
                "qwen_runtime": evaluation["qwen_runtime"]["overall"],
            }
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{variant} seed={seed} epoch={epoch} loss={total_loss / len(loader):.4f} "
            f"canonical_NDCG={evaluation['canonical']['overall']['ndcg_at_10']:.4f} "
            f"qwen_NDCG={evaluation['qwen_runtime']['overall']['ndcg_at_10']:.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    final = evaluate_forms(model, mtg, music, forms, device, batch_size, ks, groups)
    run_dir = RUNS / variant / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in best_state.items()},
            "model_config": checkpoint_config(config),
            "embedding_dim": int(config["backbone"]["embedding_dim"]),
            "selection_metric": "mean_canonical_qwen_validation_balanced_score",
            "epoch": best_epoch,
            "selection_score": best_score,
            "seed": seed,
            "run_id": f"{RUN_ID}/{variant}/seed_{seed}",
            "active_query_forms": active_forms,
        },
        checkpoint,
    )
    result = {
        "variant": variant,
        "active_query_forms": active_forms,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "validation": final,
        "history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
        "test_evaluated": False,
    }
    write_json(run_dir / "metrics.json", result)
    return result


def prepare() -> None:
    if REPORT.exists() or RUNS.exists():
        raise FileExistsError(RUN_ID)
    config_path = ROOT / "configs/training/retriever.yaml"
    plan = {
        "run_id": RUN_ID,
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "validation",
        "test_evaluated": False,
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "variants": VARIANTS,
        "loss": "mean symmetric multi-positive contrastive loss over active query forms",
        "selection": "mean of sqrt(NDCG@10 * AP) over canonical and qwen_runtime",
        "fixed_protocol": "same MTG targets, candidate tracks, split, adapters, optimizer, and metrics",
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "mapped_supervision_sha256": sha256(MAPPED),
        "qwen_embeddings_sha256": sha256(QWEN),
    }
    write_json(REPORT / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run() -> None:
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    config_path = ROOT / "configs/training/retriever.yaml"
    checks = {
        config_path: plan["config_sha256"],
        Path(__file__): plan["script_sha256"],
        MAPPED: plan["mapped_supervision_sha256"],
        QWEN: plan["qwen_embeddings_sha256"],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise RuntimeError(f"protocol input changed after freeze: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mtg, music, forms = load_inputs(config)
    rows = []
    for variant, active_forms in VARIANTS.items():
        for seed in SEEDS:
            rows.append(
                train_one(config, mtg, music, forms, variant, active_forms, seed)
            )
    write_json(REPORT / "execution.json", rows)


def summarize() -> None:
    rows = json.loads((REPORT / "execution.json").read_text(encoding="utf-8"))
    by_variant = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in VARIANTS
    }
    aggregate: dict[str, dict] = {}
    for variant, selected in by_variant.items():
        aggregate[variant] = {}
        for form in FORMS:
            aggregate[variant][form] = {
                key: {
                    "mean": float(
                        np.mean([row["validation"][form]["overall"][key] for row in selected])
                    ),
                    "std": float(
                        np.std(
                            [row["validation"][form]["overall"][key] for row in selected],
                            ddof=1,
                        )
                    ),
                }
                for key in METRICS
            }
    baseline = {row["seed"]: row for row in by_variant["legacy_only"]}
    paired = {}
    for variant, selected in by_variant.items():
        if variant == "legacy_only":
            continue
        paired[variant] = {}
        for form in PRIMARY_FORMS:
            paired[variant][form] = {}
            for key in METRICS:
                deltas = np.asarray(
                    [
                        row["validation"][form]["overall"][key]
                        - baseline[row["seed"]]["validation"][form]["overall"][key]
                        for row in selected
                    ]
                )
                paired[variant][form][key] = {
                    "mean_delta": float(deltas.mean()),
                    "std_delta": float(deltas.std(ddof=1)),
                    "positive_seeds": int((deltas > 0).sum()),
                }
    summary = {
        "run_id": RUN_ID,
        "partition": "validation",
        "test_evaluated": False,
        "aggregate": aggregate,
        "paired_vs_legacy_only": paired,
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
