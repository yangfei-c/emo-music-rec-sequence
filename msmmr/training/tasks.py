from collections.abc import Callable

import numpy as np
import torch

from msmmr.metrics import compute_mtg_metrics, compute_va_metrics
from msmmr.training.engine import TaskSpec


def mtg_metrics(logits: np.ndarray, targets: np.ndarray, _: dict) -> dict:
    return compute_mtg_metrics(logits, targets)


def tag_prediction_payload(
    logits: np.ndarray, targets: np.ndarray, metrics: dict, _: dict
) -> dict:
    return {
        "logits": logits,
        "targets": targets,
        "per_label_ap": metrics["per_label_ap"],
        "per_label_roc_auc": metrics["per_label_roc_auc"],
    }


MTG_TASK = TaskSpec(
    name="music_mtg_full183",
    dataset_name="mtg",
    output_key="tag_logits",
    score_key="tag_score",
    metric_fn=mtg_metrics,
    default_print_every=100,
    metric_fields=(
        ("genre_mAP", "genre_map"),
        ("instrument_mAP", "instrument_map"),
        ("mood_mAP", "mood_map"),
        ("tag_score", "tag_score"),
        ("overall_mAP", "overall_map"),
        ("overall_AUC", "overall_roc_auc"),
    ),
    prediction_payload=tag_prediction_payload,
)


def make_mtg_subset_task(group: str) -> TaskSpec:
    names = {"genre": "genre87", "instrument": "instrument40", "mood": "mood56"}
    if group not in names:
        raise ValueError(f"未知 MTG 标签组：{group}")
    return TaskSpec(
        name=f"music_mtg_{names[group]}",
        dataset_name="mtg",
        output_key="tag_logits",
        score_key=f"{group}_map",
        metric_fn=mtg_metrics,
        default_print_every=100,
        metric_fields=(
            (f"{group}_mAP", f"{group}_map"),
            (f"{group}_AUC", f"{group}_roc_auc"),
        ),
        prediction_payload=tag_prediction_payload,
    )


def va_prediction_payload(
    predictions: np.ndarray, targets: np.ndarray, _: dict, metadata: dict
) -> dict:
    return {
        "predictions": predictions,
        "targets": targets,
        "predictions_1_9": predictions * 4 + 5,
        "targets_1_9": targets * 4 + 5,
        **metadata,
    }


def make_va_task(
    dataset_name: str, metric_fn: Callable = compute_va_metrics
) -> TaskSpec:
    def metrics(predictions: np.ndarray, targets: np.ndarray, _: dict) -> dict:
        return metric_fn(predictions, targets)

    return TaskSpec(
        name=f"music_va_{dataset_name}",
        dataset_name=dataset_name,
        output_key="va_predictions",
        score_key="va_score",
        metric_fn=metrics,
        default_print_every=50,
        metric_fields=(
            ("CCC_v", "valence_ccc"),
            ("CCC_a", "arousal_ccc"),
            ("CCC", "va_score"),
            ("PCC", "mean_pearson"),
            ("R2", "mean_r2"),
            ("RMSE", "mean_rmse"),
            ("MAE", "mean_mae"),
        ),
        prediction_payload=va_prediction_payload,
    )


def make_domain_balanced_va_metrics(
    lengths: tuple[int, ...], names: tuple[str, ...]
) -> Callable:
    if len(lengths) != len(names):
        raise ValueError("lengths 与 names 数量不一致")

    def metric_fn(predictions: np.ndarray, targets: np.ndarray) -> dict:
        metrics, offset, domain_scores = {}, 0, []
        for name, length in zip(names, lengths):
            domain = compute_va_metrics(
                predictions[offset : offset + length], targets[offset : offset + length]
            )
            domain_scores.append(domain["va_score"])
            metrics.update({f"{name}_{key}": value for key, value in domain.items()})
            offset += length
        overall = compute_va_metrics(predictions, targets)
        metrics.update(overall)
        metrics["sample_weighted_va_score"] = overall["va_score"]
        metrics["va_score"] = float(np.mean(domain_scores))
        return metrics

    return metric_fn


def forward_text_batch(
    model, batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = batch["targets"].to(device, non_blocking=device.type == "cuda")
    return model(batch["texts"], device=device)["va_predictions"], targets


def text_va_metrics(
    predictions: np.ndarray, targets: np.ndarray, metadata: dict[str, np.ndarray]
) -> dict:
    metrics = compute_va_metrics(predictions, targets)
    metrics["sample_weighted_text_va_score"] = metrics["va_score"]
    language_scores = []
    for group_key, prefix in (("languages", "language"), ("sources", "source")):
        groups = metadata[group_key]
        for group in sorted(set(groups.tolist())):
            mask = groups == group
            if mask.sum() < 2:
                continue
            group_metrics = compute_va_metrics(predictions[mask], targets[mask])
            safe_group = (
                str(group).lower().replace("+", "_").replace("/", "_").replace(" ", "_")
            )
            metrics.update(
                {
                    f"{prefix}_{safe_group}_{key}": value
                    for key, value in group_metrics.items()
                }
            )
            if group_key == "languages":
                language_scores.append(group_metrics["va_score"])
    metrics["text_va_score"] = (
        float(np.mean(language_scores)) if language_scores else metrics["va_score"]
    )
    return metrics


def text_prediction_payload(
    predictions: np.ndarray, targets: np.ndarray, _: dict, metadata: dict
) -> dict:
    return {
        "predictions": predictions,
        "targets": targets,
        "predictions_0_1": np.clip((predictions + 1.0) / 2.0, 0.0, 1.0),
        "targets_0_1": (targets + 1.0) / 2.0,
        **metadata,
    }


TEXT_VA_TASK = TaskSpec(
    name="text_va",
    dataset_name="text_va",
    output_key="va_predictions",
    score_key="text_va_score",
    metric_fn=text_va_metrics,
    metric_fields=(
        ("CCC_v", "valence_ccc"),
        ("CCC_a", "arousal_ccc"),
        ("CCC", "va_score"),
        ("Lang_CCC", "text_va_score"),
        ("PCC", "mean_pearson"),
        ("RMSE", "mean_rmse"),
        ("MAE", "mean_mae"),
    ),
    prediction_payload=text_prediction_payload,
    default_print_every=50,
    forward_batch=forward_text_batch,
    metadata_keys=("ids", "languages", "sources"),
)
