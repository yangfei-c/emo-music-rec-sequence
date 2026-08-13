import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


TAG_GROUPS = {
    "genre": slice(0, 87),
    "instrument": slice(87, 127),
    "mood": slice(127, 183),
}


def to_numpy(value) -> np.ndarray:
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


def check_arrays(
    predictions, targets, output_dim: int, name: str
) -> tuple[np.ndarray, np.ndarray]:
    predictions, targets = to_numpy(predictions).astype(np.float64), to_numpy(
        targets
    ).astype(np.float64)
    if (
        predictions.shape != targets.shape
        or predictions.ndim != 2
        or predictions.shape[1] != output_dim
    ):
        raise ValueError(
            f"{name} predictions/targets 应为 [N,{output_dim}]，实际为 "
            f"{predictions.shape}/{targets.shape}"
        )
    if predictions.shape[0] < 2:
        raise ValueError(f"{name} 至少需要 2 个样本")
    if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
        raise ValueError(f"{name} 输入包含 NaN/Inf")
    return predictions, targets


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def compute_mtg_metrics(logits, targets) -> dict:
    logits, targets = check_arrays(logits, targets, 183, "MTG")
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("MTG targets 必须是二值标签")
    probabilities = sigmoid(logits)
    per_label_ap, per_label_auc = np.full(183, np.nan), np.full(183, np.nan)
    for index in range(183):
        label_targets, label_scores = targets[:, index], probabilities[:, index]
        if label_targets.sum() > 0:
            per_label_ap[index] = average_precision_score(label_targets, label_scores)
        if np.unique(label_targets).size == 2:
            per_label_auc[index] = roc_auc_score(label_targets, label_scores)
    metrics = {"per_label_ap": per_label_ap, "per_label_roc_auc": per_label_auc}
    for name, group in TAG_GROUPS.items():
        ap_values, auc_values = per_label_ap[group], per_label_auc[group]
        metrics[f"{name}_map"] = (
            float(np.nanmean(ap_values))
            if np.isfinite(ap_values).any()
            else float("nan")
        )
        metrics[f"{name}_roc_auc"] = (
            float(np.nanmean(auc_values))
            if np.isfinite(auc_values).any()
            else float("nan")
        )
        metrics[f"{name}_valid_ap_labels"] = int(np.isfinite(ap_values).sum())
        metrics[f"{name}_valid_auc_labels"] = int(np.isfinite(auc_values).sum())
    group_maps = [metrics[f"{name}_map"] for name in TAG_GROUPS]
    group_aucs = [metrics[f"{name}_roc_auc"] for name in TAG_GROUPS]
    metrics.update(
        tag_score=(
            float(np.mean(group_maps))
            if np.isfinite(group_maps).all()
            else float("nan")
        ),
        tag_roc_auc=(
            float(np.mean(group_aucs))
            if np.isfinite(group_aucs).all()
            else float("nan")
        ),
        overall_map=(
            float(np.nanmean(per_label_ap))
            if np.isfinite(per_label_ap).any()
            else float("nan")
        ),
        overall_roc_auc=(
            float(np.nanmean(per_label_auc))
            if np.isfinite(per_label_auc).any()
            else float("nan")
        ),
        valid_ap_labels=int(np.isfinite(per_label_ap).sum()),
        valid_auc_labels=int(np.isfinite(per_label_auc).sum()),
    )
    return metrics


def concordance_correlation_coefficient(
    targets: np.ndarray, predictions: np.ndarray, eps: float = 1.0e-12
) -> float:
    target_mean, prediction_mean = targets.mean(), predictions.mean()
    target_var = np.mean((targets - target_mean) ** 2)
    prediction_var = np.mean((predictions - prediction_mean) ** 2)
    covariance = np.mean((targets - target_mean) * (predictions - prediction_mean))
    denominator = target_var + prediction_var + (target_mean - prediction_mean) ** 2
    if denominator <= eps:
        return 1.0 if np.allclose(targets, predictions, atol=eps, rtol=0.0) else 0.0
    return float(np.clip(2.0 * covariance / denominator, -1.0, 1.0))


def compute_va_metrics(predictions, targets) -> dict[str, float]:
    predictions, targets = check_arrays(predictions, targets, 2, "VA")
    ccc = [
        concordance_correlation_coefficient(targets[:, axis], predictions[:, axis])
        for axis in range(2)
    ]
    pearson = [_pearson(targets[:, axis], predictions[:, axis]) for axis in range(2)]
    rmse = np.sqrt(mean_squared_error(targets, predictions, multioutput="raw_values"))
    mae = mean_absolute_error(targets, predictions, multioutput="raw_values")
    r2 = r2_score(targets, predictions, multioutput="raw_values", force_finite=True)
    return {
        "valence_ccc": ccc[0],
        "arousal_ccc": ccc[1],
        "va_score": float(np.mean(ccc)),
        "valence_pearson": pearson[0],
        "arousal_pearson": pearson[1],
        "mean_pearson": float(np.mean(pearson)),
        "valence_r2": float(r2[0]),
        "arousal_r2": float(r2[1]),
        "mean_r2": float(np.mean(r2)),
        "valence_rmse": float(rmse[0]),
        "arousal_rmse": float(rmse[1]),
        "mean_rmse": float(rmse.mean()),
        "valence_mae": float(mae[0]),
        "arousal_mae": float(mae[1]),
        "mean_mae": float(mae.mean()),
    }


def _pearson(
    targets: np.ndarray, predictions: np.ndarray, eps: float = 1.0e-12
) -> float:
    target_std, prediction_std = targets.std(), predictions.std()
    if target_std <= eps or prediction_std <= eps:
        return 1.0 if np.allclose(targets, predictions, atol=eps, rtol=0.0) else 0.0
    return float(np.clip(np.corrcoef(targets, predictions)[0, 1], -1.0, 1.0))
