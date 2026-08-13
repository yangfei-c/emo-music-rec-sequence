import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.metrics import compute_mtg_metrics, compute_va_metrics
from validation.helpers import require


def require_close(
    value: float, expected: float, message: str, tolerance: float = 1.0e-10
) -> None:
    require(
        abs(value - expected) <= tolerance, f"{message}：实际 {value}，预期 {expected}"
    )


def require_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise RuntimeError(f"[FAIL] {message}")


def check_perfect_mtg() -> tuple[np.ndarray, np.ndarray]:
    targets = np.zeros((6, 183), dtype=np.float64)
    for label in range(183):
        targets[label % 6, label] = 1.0
        targets[(label + 2) % 6, label] = 1.0

    logits = np.where(targets == 1.0, 10.0, -10.0)
    metrics = compute_mtg_metrics(logits, targets)

    for name, size in (("genre", 87), ("instrument", 40), ("mood", 56)):
        require_close(metrics[f"{name}_map"], 1.0, f"{name} mAP 错误")
        require_close(metrics[f"{name}_roc_auc"], 1.0, f"{name} ROC-AUC 错误")
        require(metrics[f"{name}_valid_ap_labels"] == size, f"{name} AP 标签数错误")
        require(metrics[f"{name}_valid_auc_labels"] == size, f"{name} AUC 标签数错误")

    require_close(metrics["tag_score"], 1.0, "tag_score 错误")
    require_close(metrics["tag_roc_auc"], 1.0, "tag_roc_auc 错误")
    print("[PASS] MTG 完美预测：三个标签组 mAP/ROC-AUC 均为 1")
    return logits, targets


def check_mtg_boundaries(logits: np.ndarray, targets: np.ndarray) -> None:
    no_positive_targets = targets.copy()
    no_positive_targets[:, 0] = 0.0
    no_positive = compute_mtg_metrics(logits, no_positive_targets)
    require(np.isnan(no_positive["per_label_ap"][0]), "无正样本标签的 AP 应被跳过")
    require(no_positive["valid_ap_labels"] == 182, "无正样本标签未正确排除")

    all_positive_targets, all_positive_logits = targets.copy(), logits.copy()
    all_positive_targets[:, 1], all_positive_logits[:, 1] = 1.0, 10.0
    all_positive = compute_mtg_metrics(all_positive_logits, all_positive_targets)
    require_close(all_positive["per_label_ap"][1], 1.0, "全正标签 AP 错误")
    require(
        np.isnan(all_positive["per_label_roc_auc"][1]),
        "单一类别标签的 ROC-AUC 应被跳过",
    )
    require(all_positive["valid_auc_labels"] == 182, "单一类别标签未从 ROC-AUC 排除")

    require_error(
        lambda: compute_mtg_metrics(logits[:, :-1], targets), "MTG 形状错误未报错"
    )
    invalid_targets = targets.copy()
    invalid_targets[0, 0] = 0.5
    require_error(
        lambda: compute_mtg_metrics(logits, invalid_targets), "非二值 MTG target 未报错"
    )
    invalid_logits = logits.copy()
    invalid_logits[0, 0] = np.nan
    require_error(
        lambda: compute_mtg_metrics(invalid_logits, targets), "MTG NaN 未报错"
    )
    print("[PASS] MTG 无正样本、单一类别、形状、二值和有限值检查")


def check_va() -> None:
    targets = np.array(
        [
            [-1.0, -0.8],
            [-0.3, 0.4],
            [0.2, -0.1],
            [0.9, 1.0],
        ],
        dtype=np.float64,
    )

    perfect = compute_va_metrics(targets.copy(), targets)
    for key in (
        "valence_ccc",
        "arousal_ccc",
        "va_score",
        "valence_pearson",
        "arousal_pearson",
        "mean_pearson",
    ):
        require_close(perfect[key], 1.0, f"{key} 错误")
    for key in (
        "valence_rmse",
        "arousal_rmse",
        "mean_rmse",
        "valence_mae",
        "arousal_mae",
        "mean_mae",
    ):
        require_close(perfect[key], 0.0, f"{key} 错误")

    constant = compute_va_metrics(np.zeros_like(targets), targets)
    require(
        all(np.isfinite(value) for value in constant.values()),
        "常数 VA 预测产生 NaN 或 Inf",
    )
    require(
        constant["mean_rmse"] > 0 and constant["mean_mae"] > 0, "常数 VA 预测误差错误"
    )

    require_error(
        lambda: compute_va_metrics(targets[:, :1], targets), "VA 形状错误未报错"
    )
    invalid_predictions = targets.copy()
    invalid_predictions[0, 0] = np.inf
    require_error(
        lambda: compute_va_metrics(invalid_predictions, targets), "VA Inf 未报错"
    )
    require_error(
        lambda: compute_va_metrics(targets[:1], targets[:1]), "VA 单样本未报错"
    )
    print("[PASS] VA 完美预测、常数预测、形状和有限值检查")


def main() -> None:
    logits, targets = check_perfect_mtg()
    check_mtg_boundaries(logits, targets)
    check_va()
    print("[PASS] MTG 和 VA 指标检查全部通过")


if __name__ == "__main__":
    main()
