"""Metrics and checkpoint helpers for relation experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score

from emomm_sequence.training.calibration import calibration_bins


def relation_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    modes: np.ndarray,
) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape[:2] != labels.shape or probabilities.ndim != 3:
        raise ValueError("relation probabilities/labels shape mismatch")
    predictions = probabilities.argmax(axis=-1)

    def one(mask: np.ndarray) -> dict:
        if not mask.any():
            return {}
        p = probabilities[mask]
        y = labels[mask]
        pred = predictions[mask]
        nll_axis = []
        brier_axis = []
        accuracy_axis = []
        macro_f1_axis = []
        for axis in range(2):
            nll_axis.append(float(-np.log(np.maximum(p[np.arange(len(p)), axis, y[:, axis]], 1.0e-12)).mean()))
            onehot = np.eye(p.shape[-1], dtype=np.float64)[y[:, axis]]
            brier_axis.append(float(np.square(p[:, axis] - onehot).sum(axis=1).mean()))
            accuracy_axis.append(float((pred[:, axis] == y[:, axis]).mean()))
            macro_f1_axis.append(float(f1_score(y[:, axis], pred[:, axis], labels=list(range(p.shape[-1])), average="macro", zero_division=0)))
        return {
            "count": int(mask.sum()),
            "accuracy_v": accuracy_axis[0],
            "accuracy_a": accuracy_axis[1],
            "macro_f1_v": macro_f1_axis[0],
            "macro_f1_a": macro_f1_axis[1],
            "nll_v": nll_axis[0],
            "nll_a": nll_axis[1],
            "brier_v": brier_axis[0],
            "brier_a": brier_axis[1],
            "source_axis_equal_nll": float(np.mean(nll_axis)),
            "calibration_bins": calibration_bins(p, y),
        }

    result = {"overall": one(np.ones(len(labels), dtype=bool)), "by_source": {}, "by_mode": {}}
    for source in sorted(set(sources.astype(str))):
        result["by_source"][source] = one(sources == source)
    for mode in sorted(set(modes.astype(str))):
        result["by_mode"][mode] = one(modes == mode)
    source_axis_nll = []
    for value in result["by_source"].values():
        source_axis_nll.extend([value["nll_v"], value["nll_a"]])
    result["source_axis_equal_nll"] = float(np.mean(source_axis_nll))
    return result


def aggregate_seed_metrics(seed_reports: list[dict], path: tuple[str, ...]) -> dict:
    leafs = []
    for report in seed_reports:
        value = report
        for key in path:
            value = value[key]
        leafs.append(value)
    keys = [key for key, value in leafs[0].items() if isinstance(value, (int, float)) and key != "count"]
    return {
        key: {
            "mean": float(np.mean([leaf[key] for leaf in leafs])),
            "sd": float(np.std([leaf[key] for leaf in leafs], ddof=1)) if len(leafs) > 1 else 0.0,
        }
        for key in keys
    }
