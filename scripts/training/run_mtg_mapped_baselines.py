"""Controlled MTG baselines on the same mapped labels, tracks, and split."""

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
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.retrieval.model import RetrieverModel
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.selection import tag_balanced_selection_score
from emomm_sequence.training.utilities import set_seed


RUN_ID = "mtg_mapped_baselines_20260914"
REPORT = ROOT / "results" / RUN_ID
RUNS = ROOT / "artifacts/runs" / RUN_ID
MAPPED = ROOT / "artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz"
QWEN = ROOT / "results/diagnosis_20260914_tower_freeze/query_form_embeddings.npz"
SEEDS = (133, 134, 135)
PROBE_EPOCHS = 12
TRIPLET_EPOCHS = 8
METRICS = (
    "ndcg_at_10",
    "average_precision",
    "macro_roc_auc",
    "precision_at_10",
    "recall_at_10",
)


class LinearProbe(nn.Module):
    def __init__(self, embedding_dim: int, labels: int) -> None:
        super().__init__()
        self.output = nn.Linear(embedding_dim, labels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(value)


class MLPProbe(nn.Module):
    def __init__(self, embedding_dim: int, labels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, labels),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


def load_inputs(config: dict) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    with np.load(MAPPED, allow_pickle=False) as archive:
        mtg = {key: archive[key] for key in archive.files}
    with np.load(ROOT / config["supervision"]["music_index"], allow_pickle=False) as archive:
        music = archive["embeddings"].astype(np.float32)
    with np.load(QWEN, allow_pickle=False) as archive:
        qwen = {key: archive[key] for key in archive.files}
    if not np.array_equal(mtg["preference_labels"].astype(str), qwen["labels"].astype(str)):
        raise ValueError("Qwen and MTG label order differs")
    queries = {
        "canonical": mtg["preference_label_embeddings"].astype(np.float32),
        "qwen_runtime": qwen["qwen_preference"].astype(np.float32),
    }
    return mtg, music, queries


def masks(mtg: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    splits = mtg["splits"].astype(str)
    return splits == "train", splits == "validation"


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[str, list[int]]:
    train, _ = masks(mtg)
    frequency = mtg["preference_targets"][train].sum(axis=0)
    labels = mtg["preference_labels"].astype(str)
    order = sorted(range(len(labels)), key=lambda index: (frequency[index], labels[index]))
    return {
        name: values.tolist()
        for name, values in zip(("tail", "mid", "head"), np.array_split(order, 3))
    }


def evaluate_scores(
    scores: np.ndarray, positives: np.ndarray, groups: dict[str, list[int]]
) -> dict:
    retrieval = label_to_track_metrics(scores, positives, (5, 10, 20))
    per_label = {int(row["query_index"]): row for row in retrieval["per_query"]}
    auc_rows = {
        index: float(roc_auc_score(positives[index], scores[index]))
        for index in range(len(positives))
        if np.unique(positives[index]).size == 2
    }
    overall = {
        "ndcg_at_10": float(retrieval["macro"]["ndcg_at_10"]),
        "average_precision": float(retrieval["macro"]["average_precision"]),
        "macro_roc_auc": float(np.mean(list(auc_rows.values()))),
        "precision_at_10": float(retrieval["macro"]["precision_at_10"]),
        "recall_at_10": float(retrieval["macro"]["recall_at_10"]),
    }
    grouped = {}
    for name, indices in groups.items():
        grouped[name] = {
            "ndcg_at_10": float(np.mean([per_label[index]["ndcg_at_10"] for index in indices])),
            "average_precision": float(
                np.mean([per_label[index]["average_precision"] for index in indices])
            ),
            "macro_roc_auc": float(np.mean([auc_rows[index] for index in indices])),
            "precision_at_10": float(
                np.mean([per_label[index]["precision_at_10"] for index in indices])
            ),
            "recall_at_10": float(
                np.mean([per_label[index]["recall_at_10"] for index in indices])
            ),
        }
    return {"overall": overall, "frequency_groups": grouped}


def frozen_and_centroid(
    mtg: dict[str, np.ndarray], music: np.ndarray, queries: dict[str, np.ndarray]
) -> list[dict]:
    train, validation = masks(mtg)
    train_music = music[mtg["index_positions"][train]]
    validation_music = music[mtg["index_positions"][validation]]
    train_targets = mtg["preference_targets"][train].astype(bool)
    positives = mtg["preference_targets"][validation].astype(bool).T
    groups = frequency_groups(mtg)
    rows: list[dict] = []
    for form, query in queries.items():
        rows.append(
            {
                "method": "frozen_clamp3",
                "query_form": form,
                "seed": None,
                "validation": evaluate_scores(query @ validation_music.T, positives, groups),
                "query_interface": "open_text",
            }
        )
    centroids = []
    for index in range(train_targets.shape[1]):
        centroids.append(train_music[train_targets[:, index]].mean(axis=0))
    centroid = np.asarray(centroids, dtype=np.float32)
    centroid /= np.linalg.norm(centroid, axis=1, keepdims=True).clip(min=1.0e-8)
    rows.append(
        {
            "method": "positive_centroid",
            "query_form": "closed_vocabulary_label_id",
            "seed": None,
            "validation": evaluate_scores(centroid @ validation_music.T, positives, groups),
            "query_interface": "closed_vocabulary",
        }
    )
    return rows


def train_loader(
    mtg: dict[str, np.ndarray], music: np.ndarray, batch_size: int, seed: int
) -> DataLoader:
    train, _ = masks(mtg)
    return DataLoader(
        TensorDataset(
            torch.from_numpy(music[mtg["index_positions"][train]].astype(np.float32)),
            torch.from_numpy(mtg["preference_targets"][train].astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


@torch.inference_mode()
def evaluate_probe(
    model: nn.Module,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    device: torch.device,
    groups: dict[str, list[int]],
) -> dict:
    _, validation = masks(mtg)
    features = music[mtg["index_positions"][validation]].astype(np.float32)
    parts = []
    model.eval()
    for start in range(0, len(features), 1024):
        batch = torch.from_numpy(features[start : start + 1024]).to(device)
        parts.append(model(batch).float().cpu().numpy())
    scores = np.concatenate(parts).T
    positives = mtg["preference_targets"][validation].astype(bool).T
    return evaluate_scores(scores, positives, groups)


def train_probe(
    kind: str,
    seed: int,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    embedding_dim: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = mtg["preference_targets"].shape[1]
    model: nn.Module = (
        LinearProbe(embedding_dim, labels)
        if kind == "linear_bce_probe"
        else MLPProbe(embedding_dim, labels)
    )
    model = model.to(device)
    loader = train_loader(mtg, music, 512, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PROBE_EPOCHS)
    groups = frequency_groups(mtg)
    best_score, best_epoch = -np.inf, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, PROBE_EPOCHS + 1):
        model.train()
        total = 0.0
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        evaluation = evaluate_probe(model, mtg, music, device, groups)
        score = tag_balanced_selection_score(
            evaluation["overall"]["ndcg_at_10"],
            evaluation["overall"]["average_precision"],
        )
        history.append(
            {"epoch": epoch, "loss": total / len(loader), "selection_score": score}
        )
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"{kind} seed={seed} epoch={epoch} loss={total / len(loader):.4f} "
            f"NDCG={evaluation['overall']['ndcg_at_10']:.4f} "
            f"AP={evaluation['overall']['average_precision']:.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    final = evaluate_probe(model, mtg, music, device, groups)
    run_dir = RUNS / kind / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in best_state.items()},
            "model_type": kind,
            "embedding_dim": embedding_dim,
            "labels": labels,
            "seed": seed,
            "epoch": best_epoch,
        },
        checkpoint,
    )
    row = {
        "method": kind,
        "query_form": "closed_vocabulary_label_id",
        "seed": seed,
        "validation": final,
        "query_interface": "closed_vocabulary",
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256(checkpoint),
    }
    write_json(run_dir / "metrics.json", row)
    return row


def build_triplet_model(config: dict, device: torch.device) -> RetrieverModel:
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
    ).to(device)
    for factor in ("semantic", "affect"):
        model.text_adapters[factor].requires_grad_(False)
        model.music_adapters[factor].requires_grad_(False)
    model.text_va_head.requires_grad_(False)
    model.music_va_head.requires_grad_(False)
    model.factor_logits.requires_grad_(False)
    return model


def batch_hard_triplet(scores: torch.Tensor, positive: torch.Tensor, margin: float) -> torch.Tensor:
    valid = positive.any(dim=1) & (~positive).any(dim=1)
    if not valid.any():
        return scores.sum() * 0.0
    selected_scores = scores[valid]
    selected_positive = positive[valid]
    hardest_negative = selected_scores.masked_fill(
        selected_positive, -torch.inf
    ).max(dim=1).values
    violations = F.softplus(
        margin + hardest_negative[:, None] - selected_scores
    )
    return violations[selected_positive].mean()


@torch.inference_mode()
def evaluate_dual(
    model: RetrieverModel,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    queries: dict[str, np.ndarray],
    device: torch.device,
    groups: dict[str, list[int]],
) -> dict[str, dict]:
    _, validation = masks(mtg)
    features = music[mtg["index_positions"][validation]].astype(np.float32)
    parts = []
    model.eval()
    for start in range(0, len(features), 1024):
        batch = torch.from_numpy(features[start : start + 1024]).to(device)
        parts.append(
            model.encode_factor("music", "preference", batch).float().cpu().numpy()
        )
    audio = np.concatenate(parts)
    positives = mtg["preference_targets"][validation].astype(bool).T
    result = {}
    for name, raw in queries.items():
        text = model.encode_factor(
            "text", "preference", torch.from_numpy(raw).to(device)
        ).float().cpu().numpy()
        result[name] = evaluate_scores(text @ audio.T, positives, groups)
    return result


def train_triplet(
    seed: int,
    config: dict,
    mtg: dict[str, np.ndarray],
    music: np.ndarray,
    queries: dict[str, np.ndarray],
) -> list[dict]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_triplet_model(config, device)
    canonical = torch.from_numpy(queries["canonical"]).to(device)
    loader = train_loader(mtg, music, 512, seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=3.0e-4, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRIPLET_EPOCHS)
    groups = frequency_groups(mtg)
    initial = evaluate_dual(model, mtg, music, queries, device, groups)
    best_score = tag_balanced_selection_score(
        initial["qwen_runtime"]["overall"]["ndcg_at_10"],
        initial["qwen_runtime"]["overall"]["average_precision"],
    )
    best_epoch, best_state = 0, copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(1, TRIPLET_EPOCHS + 1):
        model.train()
        total = 0.0
        for features, target in loader:
            features, target = features.to(device), target.bool().to(device)
            optimizer.zero_grad(set_to_none=True)
            text = model.encode_factor("text", "preference", canonical)
            audio = model.encode_factor("music", "preference", features)
            loss = batch_hard_triplet(text @ audio.T, target.T, margin=0.2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total += float(loss.item())
        scheduler.step()
        evaluation = evaluate_dual(model, mtg, music, queries, device, groups)
        score = tag_balanced_selection_score(
            evaluation["qwen_runtime"]["overall"]["ndcg_at_10"],
            evaluation["qwen_runtime"]["overall"]["average_precision"],
        )
        history.append({"epoch": epoch, "loss": total / len(loader), "selection_score": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"triplet_metric seed={seed} epoch={epoch} loss={total / len(loader):.4f} "
            f"Qwen_NDCG={evaluation['qwen_runtime']['overall']['ndcg_at_10']:.4f}",
            flush=True,
        )
    model.load_state_dict(best_state)
    final = evaluate_dual(model, mtg, music, queries, device, groups)
    run_dir = RUNS / "triplet_metric" / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "best.pt"
    torch.save(
        {
            "model": {key: value.detach().cpu() for key, value in best_state.items()},
            "model_config": config["model"],
            "embedding_dim": int(config["backbone"]["embedding_dim"]),
            "seed": seed,
            "epoch": best_epoch,
        },
        checkpoint,
    )
    rows = []
    for form, evaluation in final.items():
        rows.append(
            {
                "method": "triplet_metric",
                "query_form": form,
                "seed": seed,
                "validation": evaluation,
                "query_interface": "open_text",
                "best_epoch": best_epoch,
                "history": history,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": sha256(checkpoint),
            }
        )
    write_json(run_dir / "metrics.json", rows)
    return rows


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
        "methods": {
            "frozen_clamp3": "zero-shot cosine",
            "positive_centroid": "training-positive mean prototype",
            "linear_bce_probe": "frozen-feature linear multi-label classifier",
            "mlp_bce_probe": "frozen-feature MLP multi-label classifier",
            "triplet_metric": "two-tower batch-hard triplet metric learning",
        },
        "fairness": (
            "same mapped label identities and targets, split, validation candidate tracks, "
            "CLaMP3 audio features, and metric implementation"
        ),
        "open_vs_closed_vocabulary_reported_separately": True,
        "config_sha256": sha256(config_path),
        "script_sha256": sha256(Path(__file__)),
        "mapped_sha256": sha256(MAPPED),
        "qwen_sha256": sha256(QWEN),
    }
    write_json(REPORT / "plan.json", plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


def run() -> None:
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    config_path = ROOT / "configs/training/retriever.yaml"
    for path, expected in (
        (config_path, plan["config_sha256"]),
        (Path(__file__), plan["script_sha256"]),
        (MAPPED, plan["mapped_sha256"]),
        (QWEN, plan["qwen_sha256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError(f"protocol input changed: {path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mtg, music, queries = load_inputs(config)
    rows = frozen_and_centroid(mtg, music, queries)
    for kind in ("linear_bce_probe", "mlp_bce_probe"):
        for seed in SEEDS:
            rows.append(
                train_probe(
                    kind,
                    seed,
                    mtg,
                    music,
                    int(config["backbone"]["embedding_dim"]),
                )
            )
    for seed in SEEDS:
        rows.extend(train_triplet(seed, config, mtg, music, queries))
    write_json(REPORT / "execution.json", rows)


def summarize() -> None:
    rows = json.loads((REPORT / "execution.json").read_text(encoding="utf-8"))
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["method"], row["query_form"]), []).append(row)
    summary = {
        "run_id": RUN_ID,
        "partition": "validation",
        "test_evaluated": False,
        "comparability_note": (
            "Closed-vocabulary probes cannot answer arbitrary text; compare their ranking quality "
            "but do not describe them as equivalent production interfaces."
        ),
        "methods": {},
    }
    for (method, form), selected in groups.items():
        key = f"{method}/{form}"
        summary["methods"][key] = {
            "runs": len(selected),
            "query_interface": selected[0]["query_interface"],
            "overall": {
                metric: {
                    "mean": float(
                        np.mean([row["validation"]["overall"][metric] for row in selected])
                    ),
                    "std": float(
                        np.std(
                            [row["validation"]["overall"][metric] for row in selected],
                            ddof=1,
                        )
                    )
                    if len(selected) > 1
                    else 0.0,
                }
                for metric in METRICS
            },
            "frequency_groups": {
                group: {
                    metric: {
                        "mean": float(
                            np.mean(
                                [
                                    row["validation"]["frequency_groups"][group][metric]
                                    for row in selected
                                ]
                            )
                        ),
                        "std": float(
                            np.std(
                                [
                                    row["validation"]["frequency_groups"][group][metric]
                                    for row in selected
                                ],
                                ddof=1,
                            )
                        )
                        if len(selected) > 1
                        else 0.0,
                    }
                    for metric in METRICS
                }
                for group in ("tail", "mid", "head")
            },
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
