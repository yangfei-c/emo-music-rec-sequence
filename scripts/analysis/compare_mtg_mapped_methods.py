"""Assemble a same-protocol MTG mapped-label comparison including Ours."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json


BASELINE = ROOT / "artifacts/runs/mtg_mapped_baselines_20260914/evaluation/summary.json"
ALIGNMENT = ROOT / "artifacts/runs/diagnosis_20260914_query_alignment/evaluation/execution.json"
MAPPED = ROOT / "artifacts/features/clamp3/clean/mtg_factor_supervision_mapped.npz"
QWEN = ROOT / "artifacts/runs/diagnosis_20260914_tower_freeze/evaluation/query_form_embeddings.npz"
OUTPUT = ROOT / "artifacts/runs/mtg_mapped_comparison_20260914/evaluation"
VARIANTS = {
    "legacy_only": "ours_original_prompt",
    "mapped_three_view": "ours_mapped_three_view",
    "mapped_plus_qwen": "ours_mapped_plus_qwen",
}
METRICS = (
    "ndcg_at_10",
    "average_precision",
    "macro_roc_auc",
    "precision_at_10",
    "recall_at_10",
)


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[str, list[int]]:
    train = mtg["splits"].astype(str) == "train"
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
    rows = {int(row["query_index"]): row for row in retrieval["per_query"]}
    auc = {
        index: float(roc_auc_score(positives[index], scores[index]))
        for index in range(len(positives))
    }

    def summarize(indices: list[int]) -> dict[str, float]:
        return {
            "ndcg_at_10": float(np.mean([rows[index]["ndcg_at_10"] for index in indices])),
            "average_precision": float(
                np.mean([rows[index]["average_precision"] for index in indices])
            ),
            "macro_roc_auc": float(np.mean([auc[index] for index in indices])),
            "precision_at_10": float(
                np.mean([rows[index]["precision_at_10"] for index in indices])
            ),
            "recall_at_10": float(
                np.mean([rows[index]["recall_at_10"] for index in indices])
            ),
        }

    all_indices = list(range(len(positives)))
    return {
        "overall": summarize(all_indices),
        "frequency_groups": {
            name: summarize(indices) for name, indices in groups.items()
        },
    }


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    query: np.ndarray,
    raw_music: np.ndarray,
    positives: np.ndarray,
    groups: dict[str, list[int]],
    device: torch.device,
) -> dict:
    model = FrozenRetriever(checkpoint).to(device).eval()
    text = model.encode_factor(
        "text", "preference", torch.from_numpy(query).to(device)
    ).float().cpu().numpy()
    music = []
    for start in range(0, len(raw_music), 1024):
        batch = torch.from_numpy(raw_music[start : start + 1024]).to(device)
        music.append(
            model.encode_factor("music", "preference", batch).float().cpu().numpy()
        )
    return evaluate_scores(text @ np.concatenate(music).T, positives, groups)


def aggregate(rows: list[dict]) -> dict:
    return {
        "runs": len(rows),
        "query_interface": "open_text",
        "overall": {
            metric: {
                "mean": float(np.mean([row["overall"][metric] for row in rows])),
                "std": float(np.std([row["overall"][metric] for row in rows], ddof=1)),
            }
            for metric in METRICS
        },
        "frequency_groups": {
            group: {
                metric: {
                    "mean": float(
                        np.mean([row["frequency_groups"][group][metric] for row in rows])
                    ),
                    "std": float(
                        np.std(
                            [row["frequency_groups"][group][metric] for row in rows],
                            ddof=1,
                        )
                    ),
                }
                for metric in METRICS
            }
            for group in ("tail", "mid", "head")
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = yaml.safe_load(
        (ROOT / "configs/training/retriever.yaml").read_text(encoding="utf-8")
    )
    with np.load(MAPPED, allow_pickle=False) as archive:
        mtg = {key: archive[key] for key in archive.files}
    with np.load(ROOT / config["supervision"]["music_index"], allow_pickle=False) as archive:
        catalog = archive["embeddings"].astype(np.float32)
    with np.load(QWEN, allow_pickle=False) as archive:
        qwen = {key: archive[key] for key in archive.files}
    if not np.array_equal(mtg["preference_labels"].astype(str), qwen["labels"].astype(str)):
        raise ValueError("query labels do not match")
    validation = mtg["splits"].astype(str) == "validation"
    raw_music = catalog[mtg["index_positions"][validation]].astype(np.float32)
    positives = mtg["preference_targets"][validation].astype(bool).T
    query_forms = {
        "canonical": mtg["preference_label_embeddings"].astype(np.float32),
        "qwen_runtime": qwen["qwen_preference"].astype(np.float32),
    }
    groups = frequency_groups(mtg)
    methods = json.loads(BASELINE.read_text(encoding="utf-8"))["methods"]
    alignment = json.loads(ALIGNMENT.read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for variant, method in VARIANTS.items():
        selected = [row for row in alignment if row["variant"] == variant]
        for form, query in query_forms.items():
            evaluations = [
                evaluate_checkpoint(
                    ROOT / row["checkpoint"], query, raw_music, positives, groups, device
                )
                for row in selected
            ]
            methods[f"{method}/{form}"] = aggregate(evaluations)
    frozen = methods["frozen_clamp3/qwen_runtime"]["overall"]
    for key, method in methods.items():
        method["delta_vs_frozen_qwen"] = {
            metric: method["overall"][metric]["mean"] - frozen[metric]["mean"]
            for metric in METRICS
        }
    output = {
        "partition": "validation",
        "test_evaluated": False,
        "protocol": {
            "labels": int(len(mtg["preference_labels"])),
            "validation_tracks": int(validation.sum()),
            "target_matrix_sha256": sha256(MAPPED),
            "same_candidate_tracks_and_metrics": True,
            "selection_without_test": True,
        },
        "interpretation_boundary": (
            "Closed-vocabulary probes receive a label index, whereas open-text methods receive "
            "a CLaMP3 text embedding. Paper-reported numbers on other vocabularies or splits are "
            "context only and are not mixed into this ranking."
        ),
        "literature_basis": {
            "mtg_official": {
                "paper": "https://repositori.upf.edu/bitstream/handle/10230/42015/bogdanov_ICML2019__Jamendo.pdf",
                "code": "https://github.com/MTG/mtg-jamendo-dataset",
                "role": "official splits, tag cleaning, VGG-ish classification reference",
            },
            "cnn_evaluation": {
                "paper": "https://arxiv.org/abs/2006.00751",
                "code": "https://github.com/minzwon/sota-music-tagging-models",
                "role": "classification architecture reference",
            },
            "mert": {
                "paper": "https://proceedings.iclr.cc/paper_files/paper/2024/file/33dffa2e3d2ab74a783d1a8c292f66d9-Paper-Conference.pdf",
                "code": "https://github.com/yizhilll/MERT",
                "role": "frozen-backbone MLP probing protocol",
            },
            "multimodal_metric_learning": {
                "paper": "https://arxiv.org/abs/2010.16030",
                "code": "https://github.com/minzwon/tag-based-music-retrieval",
                "role": "triplet dual-encoder and balanced sampling reference",
            },
            "musical_word_embedding": {
                "paper": "https://arxiv.org/abs/2404.13569",
                "weights": "https://huggingface.co/seungheondoh/musical-word-embedding",
                "role": "music-specific text prototypes and multi-prototype motivation",
            },
            "ttmr_plus_plus": {
                "paper": "https://arxiv.org/abs/2410.03264",
                "code": "https://github.com/seungheondoh/music-text-representation-pp",
                "role": "LLM-enriched text-to-music retrieval motivation",
            },
        },
        "methods": methods,
    }
    write_json(OUTPUT / "comparison.json", output)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["method", "interface", *[f"{metric}_mean" for metric in METRICS], *[f"{metric}_std" for metric in METRICS]]
        )
        for key, method in sorted(
            methods.items(), key=lambda item: item[1]["overall"]["ndcg_at_10"]["mean"], reverse=True
        ):
            writer.writerow(
                [
                    key,
                    method["query_interface"],
                    *[method["overall"][metric]["mean"] for metric in METRICS],
                    *[method["overall"][metric]["std"] for metric in METRICS],
                ]
            )
    compact = [
        {
            "method": key,
            "interface": value["query_interface"],
            **{metric: value["overall"][metric]["mean"] for metric in METRICS},
        }
        for key, value in sorted(
            methods.items(), key=lambda item: item[1]["overall"]["ndcg_at_10"]["mean"], reverse=True
        )
    ]
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
