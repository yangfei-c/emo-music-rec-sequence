"""Diagnose MTG preference query shift: label prompt, sentence, Qwen field."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.encoding.text import FrozenTextEncoder
from emomm_sequence.evaluation.retrieval import label_to_track_metrics
from emomm_sequence.inference.query import NaturalLanguageQueryParser
from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json


DIAGNOSIS = ROOT / "artifacts/runs/diagnosis_20260914_tower_freeze/evaluation"
RUNS = ROOT / "artifacts/runs/diagnosis_20260914_tower_freeze"
REQUESTS = DIAGNOSIS / "query_forms.json"
EMBEDDINGS = DIAGNOSIS / "query_form_embeddings.npz"
EVALUATION = DIAGNOSIS / "query_form_evaluation.json"
SEEDS = (133, 134, 135)
MODEL_VARIANTS = ("both_frozen", "both_train")
FORMS = ("training_prompt", "natural_request", "qwen_preference")
METRICS = ("ndcg_at_10", "average_precision", "precision_at_10", "recall_at_10")


def load_config(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def load_mtg(config: dict) -> dict[str, np.ndarray]:
    with np.load(ROOT / config["supervision"]["mtg_factors"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def natural_request(prompt: str) -> str:
    return f"我现在心情平静，希望继续保持，想听{prompt}。"


def generate() -> None:
    if REQUESTS.exists():
        raise FileExistsError(REQUESTS)
    training_config = load_config("configs/training/retriever.yaml")
    system_config = load_config("configs/system.yaml")
    mtg = load_mtg(training_config)
    parser = NaturalLanguageQueryParser.from_config(system_config["query_parser"])
    digest = parser.model_digest()
    records = []
    for index, (label, prompt) in enumerate(
        zip(mtg["preference_labels"], mtg["preference_prompts"], strict=True)
    ):
        sentence = natural_request(str(prompt))
        try:
            parsed = parser.parse(sentence)
            record = {
                "index": index,
                "label": str(label),
                "training_prompt": str(prompt),
                "natural_request": sentence,
                "qwen_preference": parsed.music_preference,
                "qwen_current_affect": parsed.current_affect,
                "qwen_target_affect": parsed.target_affect,
                "evidence": parsed.evidence,
                "status": "success" if parsed.music_preference else "empty_preference",
            }
        except Exception as error:
            record = {
                "index": index,
                "label": str(label),
                "training_prompt": str(prompt),
                "natural_request": sentence,
                "qwen_preference": "",
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        records.append(record)
        if (index + 1) % 10 == 0 or index + 1 == len(mtg["preference_labels"]):
            successes = sum(row["status"] == "success" for row in records)
            print(f"parsed {index + 1}/{len(mtg['preference_labels'])}; success={successes}", flush=True)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "parser_model": system_config["query_parser"]["model"],
        "parser_digest": digest,
        "template": "我现在心情平静，希望继续保持，想听{training_prompt}。",
        "records": records,
        "successes": sum(row["status"] == "success" for row in records),
        "failures": sum(row["status"] != "success" for row in records),
        "test_evaluated": False,
    }
    write_json(REQUESTS, payload)


@torch.inference_mode()
def encode_texts(
    encoder: FrozenTextEncoder,
    tokenizer,
    texts: list[str],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        outputs.append(
            encoder.encode(
                tokens["input_ids"].to(device), tokens["attention_mask"].to(device)
            ).float().cpu().numpy()
        )
    return np.concatenate(outputs)


def encode() -> None:
    if EMBEDDINGS.exists():
        raise FileExistsError(EMBEDDINGS)
    payload = json.loads(REQUESTS.read_text(encoding="utf-8"))
    records = [row for row in payload["records"] if row["status"] == "success"]
    if not records:
        raise RuntimeError("qwen3:4b produced no usable preference fields")
    system_config = load_config("configs/system.yaml")
    backbone = system_config["backbone"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenTextEncoder(
        ROOT / backbone["checkpoint"], backbone["text_model"]
    ).to(device)
    tokenizer = encoder.tokenizer()
    arrays = {}
    for form in FORMS:
        texts = [row[form] for row in records]
        arrays[form] = encode_texts(
            encoder,
            tokenizer,
            texts,
            device,
            batch_size=32,
            max_length=int(backbone["max_text_length"]),
        )
        print(f"encoded {form}: {arrays[form].shape}", flush=True)
    np.savez_compressed(
        EMBEDDINGS,
        indices=np.asarray([row["index"] for row in records], dtype=np.int64),
        labels=np.asarray([row["label"] for row in records]),
        **arrays,
    )


def frequency_groups(mtg: dict[str, np.ndarray]) -> dict[int, str]:
    train = mtg["splits"].astype(str) == "train"
    frequency = mtg["preference_targets"][train].sum(axis=0)
    labels = mtg["preference_labels"].astype(str)
    ordered = sorted(range(len(labels)), key=lambda index: (frequency[index], labels[index]))
    return {
        int(index): name
        for name, values in zip(
            ("tail", "mid", "head"), np.array_split(np.asarray(ordered), 3)
        )
        for index in values
    }


def compact(metrics: dict, original_indices: np.ndarray, group_map: dict[int, str]) -> dict:
    rows = metrics["per_query"]
    groups = {}
    for name in ("tail", "mid", "head"):
        selected = [
            row
            for row in rows
            if group_map[int(original_indices[int(row["query_index"])])] == name
        ]
        groups[name] = {
            "labels": len(selected),
            **{key: float(np.mean([row[key] for row in selected])) for key in METRICS},
        }
    return {
        "labels": len(rows),
        "overall": {key: float(metrics["macro"][key]) for key in METRICS},
        "frequency_groups": groups,
    }


@torch.inference_mode()
def evaluate() -> None:
    if EVALUATION.exists():
        raise FileExistsError(EVALUATION)
    config = load_config("configs/training/retriever.yaml")
    mtg = load_mtg(config)
    with np.load(ROOT / config["supervision"]["music_index"], allow_pickle=False) as data:
        music_index = data["embeddings"].astype(np.float32)
    with np.load(EMBEDDINGS, allow_pickle=False) as data:
        query_data = {key: data[key] for key in data.files}
    indices = query_data["indices"].astype(int)
    mask = mtg["splits"].astype(str) == "validation"
    raw_music = music_index[mtg["index_positions"][mask]].astype(np.float32)
    positives = mtg["preference_targets"][mask][:, indices].astype(bool).T
    group_map = frequency_groups(mtg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(config["training"]["batch_size_mtg"])
    ks = tuple(int(value) for value in config["evaluation"]["tag_ks"])
    results = {}
    for variant in MODEL_VARIANTS:
        results[variant] = {}
        for seed in SEEDS:
            checkpoint = RUNS / variant / f"seed_{seed}/best.pt"
            model = FrozenRetriever(checkpoint).to(device).eval()
            music_parts = []
            for start in range(0, len(raw_music), batch_size):
                batch = torch.from_numpy(raw_music[start : start + batch_size]).to(device)
                music_parts.append(
                    model.encode_factor("music", "preference", batch).float().cpu().numpy()
                )
            music = np.concatenate(music_parts)
            forms = {}
            for form in FORMS:
                raw_query = torch.from_numpy(query_data[form].astype(np.float32)).to(device)
                query = model.encode_factor("text", "preference", raw_query).float().cpu().numpy()
                forms[form] = compact(
                    label_to_track_metrics(query @ music.T, positives, ks), indices, group_map
                )
            results[variant][str(seed)] = forms
            print(f"evaluated {variant} seed={seed}", flush=True)
    write_json(
        EVALUATION,
        {
            "partition": "validation",
            "test_evaluated": False,
            "forms": list(FORMS),
            "common_successful_labels": int(len(indices)),
            "request_sha256": sha256(REQUESTS),
            "embedding_sha256": sha256(EMBEDDINGS),
            "results": results,
        },
    )


def summarize() -> None:
    payload = json.loads(EVALUATION.read_text(encoding="utf-8"))
    summary = {
        "partition": "validation",
        "test_evaluated": False,
        "common_successful_labels": payload["common_successful_labels"],
        "variants": {},
    }
    for variant, seeds in payload["results"].items():
        form_summary = {}
        for form in FORMS:
            form_summary[form] = {}
            for key in METRICS:
                values = np.asarray([seeds[str(seed)][form]["overall"][key] for seed in SEEDS])
                form_summary[form][key] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
        baseline = form_summary["training_prompt"]
        for form in ("natural_request", "qwen_preference"):
            form_summary[form]["minus_training_prompt"] = {
                key: form_summary[form][key]["mean"] - baseline[key]["mean"]
                for key in METRICS
            }
        summary["variants"][variant] = form_summary
    write_json(DIAGNOSIS / "query_form_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "encode", "evaluate", "summarize"))
    args = parser.parse_args()
    {"generate": generate, "encode": encode, "evaluate": evaluate, "summarize": summarize}[args.action]()


if __name__ == "__main__":
    main()
