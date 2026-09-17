"""Re-evaluate all fixed Adapter controls and export slide-ready numeric evidence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/evaluation"))
from evaluate_retrieval import evaluate, load_npz, project
from emomm_sequence.retrieval.model import RetrieverModel
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.utilities import load_yaml

REPORT = ROOT / "results/controlled_comparison_20260913"
RUNS = ROOT / "artifacts/runs/controlled_comparison_20260913"


def load_model(config, checkpoint):
    c = config["model"]
    model = RetrieverModel(
        embedding_dim=int(config["backbone"]["embedding_dim"]),
        bottleneck_dim=int(c["bottleneck_dim"]), dropout=float(c["dropout"]),
        initial_adapter_scale=float(c["initial_adapter_scale"]),
        initial_temperature=float(c["initial_temperature"]),
        initial_factor_weights=tuple(c["initial_factor_weights"]),
        train_factor_weights=bool(c["train_factor_weights"]),
        adapter_sharing=c.get("adapter_sharing", "none"),
        adapter_kind=c.get("adapter_kind", "residual"),
        adapter_blend=float(c.get("adapter_blend", 0.2)),
        semantic_output_blend=float(c.get("semantic_output_blend", 1.0)),
        affect_output_blend=float(c.get("affect_output_blend", 1.0)),
        preference_output_blend=float(c.get("preference_output_blend", 1.0)),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload["model_config"] != c or payload["selection_metric"] != "retrieval_only":
        raise ValueError("Checkpoint/config or model-selection mismatch")
    model.load_state_dict(payload["model"], strict=True)
    return model.requires_grad_(False).eval()


def extended_evaluation(model, pairs, tags, catalog):
    result = evaluate(model, pairs, tags, catalog)
    mask = tags["splits"] == "test"
    raw_music = catalog["embeddings"][tags["index_positions"][mask]]
    for factor in ("preference", "affect"):
        scores = project(model, tags[f"{factor}_label_embeddings"], "text", factor) @ project(model, raw_music, "music", factor).T
        targets = tags[f"{factor}_targets"][mask].T
        values = [(roc_auc_score(t, s), average_precision_score(t, s)) for t, s in zip(targets, scores) if np.unique(t).size == 2]
        result[f"{factor}_tags"]["score_macro_roc_auc"] = float(np.mean([x[0] for x in values]))
        result[f"{factor}_tags"]["score_macro_average_precision"] = float(np.mean([x[1] for x in values]))
    return result


def scalar_metrics(result):
    output = {}
    for dataset, value in result["semantic_pairs"].items():
        for direction in ("text_to_music", "music_to_text"):
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
                output[f"{dataset}.{direction}.{metric}"] = value[direction][metric]
    for factor in ("affect", "preference"):
        for metric, value in result[f"{factor}_tags"]["macro"].items():
            output[f"{factor}.{metric}"] = value
        for metric in ("score_macro_roc_auc", "score_macro_average_precision"):
            output[f"{factor}.{metric}"] = result[f"{factor}_tags"][metric]
    return output


def stats(values):
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "values": [float(x) for x in values], "n_seeds": len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--completed-only", action="store_true", help="Check completed groups while training continues; write interim_summary.json, never the final summary")
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = load_yaml(ROOT / "configs/training/retriever.yaml")
    s = config["supervision"]
    pairs = [load_npz(ROOT / p) for p in s["semantic_pairs"]]
    tags = load_npz(ROOT / s["mtg_factors"])
    catalog = load_npz(ROOT / s["music_index"])
    evaluated_data_hashes = {p: sha256(ROOT / p) for p in [*s["semantic_pairs"], s["mtg_factors"], s["music_index"]]}
    plan = json.loads((REPORT / "plan.json").read_text(encoding="utf-8"))
    variants = [v for v in plan["variants"] if (RUNS / v["name"] / "aggregate.json").exists()]
    if not args.baseline_only and not args.completed_only and len(variants) != len(plan["variants"]):
        raise RuntimeError("Some training groups are incomplete; final evaluation cannot run yet")
    with threadpool_limits(limits=4):
        baseline_file = REPORT / "baseline_evaluation.json"
        baseline = extended_evaluation(None, pairs, tags, catalog)
        if not baseline_file.exists():
            write_json(baseline_file, {"model": "CLaMP3", "results": baseline,
                "data_sha256": evaluated_data_hashes})
        else:
            old = json.loads(baseline_file.read_text(encoding="utf-8"))
            if old["data_sha256"] != evaluated_data_hashes:
                raise ValueError("Evaluation data changed after the first baseline evaluation")
            if scalar_metrics(old["results"]) != scalar_metrics(baseline):
                raise ValueError("Baseline changed across evaluations")
        if args.baseline_only:
            print("Baseline re-evaluated", flush=True)
            return
        rows = [{"variant": "clamp3", "name": "CLaMP3", "trainable_parameters": 0,
                 "metrics": {k: stats([v]) for k, v in scalar_metrics(baseline).items()}}]
        seed_values = {}
        for variant in variants:
            name = variant["name"]
            manifest = json.loads((RUNS / name / "input_manifest.json").read_text(encoding="utf-8"))
            trained_data_hashes = {x["path"]: x["sha256"] for x in manifest["files"]}
            if any(trained_data_hashes.get(p) != digest for p, digest in evaluated_data_hashes.items()):
                raise ValueError(f"Training/evaluation data hash mismatch: {name}")
            effective = json.loads((RUNS / name / "effective_config.json").read_text(encoding="utf-8"))
            aggregate = json.loads((RUNS / name / "aggregate.json").read_text(encoding="utf-8"))
            if aggregate["seeds"] != plan["seeds"]:
                raise ValueError("Missing or mismatched seeds")
            all_runs = []
            scalars = []
            for seed in plan["seeds"]:
                run = next(x for x in aggregate["runs"] if x["seed"] == seed)
                checkpoint = RUNS / name / f"seed_{seed}" / "best.pt"
                model = load_model(effective, checkpoint)
                result = extended_evaluation(model, pairs, tags, catalog)
                values = scalar_metrics(result)
                # Independent CPU re-evaluation must agree with the training-time GPU evaluation.
                original = run["results"]["test"]
                for dataset in ("merge", "song_describer"):
                    for metric in ("mrr", "recall_at_1", "recall_at_5", "recall_at_10"):
                        expected = original["semantic_pairs"]["per_dataset"][dataset]["text_to_music"][metric]
                        if not np.isclose(values[f"{dataset}.text_to_music.{metric}"], expected, atol=1e-6):
                            raise ValueError(f"CPU/GPU metric disagreement: {name}/{seed}/{dataset}/{metric}")
                for modality, weight in (("text", effective["loss"]["text_va_weight"]), ("music", effective["loss"]["music_va_weight"])):
                    if weight > 0:
                        for dataset, measurements in original[f"{modality}_va"]["per_dataset"].items():
                            for metric in ("valence_ccc", "arousal_ccc"):
                                values[f"aux_va.{dataset}.{metric}"] = measurements[metric]
                provenance = json.loads((checkpoint.parent / "provenance.json").read_text(encoding="utf-8"))
                all_runs.append({"seed": seed, "best_epoch": run["best_epoch"], "best_validation_score": run["best_validation_score"],
                                 "checkpoint_sha256": sha256(checkpoint), "results": result, "auxiliary_va": {k: original[k] for k in ("text_va", "music_va")}})
                scalars.append(values)
                print(f"EVALUATED {name} seed={seed}", flush=True)
            write_json(REPORT / "evaluations" / f"{name}.json", {"config": effective, "runs": all_runs})
            seed_values[name] = scalars
            rows.append({"variant": name, "name": "Ours" if name == "full" else name,
                         "trainable_parameters": provenance["trainable_parameters"],
                         "best_epochs": [x["best_epoch"] for x in all_runs],
                         "metrics": {key: stats([v[key] for v in scalars]) for key in scalars[0]}})
        primary = "affect.ndcg_at_10"
        effects = {}
        for a, b in (("full", "no_va"), ("text_va", "no_va"), ("music_va", "no_va"),
                     ("full", "text_va"), ("full", "music_va"), ("full", "shared_capacity_matched"), ("full", "clip_style_transfer")):
            if a not in seed_values or b not in seed_values:
                continue
            effects[f"{a}_minus_{b}"] = {key: stats([x[key] - y[key] for x, y in zip(seed_values[a], seed_values[b])])
                                          for key in seed_values[a][0] if key in seed_values[b][0] and not key.startswith("aux_va.")}
        summary_name = "interim_summary.json" if args.completed_only else "summary.json"
        write_json(REPORT / summary_name, {
            "run_id": plan["run_id"], "seeds": plan["seeds"], "methods": rows, "paired_effects": effects,
            "status": "interim" if args.completed_only else "complete",
            "evaluated_data_sha256": evaluated_data_hashes,
            "primary_metric": primary, "protocol": plan,
            "counts": {"merge_test_tracks": baseline["semantic_pairs"]["merge"]["tracks"],
                       "song_describer_test_tracks": baseline["semantic_pairs"]["song_describer"]["tracks"],
                       "song_describer_test_captions": baseline["semantic_pairs"]["song_describer"]["captions"],
                       "mtg_test_tracks": baseline["affect_tags"]["tracks"],
                       "affect_labels": baseline["affect_tags"]["queries"], "preference_labels": baseline["preference_tags"]["queries"]},
        })
        print(json.dumps({"summary": summary_name, "primary_full_minus_no_va": effects.get("full_minus_no_va", {}).get(primary), "methods": [{"variant": r["variant"], "parameters": r["trainable_parameters"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
