"""Build a versioned release from new training outputs, without switching it live."""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.relation import RegressionRelation
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.text_va import load_text_va
from emomm_sequence.training.records import sha256, write_json


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--retriever-runs", type=Path, help="Verified current-route retrieval run; selected using validation only")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/system.yaml")
    args = parser.parse_args()
    torch.set_num_threads(4)
    run_dir = ROOT / "artifacts/runs" / args.run_id
    release = ROOT / "artifacts/releases" / args.run_id
    if (release / "release_manifest.json").exists():
        raise FileExistsError("Release already complete; do not overwrite a released model")
    release.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    retrieval_runs = args.retriever_runs or run_dir / "retriever"
    if not retrieval_runs.is_absolute():
        retrieval_runs = ROOT / retrieval_runs
    aggregate = json.loads((retrieval_runs / "aggregate.json").read_text(encoding="utf-8"))
    best = max(aggregate["runs"], key=lambda r: (r["best_validation_score"], -r["seed"]))
    original = retrieval_runs / f"seed_{best['seed']}" / "best.pt"
    stages = {"retriever": retrieval_runs, "text_va": run_dir / "text_va", "music_va": run_dir / "music_va"}
    for stage, stage_root in stages.items():
        for seed in (42, 43, 44):
            provenance = json.loads((stage_root / f"seed_{seed}/provenance.json").read_text(encoding="utf-8"))
            if provenance["initial_checkpoint"] is not None or provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
                raise ValueError(f"Fresh training evidence failed: {stage}/{seed}")
            if provenance["checkpoint_sha256"] != sha256(stage_root / f"seed_{seed}/best.pt"):
                raise ValueError(f"Trained checkpoint changed: {stage}/{seed}")
    shutil.copy2(original, release / "retriever.pt")
    retriever = FrozenRetriever(release / "retriever.pt").eval()
    training_config = json.loads((retrieval_runs / "effective_config.json").read_text(encoding="utf-8"))
    if training_config["model"].get("semantic_output_blend") != 0.2:
        raise ValueError("Release must use the current semantic-preservation route")
    text_aggregate = json.loads((run_dir / "text_va/aggregate.json").read_text(encoding="utf-8"))
    text_seed = text_aggregate["selected_seed"]
    text_source = run_dir / "text_va" / f"seed_{text_seed}" / "best.pt"
    load_text_va(text_source, sha256(original))  # Reject mismatched frozen Adapter.
    music_audit = json.loads((run_dir / "music_va/data_audit.json").read_text(encoding="utf-8"))
    if music_audit["retriever_checkpoint_sha256"] != sha256(original):
        raise ValueError("Music VA relation pairs were prepared with a different retriever")
    shutil.copy2(text_source, release / "text_va.pt")
    config["text_va"] = {"checkpoint": relative(release / "text_va.pt"),
                         "checkpoint_sha256": sha256(release / "text_va.pt"),
                         "selected_seed": text_seed, "selection_basis": "validation_mean_ccc_only",
                         "input": "selected_frozen_text_affect_adapter", "target_scale": [-1, 1]}
    config["affect_direction"]["unknown_text_fallback"] = "emobank_text_va"
    with np.load(ROOT / training_config["supervision"]["music_index"], allow_pickle=False) as data:
        catalog = {key: data[key] for key in ("track_ids", "audio_paths", "embeddings")}
    catalog_dir = release / "catalog"
    catalog_dir.mkdir(exist_ok=True)
    for factor in ("semantic", "affect", "preference"):
        with torch.inference_mode():
            embeddings = np.concatenate([retriever.encode_factor("music", factor, torch.from_numpy(catalog["embeddings"][i:i+512].astype(np.float32))).numpy() for i in range(0, len(catalog["track_ids"]), 512)])
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError(f"Invalid new index: {factor}")
        path = catalog_dir / f"{factor}.npy"
        np.save(path, embeddings)
        config["catalog"]["factors"][factor] = {"path": relative(path), "sha256": sha256(path)}
    np.save(catalog_dir / "factor_track_ids.npy", catalog["track_ids"])
    metadata_source = ROOT / training_config["supervision"]["mtg_factors"]
    shutil.copy2(metadata_source, catalog_dir / "metadata.npz")
    relation_dir = release / "music_va"
    relation_dir.mkdir(exist_ok=True)
    va_predictions = []
    relation_cfg = config["relation"]
    relation_cfg["checkpoints_sha256"] = {}
    relation_cfg["calibration_sha256"] = {}
    for seed in relation_cfg["seeds"]:
        source = run_dir / "music_va" / f"seed_{seed}"
        checkpoint, calibration = relation_dir / f"seed_{seed}.pt", relation_dir / f"seed_{seed}.calibration.json"
        shutil.copy2(source / "best.pt", checkpoint)
        shutil.copy2(source / "calibration.json", calibration)
        relation_cfg["checkpoints_sha256"][seed] = sha256(checkpoint)
        relation_cfg["calibration_sha256"][seed] = sha256(calibration)
        model = RegressionRelation(hidden_dim=relation_cfg["hidden_dim"], dropout=relation_cfg["dropout"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
        model.eval().requires_grad_(False)
        with torch.inference_mode():
            va_predictions.append(np.concatenate([model.coordinates(torch.from_numpy(catalog["embeddings"][i:i+512])).numpy() for i in range(0, len(catalog["track_ids"]), 512)]))
    catalog["music_va"] = np.mean(va_predictions, axis=0).astype(np.float32)
    np.savez_compressed(catalog_dir / "music.npz", **catalog)
    for key, filename in (("music", "music.npz"), ("metadata", "metadata.npz"), ("factor_track_ids", "factor_track_ids.npy")):
        path = catalog_dir / filename
        config["catalog"][key] = relative(path)
        config["catalog"][key + "_sha256"] = sha256(path)
    config["name"], config["run_id"] = "Ours", args.run_id
    config["retriever"].update({"model": "Ours", "checkpoint": relative(release / "retriever.pt"), "checkpoint_sha256": sha256(release / "retriever.pt"), "selected_seed": best["seed"], "selection_basis": "factorwise_balanced_validation_only", "semantic_output_blend": 0.2, "training_runs": relative(retrieval_runs)})
    config["query_parser"] = {"model": "qwen3:4b", "base_url": "http://127.0.0.1:11434", "expected_digest": "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7", "strict_digest": True, "timeout_seconds": 120, "seed": 42, "frozen": True, "missing_target": "calm_explicit_default", "missing_preference": "no_preference_constraint"}
    relation_cfg["checkpoint_template"] = relative(relation_dir / "seed_{seed}.pt")
    relation_cfg["calibration_template"] = relative(relation_dir / "seed_{seed}.calibration.json")
    candidate = release / "system.yaml"
    candidate.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(release / "release_manifest.json", {"run_id": args.run_id, "retriever_seed": best["seed"], "text_va_seed": text_seed, "catalog_tracks": len(catalog["track_ids"]), "music_va_seeds": relation_cfg["seeds"], "training_sources": {key: relative(path) for key, path in stages.items()}, "configuration": relative(candidate), "configuration_sha256": sha256(candidate), "all_trainable_deployed_weights_have_training_provenance": True, "live_config_switched": False, "files": {relative(path): sha256(path) for path in sorted(release.rglob("*")) if path.is_file()}})
    print(f"Built {candidate}; production config is unchanged", flush=True)


if __name__ == "__main__":
    main()
