"""Select one reproducible Route2 experiment for local inference."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.models.text_va import load_text_va
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.calibration.cqr import VAUncertaintyModel


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_run(stage: str, directory: Path) -> dict:
    aggregate = read_json(directory / "aggregate.json")
    for seed in aggregate["seeds"]:
        folder = directory / f"seed_{seed}"
        provenance = read_json(folder / "provenance.json")
        if provenance.get("initial_checkpoint") is not None:
            raise ValueError(f"{stage} seed {seed} was warm-started")
        if provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
            raise ValueError(f"{stage} seed {seed} did not train")
        if provenance["checkpoint_sha256"] != sha256(folder / "best.pt"):
            raise ValueError(f"{stage} seed {seed} checkpoint changed")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--retriever-runs", type=Path)
    parser.add_argument("--text-va-runs", type=Path)
    parser.add_argument("--quantile-runs", type=Path)
    parser.add_argument("--evaluation", type=Path, required=True,
                        help="Route2 report.json with policy selected before test evaluation")
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/system_route2.yaml")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate training provenance and policy binding without writing a selection")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,80}", args.run_id):
        parser.error("Use a simple run ID with letters, numbers, underscores or hyphens")
    torch.set_num_threads(4)

    run_dir = ROOT / "artifacts/runs" / args.run_id
    selected = run_dir / "selected"
    if selected.exists():
        raise FileExistsError(f"Refusing to overwrite an existing selection: {selected}")
    stages = {
        "retriever": args.retriever_runs or run_dir / "retriever",
        "text_va": args.text_va_runs or run_dir / "text_va",
        "music_va_quantiles": args.quantile_runs or run_dir / "music_va_quantiles",
    }
    stages = {name: path if path.is_absolute() else ROOT / path for name, path in stages.items()}
    evaluation_path = args.evaluation if args.evaluation.is_absolute() else ROOT / args.evaluation
    base_config = args.base_config if args.base_config.is_absolute() else ROOT / args.base_config
    retriever_aggregate = check_run("retriever", stages["retriever"])
    text_aggregate = check_run("text_va", stages["text_va"])
    quantile_aggregate = check_run("music_va_quantiles", stages["music_va_quantiles"])
    retriever_choice = max(retriever_aggregate["runs"], key=lambda row: (row["best_validation_score"], -row["seed"]))
    retriever_seed = int(retriever_choice["seed"])
    text_seed = int(text_aggregate["selected_seed"])
    quantile_seed = int(quantile_aggregate["selected_seed_by_validation_only"])
    retriever_source = stages["retriever"] / f"seed_{retriever_seed}/best.pt"
    text_source = stages["text_va"] / f"seed_{text_seed}/best.pt"
    quantile_source = stages["music_va_quantiles"] / f"seed_{quantile_seed}/best.pt"
    calibration_source = stages["music_va_quantiles"] / f"seed_{quantile_seed}/calibration.json"
    load_text_va(text_source, sha256(retriever_source))
    VAUncertaintyModel.from_artifacts(quantile_source, calibration_source)
    training_config = read_json(stages["retriever"] / "effective_config.json")
    if training_config["model"].get("semantic_output_blend") != 0.2:
        raise ValueError("Route2 requires the selected semantic-preservation retriever")
    policy_report = read_json(evaluation_path)
    if Path(policy_report["run"]).resolve() != stages["music_va_quantiles"].resolve():
        raise ValueError("Policy report belongs to a different quantile run")
    if quantile_seed != int(policy_report["seed_selected_by_validation_only"]):
        raise ValueError("Policy report selected a different quantile seed")
    policy = policy_report["policy"]
    if policy["selection_partition"] != "policy":
        raise ValueError("Route2 policy was not selected on the policy partition")
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    if "route2" not in config or "relation" in config:
        raise ValueError("Base configuration must be Route2 only")
    if args.check_only:
        print(json.dumps({"status": "validated", "retriever_seed": retriever_seed,
                          "text_va_seed": text_seed, "quantile_seed": quantile_seed,
                          "policy": {"lambda_retrieval": policy["selected_lambda_retrieval"],
                                     "beam_width": policy["selected_beam_width"]}}, indent=2))
        return

    selected.mkdir(parents=True)
    retriever = FrozenRetriever(retriever_source).eval()
    with np.load(ROOT / training_config["supervision"]["music_index"], allow_pickle=False) as loaded:
        catalog = {key: loaded[key] for key in ("track_ids", "audio_paths", "embeddings")}
    catalog_dir = selected / "catalog"
    catalog_dir.mkdir()
    for factor in ("semantic", "affect", "preference"):
        with torch.inference_mode():
            embeddings = np.concatenate([
                retriever.encode_factor("music", factor, torch.from_numpy(catalog["embeddings"][i:i + 512].astype(np.float32))).numpy()
                for i in range(0, len(catalog["track_ids"]), 512)
            ])
        if not np.isfinite(embeddings).all() or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-4):
            raise ValueError(f"Invalid factor index: {factor}")
        path = catalog_dir / f"{factor}.npy"
        np.save(path, embeddings)
        config["catalog"]["factors"][factor] = {"path": relative(path), "sha256": sha256(path)}
    np.save(catalog_dir / "factor_track_ids.npy", catalog["track_ids"])
    shutil.copy2(ROOT / training_config["supervision"]["mtg_factors"], catalog_dir / "metadata.npz")
    np.savez_compressed(catalog_dir / "music.npz", **catalog)
    for key, filename in (("music", "music.npz"), ("metadata", "metadata.npz"), ("factor_track_ids", "factor_track_ids.npy")):
        path = catalog_dir / filename
        config["catalog"][key] = relative(path)
        config["catalog"][key + "_sha256"] = sha256(path)

    config["name"], config["run_id"] = "Route2-CQR-Robust", args.run_id
    config["retriever"].update({
        "checkpoint": relative(retriever_source),
        "checkpoint_sha256": sha256(retriever_source),
        "selected_seed": retriever_seed,
        "training_runs": relative(stages["retriever"]),
    })
    config["text_va"].update({
        "checkpoint": relative(text_source),
        "checkpoint_sha256": sha256(text_source),
        "selected_seed": text_seed,
    })
    config["route2"].update({
        "checkpoint": relative(quantile_source),
        "checkpoint_sha256": sha256(quantile_source),
        "calibration": relative(calibration_source),
        "calibration_sha256": sha256(calibration_source),
        "lambda_retrieval": policy["selected_lambda_retrieval"],
        "beam_width": policy["selected_beam_width"],
        "selection_basis": policy["criterion"],
        "calibration_alpha": read_json(calibration_source)["alpha"],
    })
    candidate = selected / "system.yaml"
    candidate.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    write_json(selected / "selection_manifest.json", {
        "run_id": args.run_id,
        "retriever_seed": retriever_seed,
        "text_va_seed": text_seed,
        "quantile_seed": quantile_seed,
        "catalog_tracks": len(catalog["track_ids"]),
        "training_sources": {name: relative(path) for name, path in stages.items()},
        "policy_report": relative(evaluation_path),
        "configuration": relative(candidate),
        "configuration_sha256": sha256(candidate),
        "all_selected_trainable_weights_have_training_provenance": True,
        "active_config_changed": False,
        "files": {relative(path): sha256(path) for path in sorted(selected.rglob("*")) if path.is_file()},
    })
    print(f"Selected {candidate}; active config is unchanged", flush=True)


if __name__ == "__main__":
    main()
