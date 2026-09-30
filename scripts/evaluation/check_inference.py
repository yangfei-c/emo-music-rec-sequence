"""Check Route2 assets and structured variable-length recommendations."""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.inference.runtime import EmoMMSequenceRuntime
from emomm_sequence.training.records import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/system_route2.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/runs/route2_uncertainty_20260926/evaluation/inference_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    runtime = EmoMMSequenceRuntime(ROOT, config_path=args.config)
    system = runtime.system
    if not system.route2:
        raise ValueError("Inference checks require a Route2 configuration")
    rows = np.random.default_rng(42).choice(len(system.track_ids), size=64, replace=False)
    with torch.inference_mode():
        x = torch.from_numpy(system.audio_embeddings[rows]).to(system.device)
        errors = {factor: float(np.max(np.abs(system.retriever.encode_factor("music", factor, x).cpu().numpy() - system.factors[factor][rows]))) for factor in system.factors}
    if max(errors.values()) > 2e-5:
        raise AssertionError(f"New index and checkpoint disagree: {errors}")
    # Structured inputs keep the sequence check independent of Ollama availability.
    cases = [
        {"current": "annoyed", "target": "calm", "preference": "piano music", "num_tracks": 3, "method": "ours"},
        {"current": "sad", "target": "happy", "preference": "guitar music", "num_tracks": 5, "method": "ours"},
        {"current": "sad", "target": "happy", "preference": "guitar music", "num_tracks": 7, "method": "point"},
        {"current": "annoyed", "target": "calm", "preference": "piano music", "num_tracks": 5, "method": "clamp3"},
    ]
    results = []
    for case in cases:
        begin = time.monotonic()
        result = system.recommend(
            case["current"], case["target"], case["preference"],
            method=case["method"], scope="unseen_library", num_tracks=case["num_tracks"],
        )
        ids = [track["track_id"] for track in result["tracks"]]
        if result.get("requested_num_tracks") != case["num_tracks"] or len(ids) != case["num_tracks"] or len(set(ids)) != case["num_tracks"]:
            raise AssertionError(f"Expected {case['num_tracks']} unique tracks for {case['method']}")
        for position, track in enumerate(result["tracks"], start=1):
            track_id = track["track_id"]
            if track["position"] != position or system.catalog_splits[system.catalog_position[track_id]] != "test":
                raise AssertionError("Returned track position or unseen-library split is incorrect")
            if not runtime.audio_path(track_id).is_file():
                raise FileNotFoundError(track_id)
            if any(key not in track for key in ("music_v", "music_a", "music_va_lower", "music_va_upper")):
                raise AssertionError("Route2 music VA prediction and interval are required")
        results.append({**result, "input": case, "model_version": system.config["run_id"], "elapsed_seconds": time.monotonic() - begin})
        print(json.dumps({"method": case["method"], "num_tracks": case["num_tracks"], "status": result["status"], "reason": result["failure_reason"], "tracks": ids, "elapsed": results[-1]["elapsed_seconds"]}, ensure_ascii=True), flush=True)
    write_json(args.output, {"config": args.config, "index_max_absolute_errors": errors, "all_assets_hash_verified": True, "cases": results, "test_type": "four_disclosed_structured_route2_examples_not_parser_or_human_evaluation"})
    print(f"Inference verified in {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
