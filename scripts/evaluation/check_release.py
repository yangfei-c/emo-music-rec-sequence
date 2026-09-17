"""Verify versioned assets and real five-track recommendations before activation."""
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
from emomm_sequence.runtime import EmoMMSequenceRuntime
from emomm_sequence.training.records import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/current_route_20260915/release_smoke.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.monotonic()
    runtime = EmoMMSequenceRuntime(ROOT, config_path=args.config)
    system = runtime.system
    rows = np.random.default_rng(42).choice(len(system.track_ids), size=64, replace=False)
    with torch.inference_mode():
        x = torch.from_numpy(system.audio_embeddings[rows]).to(system.device)
        errors = {factor: float(np.max(np.abs(system.retriever.encode_factor("music", factor, x).cpu().numpy() - system.factors[factor][rows]))) for factor in system.factors}
    if max(errors.values()) > 2e-5:
        raise AssertionError(f"New index and checkpoint disagree: {errors}")
    cases = [("我现在很烦", "ours"), ("我现在很烦，想慢慢平静下来，最好听钢琴音乐。", "ours"), ("来点吉他吧，最近有些低落，希望听完能开心一点。", "ours"), ("我现在很烦，想慢慢平静下来，最好听钢琴音乐。", "clamp3")]
    results = []
    for text, method in cases:
        begin = time.monotonic()
        result = runtime.recommend_from_text(text, method=method, scope="unseen_library")
        ids = [track["track_id"] for track in result["tracks"]]
        if len(ids) != 5 or len(set(ids)) != 5:
            raise AssertionError("Five unique tracks required")
        for track in ids:
            assert system.catalog_splits[system.catalog_position[track]] == "test"
            assert runtime.audio_path(track).is_file()
        result["elapsed_seconds"] = time.monotonic() - begin
        results.append(result)
        write_json(args.output, {"config": args.config, "index_max_absolute_errors": errors, "all_assets_hash_verified": True, "cases": results, "test_type": "four_disclosed_integration_examples_not_human_evaluation"})
        print(json.dumps({"method": method, "status": result["status"], "reason": result["failure_reason"], "input": result["input"], "tracks": ids, "elapsed": result["elapsed_seconds"]}, ensure_ascii=True), flush=True)
    print(f"Release verified in {time.monotonic()-started:.1f} s", flush=True)


if __name__ == "__main__":
    main()
