"""Executable evidence for the current route, real feature inputs and weights."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.encoding.clamp3 import Clamp3TextAudioEncoder, collate_audio_tokens, load_audio_tokens
from emomm_sequence.retriever import FrozenRetriever
from emomm_sequence.training.records import sha256, write_json
from emomm_sequence.training.utilities import load_yaml


def audit_checkpoints(run_root: Path) -> dict:
    report = {}
    for directory in sorted(run_root.glob("seed_*")):
        initial = torch.load(directory / "initial.pt", map_location="cpu", weights_only=True)["model"]
        payload = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
        model = FrozenRetriever(directory / "best.pt")
        changes = {}
        for side in ("text", "music"):
            for factor in ("semantic", "affect", "preference"):
                prefix = f"{side}_adapters.{factor}."
                keys = [key for key in initial if key.startswith(prefix)]
                changes[f"{side}_{factor}"] = max(float((initial[key] - payload["model"][key]).abs().max()) for key in keys)
        if not all(value > 0 for value in changes.values()):
            raise AssertionError(f"An Adapter did not update: {changes}")
        assert model.semantic_output_blend == .2
        with torch.inference_mode():
            vectors = {f"{side}_{factor}": model.encode_factor(side, factor, torch.eye(4, 768)).numpy()
                       for side in ("text", "music") for factor in ("semantic", "affect", "preference")}
        assert all(np.isfinite(value).all() and np.allclose(np.linalg.norm(value, axis=1), 1, atol=1e-5) for value in vectors.values())
        report[directory.name] = {"checkpoint_sha256": sha256(directory / "best.pt"),
                                  "all_six_adapters_changed": True, "max_absolute_change": changes,
                                  "strict_reload_and_normalized_output": True, "semantic_output_blend": .2}
    if not report:
        raise FileNotFoundError(run_root)
    return report


def audit_encoder(output: Path) -> dict:
    cfg = load_yaml(ROOT / "configs/data.yaml")["clamp3"]
    spec = importlib.util.spec_from_file_location("feature_extractor", ROOT / "scripts/features/extract_clamp3_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Clamp3TextAudioEncoder(ROOT / cfg["checkpoint"], normalize=False).to(device).eval()
    assert not any(parameter.requires_grad for parameter in model.parameters())
    frame = pd.read_csv(ROOT / "data/manifests/pairs_merge.csv", keep_default_na=False)
    sample = frame.iloc[0]
    tokenizer = model.tokenizer(cfg["text_model"])
    text = module.encode_texts([sample["text"]], model, tokenizer, device, 8, cfg["max_text_length"])
    tokens = load_audio_tokens(Path(sample["feature_path"]))
    audio_batch, mask = collate_audio_tokens([tokens])
    with torch.inference_mode():
        audio = torch.nn.functional.normalize(model.encode_audio(audio_batch.to(device), mask.to(device)), dim=-1).cpu().numpy()
    with np.load(ROOT / "artifacts/features/clamp3/clean/pairs_merge_clean.npz", allow_pickle=False) as data:
        positions = np.flatnonzero(data["global_identities"].astype(str) == sample["global_identity"])
        if len(positions) != 1:
            raise ValueError("Cannot identify real MERGE sample uniquely")
        i = positions[0]
        errors = {"text": float(np.abs(data["text_embeddings"][i] - text[0]).max()),
                  "audio": float(np.abs(data["audio_embeddings"][i] - audio[0]).max())}
    if max(errors.values()) > 1e-4:
        raise AssertionError(f"Extracted features disagree with training features: {errors}")
    np.savez_compressed(output / "feature_roundtrip.npz", text=text, audio=audio)
    result = {"real_sample": sample["global_identity"], "text_shape": list(text.shape), "audio_shape": list(audio.shape),
            "cached_training_feature_max_absolute_errors": errors, "backbone_frozen": True,
            "mert_tokens_path": sample["feature_path"], "audio_path": sample["audio_path"]}
    strict_tokens = output / "mert_strict_roundtrip.npy"
    if strict_tokens.is_file():
        raw, cached = np.load(strict_tokens), np.load(sample["feature_path"])
        error = float(np.abs(raw - cached).max())
        assert error < 1e-4, "Raw audio tokens disagree with the declared training input"
        result["raw_mert_reextraction"] = {"shape": list(raw.shape), "max_absolute_error": error,
                                          "strict_weight_loading": True, "file_sha256": sha256(strict_tokens)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/current_route_20260915")
    parser.add_argument("--skip-encoder", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"current_full_training": audit_checkpoints(ROOT / "artifacts/runs/semantic_preservation_20260914/preserved_multiview"),
              "fresh_one_epoch_verification": audit_checkpoints(ROOT / "artifacts/runs/current_route_20260915/retriever_verification")}
    if not args.skip_encoder:
        report["encoder"] = audit_encoder(args.output)
    write_json(args.output / "pipeline_audit.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
