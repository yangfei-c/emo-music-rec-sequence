"""Cache frozen supervised VA/factor predictions without fitting any test labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import ROOT, load_config, load_bank
from emomm_sequence.data.music_va import validation_partition
from emomm_sequence.calibration.cqr import VAUncertaintyModel, apply_joint_cqr
from emomm_sequence.models.frozen_retriever import FrozenRetriever
from emomm_sequence.models.text_va import load_text_va
from emomm_sequence.training.records import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    torch.set_num_threads(2)
    config, root = load_config(args.config)
    legacy_path = ROOT / config["assets"]["legacy_supervised_config"]
    legacy = yaml.safe_load(legacy_path.read_text(encoding="utf-8"))
    checks = {"legacy_config": sha256(legacy_path)}
    for section in ("retriever", "text_va", "route2"):
        entry = legacy[section]
        observed = sha256(ROOT / entry["checkpoint"])
        if observed != entry["checkpoint_sha256"]:
            raise ValueError(f"wrong frozen {section} checkpoint")
        checks[section] = observed
    if sha256(ROOT / legacy["route2"]["calibration"]) != legacy["route2"]["calibration_sha256"]:
        raise ValueError("calibration hash mismatch")
    checks["calibration"] = legacy["route2"]["calibration_sha256"]
    retriever = FrozenRetriever(ROOT / legacy["retriever"]["checkpoint"]).to(args.device)
    text_va = load_text_va(ROOT / legacy["text_va"]["checkpoint"], checks["retriever"]).to(args.device).eval()
    va = VAUncertaintyModel.from_artifacts(ROOT / legacy["route2"]["checkpoint"], ROOT / legacy["route2"]["calibration"], args.device)
    out = root / "cache" / "va_extension"
    out.mkdir(parents=True, exist_ok=True)
    with np.load(root / "cache" / "text_bank.npz", allow_pickle=False) as z:
        text_features = z["embeddings"]
    with torch.inference_mode():
        tx = torch.from_numpy(text_features).to(args.device)
        predictions = ((text_va(retriever.encode_text("affect", tx)) + 1) / 2).cpu().numpy()
        pref_text = retriever.encode_text("preference", tx).cpu().numpy()
    np.savez_compressed(out / "text.npz", predicted_va=predictions, preference_embeddings=pref_text)
    meta = json.loads((root / "protocol" / "text_protocol.json").read_text(encoding="utf-8"))
    preference_text_ids = [p["text_id"] for p in meta["preferences"]]
    manifest = {"input_hashes": checks, "refit": False, "supervised_features": True,
                "playlist_coverage_guarantee": False, "unknown_source_policy": "maximum original source correction",
                "validation_eligibility": {}, "banks": {}}
    for source in ("deam", "pmemo", "merge", "mtg"):
        for split in ("train", "validation", "test", "all"):
            bank = load_bank(root, source, split)
            quantiles = va.predict_quantiles(bank.x)
            med, lo, hi = apply_joint_cqr(quantiles, va.calibration, source if source in ("deam", "pmemo") else None)
            pref = []
            with torch.inference_mode():
                for start in range(0, len(bank.ids), 512):
                    audio = torch.from_numpy(bank.x[start:start+512]).to(args.device)
                    factor = retriever.encode_factor("music", "preference", audio).cpu().numpy()
                    pref.append(np.clip((1 - factor @ pref_text[preference_text_ids].T) / 2, 0, 1))
            np.savez_compressed(out / f"{source}_{split}.npz", track_ids=bank.ids, quantiles=quantiles,
                                cqr_lower=lo, cqr_upper=hi, preference_distances=np.concatenate(pref),
                                preference_text_ids=np.asarray(preference_text_ids))
            manifest["banks"][f"{source}_{split}"] = {"unique_content": len(bank.ids),
                                                          "source_calibration": source if source in ("deam", "pmemo") else "unknown"}
            if split == "validation":
                if source in ("deam", "pmemo"):
                    partitions = validation_partition(bank.groups, 20260906)
                    allowed = [int(i) for i, group in enumerate(bank.groups) if partitions[group] == "policy"]
                    manifest["validation_eligibility"][source] = {"partition": "existing_policy_only", "allowed_bank_indices": allowed,
                                                                  "excluded": "calibration and validation_model; never tune on calibration songs"}
                else:
                    manifest["validation_eligibility"][source] = {"partition": "validation", "allowed_bank_indices": list(range(len(bank.ids)))}
            print(f"prepared {source}/{split}: {len(bank.ids)}", flush=True)
    write_json(out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
