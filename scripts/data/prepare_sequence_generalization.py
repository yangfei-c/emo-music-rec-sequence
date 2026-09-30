"""Freeze separate length/scale/artist manifests without changing primary tasks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import ROOT, load_config, load_bank, read_jsonl, write_jsonl
from emomm_sequence.training.records import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = p.parse_args()
    config, root = load_config(args.config)
    source = root / "tasks" / args.profile / f"evaluate_{args.split}.jsonl"
    tasks = read_jsonl(source)
    base = {}
    for task in tasks:
        if task["suite"] in ("e2", "e3") and task["phrase_index"] == 2:
            base.setdefault((task["source"], task["query_family_id"]), task)
    by_kind = {"lengths": [], "artist_disjoint": [], "all_library": []}
    with np.load(ROOT / config["assets"]["mtg_labels"], allow_pickle=False) as z:
        ids, artists, splits = z["track_ids"].astype(str), z["artist_ids"].astype(str), z["splits"].astype(str)
    train_artists = set(artists[splits == "train"])
    artist_by_id = dict(zip(ids, artists))
    mtg = load_bank(root, "mtg", args.split)
    eligible = [i for i, track in enumerate(mtg.ids) if artist_by_id[track] not in train_artists and artist_by_id[track] not in ("", "nan", "None", "unknown")]
    if len(eligible) < 20:
        raise ValueError("artist holdout too small; do not relabel an overlapping bank as disjoint")
    for (source_name, _), task in base.items():
        for n in (10, 15, 20):
            for budget in (32, 64, 128):
                by_kind["lengths"].append({**task, "num_tracks": n, "budget": budget, "extension": "unseen_length"})
        if source_name == "mtg":
            for n in (5, 10):
                for budget in (64, 128):
                    by_kind["artist_disjoint"].append({**task, "indices": eligible, "num_tracks": n, "budget": budget,
                                                        "extension": "artist_disjoint_from_all_training_artists"})
                    by_kind["all_library"].append({**task, "split": "all", "indices": None, "num_tracks": n, "budget": budget,
                                                    "extension": "deployment_all_library_includes_training_songs"})
    directory = root / "tasks" / args.profile / "generalization" / args.split
    provenance = {"source_task_sha256": sha256(source), "mtg_labels_sha256": sha256(ROOT / config["assets"]["mtg_labels"]),
                  "artist_eligible": len(eligible), "artist_identity_scope": "MTG only; cross-dataset artist identities are not established", "files": {}}
    for kind, rows in by_kind.items():
        for i, row in enumerate(rows):
            row["query_id"] = f"{args.profile}_{args.split}_{kind}_{i:06d}"
        path = directory / f"{kind}.jsonl"
        if path.exists():
            if read_jsonl(path) != rows:
                raise ValueError("existing generalization tasks differ; preserve and version them")
        else:
            write_jsonl(path, rows)
        provenance["files"][kind] = {"sha256": sha256(path), "requests": len(rows), "path": str(path)}
    write_json(directory / "manifest.json", provenance)
    print(json.dumps(provenance))


if __name__ == "__main__":
    main()
