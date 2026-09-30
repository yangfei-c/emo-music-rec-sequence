"""External caption retrieval with one candidate per music-content identity."""
from __future__ import annotations
import argparse
from collections import defaultdict
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import load_config, ROOT
from emomm_sequence.losses.sequence_cost import normalize
from emomm_sequence.training.records import write_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/sequence_routes.yaml")
    args = p.parse_args()
    config, root = load_config(args.config)
    path = ROOT / config["assets"]["song_describer"]
    with np.load(path, allow_pickle=False) as z:
        text, audio = normalize(z["text_embeddings"]), normalize(z["audio_embeddings"])
        groups, ids, splits = z["content_group_ids"].astype(str), z["global_identities"].astype(str), z["splits"].astype(str)
    if not np.all(splits == "test"):
        raise ValueError("external semantic file contains non-test rows")
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    bank = audio[first]
    scores = text @ bank.T
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.argmax(order == inverse[:, None], axis=1) + 1
    per_song = defaultdict(list)
    for group, rank in zip(groups, ranks):
        per_song[group].append(int(rank))
    macro = {f"recall_at_{k}": float(np.mean([np.mean(np.array(v) <= k) for v in per_song.values()])) for k in (1, 5, 10)}
    macro["mrr"] = float(np.mean([np.mean(1 / np.array(v)) for v in per_song.values()]))
    # Audio-to-text considers every caption with the matching content identity positive.
    reverse = scores.T
    reverse_ranks = []
    for i, row in enumerate(reverse):
        ranked = np.argsort(-row, kind="stable")
        reverse_ranks.append(int(np.flatnonzero(inverse[ranked] == i)[0]) + 1)
    result = {"status": "completed", "model": "frozen_CLaMP3", "source_sha256": sha256(path),
              "caption_queries": len(text), "unique_content_candidates": len(bank),
              "text_to_audio_song_macro": macro,
              "audio_to_text_any_positive": {**{f"recall_at_{k}": float(np.mean(np.array(reverse_ranks) <= k)) for k in (1, 5, 10)},
                                              "mrr": float(np.mean(1 / np.array(reverse_ranks)))},
              "used_to_select_checkpoint": False,
              "scope": "external retrieval diagnostic shared by A and B; not a playlist quality metric"}
    write_json(root / "reports" / "e4_semantic.json", result)
    print(result)


if __name__ == "__main__":
    main()
