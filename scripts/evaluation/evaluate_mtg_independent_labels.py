"""Evaluate route-2 sequence endpoints against held-out MTG mood votes.

The official MTG annotations contain categorical *expressed music mood* votes.
This experiment measures whether the first and last recommended tracks carry
the requested categories. It cannot measure a continuous VA trajectory, a
listener's emotional response, or coverage of the conformal intervals after
retrieval/sequence selection.

Run with the selected Route2 checkpoints, for example::

    D:\\Anaconda3\\envs\\emmr\\python.exe scripts/evaluation/evaluate_mtg_independent_labels.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

LABEL_PATH = ROOT / "data/raw/mtg-jamendo/music-classification-annotations-clean.tsv"
LABEL_SHA256 = "9b060b3db785d475870ceb0682241782bf20053d751863fbd42e02cf11088e1d"
OUTPUT_PATH = ROOT / "artifacts/runs/route2_uncertainty_20260926/evaluation/mtg_independent_endpoint_labels.json"
MOODS = ("sad", "happy", "aggressive", "relaxed")
REQUESTS = (
    ("sad_to_happy", "sad", "happy"),
    ("happy_to_sad", "happy", "sad"),
    ("aggressive_to_relaxed", "aggressive", "relaxed"),
    ("relaxed_to_aggressive", "relaxed", "aggressive"),
)
METHODS = ("ours", "point", "clamp3")
LENGTHS = (5, 7)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_consensus_moods(path: Path) -> dict[str, dict[str, bool]]:
    """Parse variable-width official TSV; an absent label is not a negative vote."""
    result: dict[str, dict[str, bool]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        header = stream.readline().rstrip("\r\n").split("\t")
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ValueError("unexpected MTG annotation header")
        for line_number, line in enumerate(stream, start=2):
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 6 or not fields[0].startswith("track_"):
                raise ValueError(f"bad MTG annotation row {line_number}")
            track_id = fields[0]
            if track_id in result:
                raise ValueError(f"duplicate MTG track: {track_id}")
            moods: dict[str, bool] = {}
            for field in fields[5:]:
                if not field.startswith("mood_"):
                    continue
                try:
                    tag, vote_string = field.split("---", 1)
                except ValueError as error:
                    raise ValueError(f"bad mood field on row {line_number}") from error
                mood = tag.removeprefix("mood_")
                if mood not in MOODS:
                    continue
                votes = vote_string.split(",")
                if len(votes) != 3 or len(set(votes)) != 1:
                    raise ValueError(f"non-consensus mood field on row {line_number}")
                if votes[0] not in (mood, f"not_{mood}"):
                    raise ValueError(f"unexpected mood vote on row {line_number}: {field}")
                if mood in moods:
                    raise ValueError(f"duplicate mood field on row {line_number}")
                moods[mood] = votes[0] == mood
            result[track_id] = moods
    return result


def label_state(labels: dict[str, dict[str, bool]], track_id: str, mood: str) -> str:
    value = labels.get(track_id, {}).get(mood)
    if value is None:
        return "unobserved"
    return "positive" if value else "negative"


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    first = Counter(item["first_expected_label"] for item in cases)
    last = Counter(item["last_expected_label"] for item in cases)
    both_observed = sum(item["both_endpoints_observed"] for item in cases)
    both_positive = sum(item["both_endpoints_positive"] for item in cases)
    observed_endpoints = first["positive"] + first["negative"] + last["positive"] + last["negative"]
    return {
        "request_count": count,
        "first_label_counts": dict(first),
        "last_label_counts": dict(last),
        "first_positive_rate_all_requests": _rate(first["positive"], count),
        "first_positive_rate_observed": _rate(first["positive"], first["positive"] + first["negative"]),
        "last_positive_rate_all_requests": _rate(last["positive"], count),
        "last_positive_rate_observed": _rate(last["positive"], last["positive"] + last["negative"]),
        "endpoint_label_observed_rate": _rate(observed_endpoints, 2 * count),
        "endpoint_label_unobserved_rate": _rate(2 * count - observed_endpoints, 2 * count),
        "both_endpoints_observed_count": both_observed,
        "both_endpoints_positive_count": both_positive,
        "both_endpoints_positive_rate_all_requests": _rate(both_positive, count),
        "both_endpoints_positive_rate_both_observed": _rate(both_positive, both_observed),
    }


def evaluate_case(
    result: dict[str, Any], labels: dict[str, dict[str, bool]],
    request_name: str, start_mood: str, target_mood: str,
    method: str, num_tracks: int, elapsed_seconds: float,
) -> dict[str, Any]:
    tracks = result["tracks"]
    if len(tracks) != num_tracks or len({track["track_id"] for track in tracks}) != num_tracks:
        raise AssertionError("the system did not return the requested unique sequence length")
    detailed_tracks = [
        {
            "position": item["position"],
            "track_id": item["track_id"],
            "requested_start_mood_label": label_state(labels, item["track_id"], start_mood),
            "requested_target_mood_label": label_state(labels, item["track_id"], target_mood),
        }
        for item in tracks
    ]
    first_state = detailed_tracks[0]["requested_start_mood_label"]
    last_state = detailed_tracks[-1]["requested_target_mood_label"]
    return {
        "request": request_name,
        "current_mood": start_mood,
        "target_mood": target_mood,
        "method": method,
        "num_tracks": num_tracks,
        "status": result["status"],
        "failure_reason": result["failure_reason"],
        "direction_source": result["direction_source"],
        "direction": result["direction"],
        "candidate_count": result["candidate_count"],
        "elapsed_seconds": round(elapsed_seconds, 3),
        "first_expected_label": first_state,
        "last_expected_label": last_state,
        "both_endpoints_observed": first_state != "unobserved" and last_state != "unobserved",
        "both_endpoints_positive": first_state == "positive" and last_state == "positive",
        "tracks": detailed_tracks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/system_route2.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--scope", choices=("unseen_library", "all_library"), default="unseen_library")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--force", action="store_true", help="replace an existing report")
    args = parser.parse_args()

    if args.output.exists() and not args.force:
        raise FileExistsError(f"report exists: {args.output}; pass --force to replace it")
    actual_label_hash = sha256(LABEL_PATH)
    if actual_label_hash != LABEL_SHA256:
        raise ValueError("MTG consensus annotation SHA256 differs from frozen evaluation source")
    labels = load_consensus_moods(LABEL_PATH)

    from emomm_sequence.inference.system import EmoMMSequence

    system = EmoMMSequence(root=ROOT, config=args.config, device=args.device)
    if not system.route2:
        raise ValueError("this evaluation requires the route-2 config")
    if args.scope != "unseen_library":
        raise ValueError("the pre-registered independent-label evaluation uses unseen_library only")
    catalog_test = set(system.track_ids[system.catalog_splits == "test"].astype(str))
    label_ids = set(labels)
    overlap = catalog_test & label_ids
    if not overlap:
        raise ValueError("no official mood labels overlap the unseen catalog")

    source_counts: dict[str, dict[str, int]] = {}
    for mood in MOODS:
        observed = [labels[track_id][mood] for track_id in overlap if mood in labels[track_id]]
        source_counts[mood] = {
            "positive": sum(observed), "negative": len(observed) - sum(observed),
            "unobserved": len(overlap) - len(observed),
        }

    cases: list[dict[str, Any]] = []
    for request_name, start_mood, target_mood in REQUESTS:
        for num_tracks in LENGTHS:
            for method in METHODS:
                started = time.monotonic()
                recommendation = system.recommend(
                    start_mood, target_mood, "", method=method,
                    scope=args.scope, num_tracks=num_tracks,
                )
                elapsed = time.monotonic() - started
                case = evaluate_case(
                    recommendation, labels, request_name, start_mood,
                    target_mood, method, num_tracks, elapsed,
                )
                for track in case["tracks"]:
                    if track["track_id"] not in catalog_test:
                        raise AssertionError("selected song is outside the frozen unseen split")
                cases.append(case)
                print(
                    f"{request_name} N={num_tracks} {method}: "
                    f"first={case['first_expected_label']} last={case['last_expected_label']} "
                    f"status={case['status']} {elapsed:.1f}s", flush=True,
                )

    by_method_length = {
        f"{method}_N{num_tracks}": summarize_cases([
            case for case in cases
            if case["method"] == method and case["num_tracks"] == num_tracks
        ])
        for method in METHODS for num_tracks in LENGTHS
    }
    report = {
        "run_id": "route2_uncertainty_20260926_mtg_independent_endpoint_labels",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "evaluation_target": "official held-out three-vote consensus categorical labels for selected first and last songs",
            "requests_predeclared": [
                {"name": name, "current_mood": start, "target_mood": target}
                for name, start, target in REQUESTS
            ],
            "methods": list(METHODS), "lengths": list(LENGTHS),
            "scope": args.scope, "music_preference": "",
            "no_new_human_ratings": True,
            "limits": [
                "Four fixed categorical prompts are a diagnostic, not a broad claim of generalization.",
                "The clean file keeps only unanimous votes; missing category votes reflect missing or non-unanimous annotations and are unobserved, not negative labels.",
                "Observed-label positive rates are conditional on unanimous votes and can be selection-biased.",
                "These labels support only categorical song-mood endpoint evaluation, not continuous VA full-path evaluation or listener state change.",
                "DEAM/PMEmo song-level marginal CQR coverage is not 90% coverage for MTG catalog songs selected by retrieval and search.",
                "The MTG test tracks are unseen in the catalog split; related MTG labels helped train the frozen retrieval model on other tracks, so this is held-out-song evaluation, not a fully external dataset.",
                "Ours and point use the same fixed start/end retrieval anchors for each request; this endpoint-label test cannot distinguish their middle-sequence planners.",
            ],
        },
        "sources": {
            "annotation_url": "https://github.com/MTG/mtg-jamendo-dataset/blob/master/derived/music-classification-annotations/README.md",
            "annotation_path": str(LABEL_PATH.relative_to(ROOT)),
            "annotation_sha256": actual_label_hash,
            "route2_config": str(Path(args.config)),
            "route2_config_sha256": system.config_sha256,
            "official_consensus_tracks": len(labels),
            "catalog_unseen_tracks": len(catalog_test),
            "official_consensus_unseen_overlap": len(overlap),
            "overlap_mood_label_counts": source_counts,
        },
        "cases": cases,
        "summary_by_method_and_length": by_method_length,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
