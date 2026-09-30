"""Build leakage-aware manifests for the frozen-CLaMP3 VA experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.training.utilities import load_yaml
from emomm_sequence.data.tags import read_mtg_autotagging_tsv


def normalize_va(value: float, low: float, high: float) -> float:
    if high <= low:
        raise ValueError("invalid VA source scale")
    normalized = 2.0 * (float(value) - low) / (high - low) - 1.0
    if not -1.000001 <= normalized <= 1.000001:
        raise ValueError(f"VA value {value} is outside [{low},{high}]")
    return max(-1.0, min(1.0, normalized))


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def stable_split(identity: str) -> str:
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def write_manifest(rows: list[dict], path: Path) -> None:
    if not rows:
        raise ValueError(f"empty manifest: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError(f"inconsistent manifest fields: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def emobank_manifest(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    split_names = {"train": "train", "dev": "validation", "test": "test"}
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            {
                "sample_id": str(row.id),
                "dataset": "emobank",
                "split": split_names[str(row.split)],
                "text": clean_text(row.text),
                "valence": normalize_va(row.V, 1.0, 5.0),
                "arousal": normalize_va(row.A, 1.0, 5.0),
                "has_va": 1,
                "global_identity": f"emobank:{row.id}",
            }
        )
    if len(rows) != 10062:
        raise ValueError(f"EmoBank row count mismatch: {len(rows)}")
    return rows


def music_va_manifest(name: str, paths: dict) -> list[dict]:
    annotation_file = Path(paths[f"{name}_annotations"])
    split_dir = Path(paths[f"{name}_splits"])
    feature_root = Path(paths[f"{name}_features"])
    audio_root = Path(paths[f"{name}_audio"])
    annotations = pd.read_csv(annotation_file)
    by_id = {
        str(int(float(row.song_id))): (float(row.valence_mean), float(row.arousal_mean))
        for row in annotations.itertuples(index=False)
    }
    split_by_id = {}
    for split in ("train", "validation", "test"):
        ids = [line.strip() for line in (split_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        for song_id in ids:
            canonical = str(int(float(song_id)))
            if canonical in split_by_id:
                raise ValueError(f"{name} split overlap: {canonical}")
            split_by_id[canonical] = split
    if set(by_id) != set(split_by_id):
        raise ValueError(f"{name} annotations and splits disagree")
    audio_lookup = {path.stem: path for path in audio_root.glob("*.*")}
    rows = []
    for song_id in sorted(by_id, key=int):
        valence, arousal = by_id[song_id]
        feature_path = feature_root / song_id
        audio_path = audio_lookup.get(song_id)
        if not feature_path.is_dir() or audio_path is None:
            raise FileNotFoundError(f"{name} missing audio/features: {song_id}")
        rows.append(
            {
                "sample_id": f"{name}:{song_id}",
                "dataset": name,
                "split": split_by_id[song_id],
                "feature_path": feature_path.as_posix(),
                "audio_path": audio_path.as_posix(),
                "valence": normalize_va(valence, 1.0, 9.0),
                "arousal": normalize_va(arousal, 1.0, 9.0),
                "has_va": 1,
                "global_identity": f"{name}:{song_id}",
            }
        )
    return rows


def merge_manifest(root: Path, feature_root: Path) -> list[dict]:
    values = pd.read_csv(root / "merge_bimodal_complete_av_values.csv")
    split_root = root / "tvt_dataframes" / "tvt_70_15_15"
    split_by_song = {}
    for split, token in (("train", "train"), ("validation", "validate"), ("test", "test")):
        file = next(split_root.glob(f"*_{token}_*.csv"))
        for song in pd.read_csv(file)["Song"].astype(str):
            if song in split_by_song:
                raise ValueError(f"MERGE split overlap: {song}")
            split_by_song[song] = split
    audio_lookup = {path.stem: path for path in (root / "audio").rglob("*.mp3")}
    lyric_lookup = {}
    for path in (root / "lyrics").rglob("*.txt"):
        lyric_id = path.stem
        if lyric_id in lyric_lookup:
            raise ValueError(f"duplicate MERGE lyric id: {lyric_id}")
        lyric_lookup[lyric_id] = path
    rows = []
    for row in values.itertuples(index=False):
        audio_id, lyric_id = str(row.Audio_Song), str(row.Lyric_Song)
        audio_path, lyric_path = audio_lookup.get(audio_id), lyric_lookup.get(lyric_id)
        if audio_path is None or lyric_path is None or audio_id not in split_by_song:
            raise FileNotFoundError(f"MERGE incomplete pair: {audio_id}/{lyric_id}")
        relative = audio_path.relative_to(root / "audio").with_suffix(".npy")
        rows.append(
            {
                "pair_id": f"merge:{audio_id}:{lyric_id}",
                "dataset": "merge",
                "split": split_by_song[audio_id],
                "text": clean_text(lyric_path.read_text(encoding="utf-8", errors="replace")),
                "feature_path": (feature_root / relative).as_posix(),
                "audio_path": audio_path.as_posix(),
                "valence": normalize_va(row.Valence, 0.0, 1.0),
                "arousal": normalize_va(row.Arousal, 0.0, 1.0),
                "has_va": 1,
                "global_identity": f"merge:{audio_id}",
                "alignment_verified": 1,
            }
        )
    if len(rows) != 2216:
        raise ValueError(f"MERGE pair count mismatch: {len(rows)}")
    return rows


def song_describer_manifest(
    root: Path,
    mtg_audio: Path,
    mtg_features: Path,
    precomputed_embeddings: Path,
    official_split_dir: Path,
) -> list[dict]:
    frame = pd.read_csv(root / "song_describer.csv")
    official = {}
    for split in ("train", "validation", "test"):
        for track in read_mtg_autotagging_tsv(official_split_dir / f"autotagging-{split}.tsv", split):
            canonical_id = str(int(track.track_id.removeprefix("track_")))
            if canonical_id in official:
                raise ValueError(f"Official MTG split overlaps: {canonical_id}")
            official[canonical_id] = split
    rows = []
    for row in frame.itertuples(index=False):
        relative = Path(str(row.path).replace("\\", "/"))
        audio_path = mtg_audio / relative
        feature_path = mtg_features / relative.with_suffix("")
        if not audio_path.is_file() or not feature_path.is_dir():
            raise FileNotFoundError(f"Song Describer cannot resolve {relative}")
        track_id = str(row.track_id)
        track_stem = f"track_{int(row.track_id):07d}"
        embedding_path = precomputed_embeddings / f"{track_stem}.npy"
        rows.append(
            {
                "pair_id": f"song_describer:{row.caption_id}",
                "dataset": "song_describer",
                "split": official[str(int(track_id))],
                "text": clean_text(row.caption),
                "feature_path": feature_path.as_posix(),
                "clamp3_embedding_path": embedding_path.as_posix() if embedding_path.is_file() else "",
                "audio_path": audio_path.as_posix(),
                "valence": "",
                "arousal": "",
                "has_va": 0,
                "global_identity": f"mtg:{track_id}",
                "alignment_verified": 1,
            }
        )
    if len(rows) != 1106 or len({row["global_identity"] for row in rows}) != 706:
        raise ValueError("Song Describer published counts do not match")
    return rows


def audit(rows_by_name: dict[str, list[dict]]) -> dict:
    report = {"manifests": {}}
    for name, rows in rows_by_name.items():
        splits = {split: sum(row["split"] == split for row in rows) for split in ("train", "validation", "test")}
        identities = {}
        for row in rows:
            identities.setdefault(row["global_identity"], set()).add(row["split"])
        overlaps = {identity: sorted(values) for identity, values in identities.items() if len(values) > 1}
        if overlaps:
            raise ValueError(f"{name} has identity leakage: {list(overlaps.items())[:3]}")
        report["manifests"][name] = {
            "rows": len(rows),
            "unique_identities": len(identities),
            "splits": splits,
            "identity_overlap": 0,
        }
    all_identities = {}
    for name, rows in rows_by_name.items():
        for row in rows:
            all_identities.setdefault(row["global_identity"], {}).setdefault(row["split"], set()).add(name)
    cross_split = {
        identity: {split: sorted(names) for split, names in split_map.items()}
        for identity, split_map in all_identities.items()
        if len(split_map) > 1
    }
    if cross_split:
        raise ValueError(f"cross-manifest identity leakage: {list(cross_split.items())[:3]}")
    report["global_identity_audit"] = {
        "unique_identities": len(all_identities),
        "cross_split_overlap": 0,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "data.yaml")
    parser.add_argument("--datasets", nargs="+", choices=("text_emobank", "music_deam", "music_pmemo", "pairs_merge", "pairs_song_describer"))
    parser.add_argument("--overwrite", action="store_true", help="Replace existing manifests and their experiment hashes")
    args = parser.parse_args()
    config = load_yaml(args.config)
    paths = config["paths"]
    output = ROOT / paths["manifest_dir"]
    output.mkdir(parents=True, exist_ok=True)
    selected = args.datasets or ("text_emobank", "music_deam", "music_pmemo", "pairs_merge", "pairs_song_describer")
    existing = [output / f"{name}.csv" for name in selected if (output / f"{name}.csv").exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Existing manifests would change experiment hashes: {existing}. Pass --overwrite deliberately.")
    builders = {
        "text_emobank": lambda: emobank_manifest(ROOT / paths["emobank"]),
        "music_deam": lambda: music_va_manifest("deam", paths),
        "music_pmemo": lambda: music_va_manifest("pmemo", paths),
        "pairs_merge": lambda: merge_manifest(
            ROOT / paths["merge_root"], ROOT / "artifacts" / "features" / "mert_clamp3" / "merge"
        ),
        "pairs_song_describer": lambda: song_describer_manifest(
            ROOT / paths["song_describer_root"],
            Path(paths["mtg_audio"]),
            Path(paths["mtg_features"]),
            Path(paths["clamp3_precomputed_audio_embeddings"]),
            Path(paths["mtg_split_dir"]),
        ),
    }
    rows_by_name = {name: builders[name]() for name in (args.datasets or builders)}
    report = audit(rows_by_name)
    for name, rows in rows_by_name.items():
        write_manifest(rows, output / f"{name}.csv")
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
