"""Read original datasets and their supplied splits; build training batches."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


# MTG-Jamendo's public autotagging files already apply the dataset authors'
# tag_map.json.  The remaining values are canonical identifiers, not display
# text.  In particular, many multi-word identifiers are intentionally stored
# without spaces.  Keep the identifiers unchanged for supervision and map only
# their language-facing form here.
MTG_VALUE_DISPLAY: dict[str, str] = {
    # Genre identifiers.
    "70s": "1970s",
    "80s": "1980s",
    "90s": "1990s",
    "acidjazz": "acid jazz",
    "alternativerock": "alternative rock",
    "bossanova": "bossa nova",
    "chillout": "chill-out",
    "classicrock": "classic rock",
    "darkambient": "dark ambient",
    "darkwave": "dark wave",
    "deephouse": "deep house",
    "drumnbass": "drum and bass",
    "easylistening": "easy listening",
    "edm": "electronic dance music",
    "electropop": "electro-pop",
    "eurodance": "Eurodance",
    "hardrock": "hard rock",
    "hiphop": "hip-hop",
    "idm": "intelligent dance music",
    "instrumentalpop": "instrumental pop",
    "instrumentalrock": "instrumental rock",
    "jazzfusion": "jazz fusion",
    "newage": "new age",
    "newwave": "new wave",
    "popfolk": "pop folk",
    "poprock": "pop rock",
    "postrock": "post-rock",
    "punkrock": "punk rock",
    "rnb": "R&B",
    "rocknroll": "rock and roll",
    "singersongwriter": "singer-songwriter",
    "synthpop": "synth-pop",
    "triphop": "trip-hop",
    # Instrument identifiers.
    "acousticbassguitar": "acoustic bass guitar",
    "acousticguitar": "acoustic guitar",
    "classicalguitar": "classical guitar",
    "doublebass": "double bass",
    "drummachine": "drum machine",
    "electricguitar": "electric guitar",
    "electricpiano": "electric piano",
    "pipeorgan": "pipe organ",
}


# Context-dependent labels need more than whitespace restoration.  These
# phrases state the musical meaning explicitly so that a general text encoder
# does not interpret, for example, "house" as a building or "computer" as a
# generic device.
MTG_PROMPT_OVERRIDES: dict[str, str] = {
    "genre---choir": "choral music",
    "genre---club": "club music",
    "genre---house": "house music genre",
    "instrument---bass": "music featuring bass guitar",
    "instrument---beat": "music with a prominent beat",
    "instrument---brass": "music featuring brass instruments",
    "instrument---computer": "music featuring computer-generated sounds",
    "instrument---horn": "music featuring a brass horn",
    "instrument---orchestra": "music performed by an orchestra",
    "instrument---pad": "music featuring synthesizer pads",
    "instrument---rhodes": "music featuring Rhodes electric piano",
    "instrument---strings": "music featuring string instruments",
    "instrument---voice": "music featuring vocals",
    "mood/theme---action": "music for action scenes",
    "mood/theme---adventure": "adventure-themed music",
    "mood/theme---advertising": "music for advertising",
    "mood/theme---background": "background music",
    "mood/theme---children": "children's music",
    "mood/theme---commercial": "commercial music",
    "mood/theme---corporate": "corporate background music",
    "mood/theme---deep": "music with a deep mood",
    "mood/theme---documentary": "music for documentaries",
    "mood/theme---dream": "dreamy music",
    "mood/theme---emotional": "emotionally expressive music",
    "mood/theme---fast": "fast-paced music",
    "mood/theme---film": "film score music",
    "mood/theme---game": "video game music",
    "mood/theme---holiday": "holiday-themed music",
    "mood/theme---love": "love-themed music",
    "mood/theme---movie": "movie soundtrack music",
    "mood/theme---nature": "nature-themed music",
    "mood/theme---slow": "slow-tempo music",
    "mood/theme---space": "space-themed music",
    "mood/theme---sport": "music for sports",
    "mood/theme---summer": "summer-themed music",
    "mood/theme---trailer": "movie trailer music",
    "mood/theme---travel": "travel-themed music",
}


@dataclass(frozen=True)
class MTGTrack:
    track_id: str
    artist_id: str
    album_id: str
    relative_path: str
    duration: float
    tags: tuple[str, ...]
    split: str


def read_mtg_autotagging_tsv(path: str | Path, split: str) -> list[MTGTrack]:
    """Read MTG's variable-width TSV (one trailing column per tag)."""

    path = Path(path)
    rows: list[MTGTrack] = []
    with path.open("r", encoding="utf-8") as file:
        header = file.readline().rstrip("\n\r").split("\t")
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ValueError(f"unexpected MTG header in {path}: {header}")
        for line_number, line in enumerate(file, start=2):
            fields = line.rstrip("\n\r").split("\t")
            if len(fields) < 5:
                raise ValueError(f"invalid MTG row at {path}:{line_number}")
            track_id, artist_id, album_id, relative_path, duration, *tags = fields
            rows.append(
                MTGTrack(
                    track_id=track_id,
                    artist_id=artist_id,
                    album_id=album_id,
                    relative_path=relative_path,
                    duration=float(duration),
                    tags=tuple(sorted(set(tag for tag in tags if tag))),
                    split=str(split),
                )
            )
    return rows


def select_supported_labels(
    tracks: list[MTGTrack],
    prefixes: tuple[str, ...],
    *,
    min_train_tracks: int,
    min_train_artists: int,
) -> tuple[list[str], dict[str, dict[str, int]]]:
    if min_train_tracks <= 0 or min_train_artists <= 0:
        raise ValueError("label support thresholds must be positive")
    track_counts: Counter[str] = Counter()
    artists: dict[str, set[str]] = defaultdict(set)
    for row in tracks:
        if row.split != "train":
            continue
        for tag in row.tags:
            if tag.startswith(prefixes):
                track_counts[tag] += 1
                artists[tag].add(row.artist_id)
    labels = sorted(
        tag
        for tag, count in track_counts.items()
        if count >= min_train_tracks and len(artists[tag]) >= min_train_artists
    )
    support = {
        tag: {"train_tracks": int(track_counts[tag]), "train_artists": len(artists[tag])}
        for tag in labels
    }
    return labels, support


def split_mtg_tag(tag: str) -> tuple[str, str]:
    try:
        namespace, value = tag.split("---", 1)
    except ValueError as error:
        raise ValueError(f"invalid MTG tag: {tag}") from error
    if namespace not in {"genre", "instrument", "mood/theme"} or not value:
        raise ValueError(f"unsupported MTG tag: {tag}")
    return namespace, value


def tag_display_name(tag: str) -> str:
    """Turn an official MTG identifier into a readable musical term."""

    _, value = split_mtg_tag(tag)
    normalized = value.replace("_", " ").replace("-", " ").strip()
    return MTG_VALUE_DISPLAY.get(value, normalized)


def tag_prompt(tag: str) -> str:
    """Map an official MTG identifier to a canonical natural query."""

    namespace, _ = split_mtg_tag(tag)
    if tag in MTG_PROMPT_OVERRIDES:
        return MTG_PROMPT_OVERRIDES[tag]
    display = tag_display_name(tag)
    if namespace == "genre":
        return f"{display} music"
    if namespace == "instrument":
        return f"music featuring {display}"
    return f"{display} music"



import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from .train import check_identity_splits
from . import SCHEMA

DATASETS = ("text_emobank", "music_deam", "music_pmemo", "pairs_merge", "mtg_music", "mtg_tags", "pairs_song_describer")
SPLITS = ("train", "validation", "test")


def normalize_va(values, scale):
    low, high = map(float, scale)
    values = np.asarray(values, dtype=np.float32)
    if high <= low or not np.isfinite(values).all() or (values < low).any() or (values > high).any():
        raise ValueError(f"VA 标签不符合源尺度 {scale}")
    return (2 * (values - low) / (high - low) - 1).tolist()


def _record(identity, sample_id, dataset, split, *, text="", audio="", va=None, **extra):
    return dict(identity=identity, id=sample_id, dataset=dataset, split=split,
                text=" ".join(str(text).split()), audio_path=str(audio),
                va=va if va is not None else [0.0, 0.0], has_va=va is not None, **extra)


def load_raw(config, requested=None):
    """Read annotations directly. Preserve provided splits; never invent a split."""
    requested = list(requested or config["datasets"])
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(DATASETS):
        raise ValueError("数据集名称无效或重复")
    if "mtg_tags" in requested and "mtg_music" not in requested:
        requested.append("mtg_music")
    sources, result, missing, source_files = config["sources"], {}, {}, []
    if "text_emobank" in requested:
        source = sources["emobank"]
        path = Path(source["csv"])
        source_files.append(path)
        frame = pd.read_csv(path)
        names = {"train": "train", "dev": "validation", "test": "test"}
        result["text_emobank"] = [_record(f"emobank:{r.id}", str(r.id), "emobank", names[str(r.split)],
            text=r.text, va=normalize_va([r.V, r.A], source["source_scale"])) for r in frame.itertuples(index=False)]
    for name in ("deam", "pmemo"):
        key = f"music_{name}"
        if key not in requested:
            continue
        source = sources[name]
        path = Path(source["annotations"])
        source_files.append(path)
        by_id = {}
        for r in pd.read_csv(path).itertuples(index=False):
            identity = str(int(float(r.song_id)))
            if identity in by_id:
                raise ValueError(f"{name} 标签 ID 重复: {identity}")
            by_id[identity] = normalize_va([r.valence_mean, r.arousal_mean], source["source_scale"])
        splits = {}
        for split in SPLITS:
            path = Path(source["splits"]) / f"{split}.txt"
            source_files.append(path)
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                identity = str(int(float(line.strip())))
                if identity in splits:
                    raise ValueError(f"{name} 划分 ID 重复: {identity}")
                splits[identity] = split
        if set(splits) != set(by_id):
            raise ValueError(f"{name} 标签与划分不一致")
        audio = {p.stem: p for p in Path(source["audio"]).iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac"}}
        result[key] = [_record(f"{name}:{i}", f"{name}:{i}", name, splits[i],
                              audio=audio[i], va=by_id[i]) for i in sorted(by_id, key=int)]
    if "pairs_merge" in requested:
        root = Path(sources["merge"]["root"])
        path = root / "merge_bimodal_complete_av_values.csv"
        source_files.append(path)
        values = pd.read_csv(path)
        splits = {}
        for split, token in (("train", "train"), ("validation", "validate"), ("test", "test")):
            files = list((root / "tvt_dataframes/tvt_70_15_15").glob(f"*_{token}_*.csv"))
            if len(files) != 1:
                raise ValueError(f"MERGE {split}: 需要唯一的划分 CSV，实际为 {files}")
            source_files.extend(files)
            for identity in pd.read_csv(files[0])["Song"].astype(str):
                if identity in splits:
                    raise ValueError(f"MERGE 划分重复: {identity}")
                splits[identity] = split
        audio = {p.stem: p for p in (root / "audio").rglob("*.mp3")}
        lyrics = {p.stem: p for p in (root / "lyrics").rglob("*.txt")}
        rows = []
        for row in values.itertuples(index=False):
            a, t = str(row.Audio_Song), str(row.Lyric_Song)
            source_files.append(lyrics[t])
            rows.append(_record(f"merge:{a}", f"merge:{a}:{t}", "merge", splits[a],
                                text=lyrics[t].read_text(encoding="utf-8"), audio=audio[a]))
        result["pairs_merge"] = rows
    need_mtg = any(k in requested for k in ("mtg_music", "mtg_tags", "pairs_song_describer"))
    if need_mtg:
        tracks = []
        for split in SPLITS:
            path = Path(sources["mtg"]["splits"]) / f"autotagging-{split}.tsv"
            source_files.append(path)
            tracks.extend(read_mtg_autotagging_tsv(path, split))
        lookup = {str(int(t.track_id.removeprefix("track_"))): t for t in tracks}
        if len(lookup) != len(tracks):
            raise ValueError("MTG 官方划分存在重复歌曲 ID")
        if "mtg_music" in requested:
            rows, absent = [], []
            for t in sorted(tracks, key=lambda t: t.track_id):
                path = Path(sources["mtg"]["audio"]) / t.relative_path
                if not path.is_file():
                    absent.append(t.track_id)
                    continue
                rows.append(_record(f"mtg:{int(t.track_id.removeprefix('track_'))}", t.track_id, "mtg", t.split,
                            audio=path, artist_id=t.artist_id, album_id=t.album_id, tags=t.tags))
            if absent and sources["mtg"]["missing_audio"] != "skip_and_report":
                raise FileNotFoundError(f"MTG 缺少 {len(absent)} 首原始音频")
            result["mtg_music"], missing["mtg_music"] = rows, absent
        if "pairs_song_describer" in requested:
            path = Path(sources["song_describer"]["csv"])
            source_files.append(path)
            rows = []
            for r in pd.read_csv(path).itertuples(index=False):
                t = lookup[str(int(r.track_id))]
                rows.append(_record(f"mtg:{int(r.track_id)}", f"song_describer:{r.caption_id}", "song_describer",
                    t.split, text=r.caption, audio=Path(sources["mtg"]["audio"]) / t.relative_path))
            result["pairs_song_describer"] = rows
    all_ids, all_splits = [], []
    for key, rows in result.items():
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"{key} 为空或样本 ID 重复")
        for r in rows:
            if r["audio_path"]:
                stat = Path(r["audio_path"]).stat()
                r["audio_size"], r["audio_mtime_ns"] = stat.st_size, stat.st_mtime_ns
            if key.startswith(("text_", "pairs_")) and not r["text"]:
                raise ValueError(f"空文本: {r['id']}")
        all_ids.extend(r["identity"] for r in rows)
        all_splits.extend(r["split"] for r in rows)
    check_identity_splits(np.asarray(all_ids), np.asarray(all_splits), "原始数据")
    from .train import sha256
    report = {"datasets": {k: {"rows": len(v), "splits": {s: sum(r["split"] == s for r in v) for s in SPLITS}}
                           for k, v in result.items()}, "missing_audio": missing,
              "source_files": {str(p): sha256(p) for p in sorted(set(source_files))}}
    return result, report


def metadata_arrays(rows):
    return dict(ids=np.asarray([r["id"] for r in rows]),
                global_identities=np.asarray([r["identity"] for r in rows]),
                datasets=np.asarray([r["dataset"] for r in rows]),
                splits=np.asarray([r["split"] for r in rows]),
                va_targets=np.asarray([r["va"] for r in rows], dtype=np.float32),
                has_va=np.asarray([r["has_va"] for r in rows], dtype=bool))


def read_features(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {k: archive[k] for k in archive.files}
    if str(result.get("schema", "")) != SCHEMA:
        raise ValueError(f"不接受旧项目或无来源的特征: {path}")
    for k, v in result.items():
        if k.endswith("embeddings"):
            if v.ndim != 2 or v.shape[1] != 768 or not np.isfinite(v).all():
                raise ValueError(f"{path}: {k} 不是有效的 [N,768] 特征")
            if not np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4):
                raise ValueError(f"{path}: {k} 未归一化")
    if "global_identities" in result:
        check_identity_splits(result["global_identities"], result["splits"], str(path))
    return result

def concatenate_npz(paths: list[Path], keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    parts = {key: [] for key in keys}
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            for key in keys:
                parts[key].append(data[key])
    return {key: np.concatenate(values) for key, values in parts.items()}


def load_pair_data(paths: list[Path]) -> dict[str, np.ndarray]:
    return concatenate_npz(
        paths,
        (
            "text_embeddings",
            "audio_embeddings",
            "splits",
            "datasets",
            "global_identities",
        ),
    )




def make_pair_loader(
    data: dict[str, np.ndarray], split: str, batch_size: int, seed: int
) -> DataLoader:
    mask = data["splits"].astype(str) == split
    identities = data["global_identities"][mask].astype(str)
    _, identity_ids = np.unique(identities, return_inverse=True)
    domains = data["datasets"][mask].astype(str)
    _, domain_ids = np.unique(domains, return_inverse=True)
    dataset = TensorDataset(
        torch.from_numpy(data["text_embeddings"][mask].astype(np.float32)),
        torch.from_numpy(data["audio_embeddings"][mask].astype(np.float32)),
        torch.from_numpy(identity_ids.astype(np.int64)),
        torch.from_numpy(domain_ids.astype(np.int64)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      generator=torch.Generator().manual_seed(seed), num_workers=0,
                      drop_last=False)



def make_mtg_loader(
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    split: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    mask = mtg["splits"].astype(str) == split
    embeddings = music_index[mtg["index_positions"][mask]].astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy(mtg["preference_targets"][mask].astype(bool)),
        torch.from_numpy(mtg["affect_targets"][mask].astype(bool)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        drop_last=False,
    )


