"""Public MTG-Jamendo supervision utilities for Ours."""

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


MTG_SEMANTICALLY_AMBIGUOUS: frozenset[str] = frozenset(
    {
        "genre---alternative",
        "genre---atmospheric",
        "genre---contemporary",
        "genre---fusion",
        "genre---hard",
        "genre---house",
        "genre---minimal",
        "genre---progressive",
        "instrument---bass",
        "instrument---beat",
        "instrument---computer",
        "instrument---horn",
        "instrument---pad",
        "instrument---voice",
        "mood/theme---commercial",
        "mood/theme---cool",
        "mood/theme---deep",
        "mood/theme---emotional",
        "mood/theme---heavy",
        "mood/theme---positive",
        "mood/theme---powerful",
        "mood/theme---soft",
    }
)


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


def legacy_tag_prompt(tag: str) -> str:
    """Return the language-facing form used before the mapping audit."""

    namespace, value = split_mtg_tag(tag)
    value = value.replace("_", " ").replace("-", " ").strip()
    if namespace == "genre":
        return f"{value} music"
    if namespace == "instrument":
        return f"music featuring {value}"
    if namespace == "mood/theme":
        return f"{value} music"
    raise ValueError(f"unsupported MTG namespace: {namespace}")


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


def tag_query_views(tag: str) -> tuple[str, ...]:
    """Return deterministic query paraphrases for alignment training."""

    namespace, _ = split_mtg_tag(tag)
    display = tag_display_name(tag)
    canonical = tag_prompt(tag)
    if namespace == "genre":
        candidates = (canonical, f"music in the {display} genre", f"a {display} track")
    elif namespace == "instrument":
        candidates = (canonical, f"music with {display}", f"a track featuring {display}")
    else:
        candidates = (
            canonical,
            f"music associated with {display}",
            f"a {display} music track",
        )
    return tuple(dict.fromkeys(candidate.strip() for candidate in candidates if candidate.strip()))
