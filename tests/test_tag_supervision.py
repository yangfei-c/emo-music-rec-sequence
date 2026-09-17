from __future__ import annotations

from pathlib import Path

from emomm_sequence.data.tags import (
    legacy_tag_prompt,
    read_mtg_autotagging_tsv,
    select_supported_labels,
    tag_prompt,
    tag_query_views,
)


def test_variable_width_mtg_tsv_and_support_filter(tmp_path: Path) -> None:
    path = tmp_path / "autotagging-train.tsv"
    path.write_text(
        "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"
        "track_1\tartist_1\talbum_1\t1.mp3\t10\tgenre---rock\tinstrument---piano\n"
        "track_2\tartist_2\talbum_2\t2.mp3\t11\tgenre---rock\tmood/theme---happy\n",
        encoding="utf-8",
    )
    tracks = read_mtg_autotagging_tsv(path, "train")
    labels, support = select_supported_labels(
        tracks,
        ("genre---", "instrument---"),
        min_train_tracks=2,
        min_train_artists=2,
    )
    assert labels == ["genre---rock"]
    assert support["genre---rock"] == {"train_tracks": 2, "train_artists": 2}


def test_tag_prompts_are_deterministic() -> None:
    assert tag_prompt("genre---post-rock") == "post rock music"
    assert tag_prompt("genre---acidjazz") == "acid jazz music"
    assert tag_prompt("genre---rnb") == "R&B music"
    assert tag_prompt("instrument---electricguitar") == "music featuring electric guitar"
    assert tag_prompt("instrument---voice") == "music featuring vocals"
    assert tag_prompt("mood/theme---calm") == "calm music"


def test_legacy_prompt_is_preserved_for_controlled_ablation() -> None:
    assert legacy_tag_prompt("genre---acidjazz") == "acidjazz music"
    assert legacy_tag_prompt("instrument---electricguitar") == "music featuring electricguitar"


def test_query_views_are_unique_and_keep_canonical_first() -> None:
    views = tag_query_views("instrument---piano")
    assert views[0] == "music featuring piano"
    assert len(views) == len(set(views))
    assert "music with piano" in views
