"""Deterministic, provenance-friendly text views for semantic retrieval training."""

from __future__ import annotations

import re


_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Collapse manifest whitespace without otherwise rewriting copyrighted text."""

    return _WHITESPACE.sub(" ", str(text)).strip()


def lyric_window_candidates(
    text: str,
    *,
    window_words: int = 96,
    candidate_count: int = 5,
    min_lyrics_words: int = 64,
) -> list[str]:
    """Return distinct, evenly spaced lyric excerpts for train-only augmentation.

    Long lyrics use fixed-size windows.  Lyrics shorter than ``window_words``
    use 75% windows so that the view is not an exact duplicate of the original
    full-text embedding.  Very short lyrics are skipped because fragmenting
    them would make the positive audio/text relation too ambiguous.
    """

    if window_words <= 0:
        raise ValueError("window_words must be positive")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    if min_lyrics_words < 2:
        raise ValueError("min_lyrics_words must be at least two")

    normalized = normalize_text(text)
    words = normalized.split()
    count = len(words)
    if count < min_lyrics_words:
        return []

    if count > window_words:
        span = window_words
    else:
        span = max(2, min(count - 1, round(0.75 * count)))
    max_start = count - span
    if max_start <= 0:
        return []

    if candidate_count == 1:
        starts = [max_start // 2]
    else:
        starts = [
            round(index * max_start / (candidate_count - 1))
            for index in range(candidate_count)
        ]

    candidates: list[str] = []
    seen: set[str] = set()
    for start in starts:
        excerpt = " ".join(words[start : start + span])
        if excerpt and excerpt != normalized and excerpt not in seen:
            seen.add(excerpt)
            candidates.append(excerpt)
    return candidates


def rank_view_candidates(
    candidate_to_audio_similarity: list[float],
    candidate_to_full_text_similarity: list[float],
    *,
    audio_weight: float = 0.5,
) -> list[int]:
    """Rank views by frozen-backbone cross-modal and semantic consistency.

    The tie break is the original candidate order, which makes regeneration
    byte-for-byte deterministic for fixed frozen embeddings.
    """

    if len(candidate_to_audio_similarity) != len(candidate_to_full_text_similarity):
        raise ValueError("candidate similarity lists must have equal length")
    if not 0.0 <= audio_weight <= 1.0:
        raise ValueError("audio_weight must be in [0, 1]")
    text_weight = 1.0 - audio_weight
    scores = [
        audio_weight * float(audio) + text_weight * float(text)
        for audio, text in zip(
            candidate_to_audio_similarity,
            candidate_to_full_text_similarity,
            strict=True,
        )
    ]
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))
