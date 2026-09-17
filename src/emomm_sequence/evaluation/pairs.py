"""Group-aware text--music retrieval metrics for multi-caption datasets."""

from __future__ import annotations

import numpy as np


def _summary(ranks: np.ndarray, ks: tuple[int, ...]) -> dict[str, float]:
    ranks = np.asarray(ranks, dtype=np.int64)
    if ranks.ndim != 1 or len(ranks) == 0 or (ranks < 1).any():
        raise ValueError("retrieval ranks must be a non-empty positive vector")
    result = {
        "queries": int(len(ranks)),
        "mrr": float(np.mean(1.0 / ranks)),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    result.update({f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks})
    return result


def grouped_retrieval_metrics(
    text_embeddings: np.ndarray,
    music_embeddings: np.ndarray,
    identities: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    text = np.asarray(text_embeddings, dtype=np.float64)
    music = np.asarray(music_embeddings, dtype=np.float64)
    identities = np.asarray(identities).astype(str)
    if text.shape != music.shape or text.ndim != 2 or len(text) != len(identities):
        raise ValueError("paired embeddings/identities have inconsistent shapes")
    text /= np.linalg.norm(text, axis=1, keepdims=True).clip(min=1.0e-12)
    music /= np.linalg.norm(music, axis=1, keepdims=True).clip(min=1.0e-12)
    unique, first = np.unique(identities, return_index=True)
    order = np.argsort(first)
    unique, first = unique[order], first[order]
    unique_music = music[first]
    identity_to_music = {identity: index for index, identity in enumerate(unique)}

    text_scores = text @ unique_music.T
    correct = np.asarray([identity_to_music[value] for value in identities])
    correct_scores = text_scores[np.arange(len(text)), correct]
    text_ranks = 1 + (text_scores > correct_scores[:, None]).sum(axis=1)

    music_scores = unique_music @ text.T
    music_ranks = []
    for row, identity in enumerate(unique):
        relevant = identities == identity
        best = music_scores[row, relevant].max()
        music_ranks.append(1 + int((music_scores[row] > best).sum()))
    music_ranks = np.asarray(music_ranks)
    positive = np.sum(text * music, axis=1)
    return {
        "captions": int(len(text)),
        "tracks": int(len(unique)),
        "text_to_music": _summary(text_ranks, ks),
        "music_to_text": _summary(music_ranks, ks),
        "mean_positive_cosine": float(positive.mean()),
        "std_positive_cosine": float(positive.std(ddof=1)) if len(positive) > 1 else 0.0,
    }


def grouped_retrieval_metrics_from_scores(
    scores: np.ndarray,
    text_identities: np.ndarray,
    music_identities: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict:
    """Evaluate a caption-by-track score matrix with grouped positives.

    Unlike :func:`grouped_retrieval_metrics`, this accepts scores from a
    non-factorizable reranker such as query-conditioned segment pooling.
    Candidate music identities must be unique; caption identities may repeat.
    """

    values = np.asarray(scores, dtype=np.float64)
    text_ids = np.asarray(text_identities).astype(str)
    music_ids = np.asarray(music_identities).astype(str)
    if values.ndim != 2 or values.shape != (len(text_ids), len(music_ids)):
        raise ValueError("scores and identity vectors have inconsistent shapes")
    if not np.isfinite(values).all():
        raise ValueError("retrieval scores contain NaN/Inf")
    if len(set(music_ids.tolist())) != len(music_ids):
        raise ValueError("music identities must be unique")
    music_index = {identity: index for index, identity in enumerate(music_ids)}
    try:
        correct = np.asarray([music_index[value] for value in text_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"caption identity has no candidate track: {error.args[0]}") from error

    correct_scores = values[np.arange(len(text_ids)), correct]
    text_ranks = 1 + (values > correct_scores[:, None]).sum(axis=1)
    music_ranks = []
    for column, identity in enumerate(music_ids):
        relevant = text_ids == identity
        if not relevant.any():
            raise ValueError(f"candidate track has no positive caption: {identity}")
        best = values[relevant, column].max()
        music_ranks.append(1 + int((values[:, column] > best).sum()))
    positive = correct_scores
    return {
        "captions": int(len(text_ids)),
        "tracks": int(len(music_ids)),
        "text_to_music": _summary(text_ranks, ks),
        "music_to_text": _summary(np.asarray(music_ranks), ks),
        "mean_positive_score": float(positive.mean()),
        "std_positive_score": float(positive.std(ddof=1)) if len(positive) > 1 else 0.0,
    }
