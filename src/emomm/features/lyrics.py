"""Fixed train-only MERGE lyric views for the selected semantic training route."""
from __future__ import annotations

import re

import numpy as np


VIEW_METHOD = "merge_lyric_views_semantic_preservation_v1"
# The selected route always uses these views; there is no augmentation switch.
VIEW_PARAMETERS = dict(window_words=96, candidate_count=5, selected_views=2,
                       min_lyrics_words=64, audio_weight=0.5)


def lyric_candidates(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    words = normalized.split()
    count = len(words)
    if count < VIEW_PARAMETERS["min_lyrics_words"]:
        return []
    width = VIEW_PARAMETERS["window_words"]
    span = width if count > width else max(2, min(count - 1, round(0.75 * count)))
    last_start = count - span
    starts = [round(i * last_start / (VIEW_PARAMETERS["candidate_count"] - 1))
              for i in range(VIEW_PARAMETERS["candidate_count"])]
    return list(dict.fromkeys(excerpt for start in starts
                             if (excerpt := " ".join(words[start:start + span]))
                             and excerpt != normalized))


def build_lyric_views(rows, base, encode):
    """Encode fresh excerpts and select two by frozen audio/full-lyric consistency.

    Validation/test texts are never sent to the excerpt encoder. Views retain
    the source song identity, so the pair loader treats same-song views as
    positives and can exclude all views of a cross-split duplicate together.
    """
    ids = base["ids"].astype(str)
    lookup = {str(row["id"]): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != set(ids):
        raise ValueError("歌词原始记录与本批次 MERGE 特征的样本 ID 不一致")
    records = []
    train_rows = 0
    for source, identity in enumerate(ids):
        row = lookup[identity]
        if str(base["global_identities"][source]) != str(row["identity"]):
            raise ValueError(f"歌词原始记录与 MERGE 歌曲身份不一致: {identity}")
        if str(base["splits"][source]) != str(row["split"]):
            raise ValueError(f"歌词原始记录与 MERGE 划分不一致: {identity}")
        if str(base["splits"][source]) != "train":
            continue
        train_rows += 1
        records.extend((source, text) for text in lyric_candidates(row["text"]))
    if not records:
        raise ValueError("MERGE 训练集没有可生成片段的歌词，不能运行固定增强方案")
    vectors = np.asarray(encode([text for _, text in records]), dtype=np.float32)
    if vectors.shape != (len(records), 768) or not np.isfinite(vectors).all():
        raise ValueError("歌词片段编码无效")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
        raise ValueError("歌词片段编码未归一化")
    by_source = {}
    for position, (source, _) in enumerate(records):
        by_source.setdefault(source, []).append(position)
    selected, ranks, scores = [], [], []
    for source, positions in by_source.items():
        candidates = vectors[positions]
        values = (0.5 * (candidates @ base["audio_embeddings"][source])
                  + 0.5 * (candidates @ base["text_embeddings"][source]))
        order = sorted(range(len(positions)), key=lambda i: (-float(values[i]), i))
        for rank, local in enumerate(order[:VIEW_PARAMETERS["selected_views"]], start=1):
            selected.append(positions[local])
            ranks.append(rank)
            scores.append(float(values[local]))
    source_rows = np.asarray([records[i][0] for i in selected], dtype=np.int64)
    payload = {key: base[key][source_rows].copy()
               for key in ("audio_embeddings", "audio_content_hashes", "global_identities",
                           "splits", "datasets", "va_targets", "has_va")}
    payload.update(
        text_embeddings=vectors[selected],
        ids=np.asarray([f"{ids[source]}:lyric_view_rank_{rank}"
                        for source, rank in zip(source_rows, ranks, strict=True)]),
        source_pair_ids=ids[source_rows],
        view_rank=np.asarray(ranks, dtype=np.int64),
        texts=np.asarray([records[i][1] for i in selected]),
        selection_scores=np.asarray(scores, dtype=np.float32),
        view_method=np.asarray(VIEW_METHOD),
    )
    return payload, dict(train_rows=train_rows, eligible_rows=len(by_source),
                         skipped_short_rows=train_rows - len(by_source),
                         candidate_views=len(records), selected_views=len(selected))
