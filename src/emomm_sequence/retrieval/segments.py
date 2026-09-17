"""Query-conditioned segment matching for dense text-to-music reranking."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def query_conditioned_segment_similarity(
    queries: torch.Tensor,
    segments: torch.Tensor,
    segment_mask: torch.Tensor,
    *,
    temperature: float = 0.07,
    top_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool per-window cosine scores using query-conditioned attention.

    Args:
        queries: Text embeddings with shape ``[Q, D]``.
        segments: Music-window embeddings with shape ``[M, T, D]``.
        segment_mask: True for real windows, shape ``[M, T]``.
        temperature: Softmax temperature over windows.
        top_k: If given, attend only to the best ``k`` valid windows per
            query/track pair.  This prevents unrelated sections of a long song
            from diluting a locally matching passage.

    Returns:
        A ``[Q, M]`` local similarity matrix and the corresponding
        ``[Q, M, T]`` attention weights.
    """

    if queries.ndim != 2 or segments.ndim != 3:
        raise ValueError("queries/segments must have shape [Q,D]/[M,T,D]")
    if queries.shape[1] != segments.shape[2]:
        raise ValueError("query and segment embedding dimensions differ")
    if segment_mask.shape != segments.shape[:2]:
        raise ValueError("segment mask must have shape [M,T]")
    if temperature <= 0:
        raise ValueError("segment temperature must be positive")
    mask = segment_mask.bool()
    if (~mask.any(dim=1)).any().item():
        raise ValueError("every track must contain at least one valid segment")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive or None")

    normalized_queries = F.normalize(queries, dim=-1)
    normalized_segments = F.normalize(segments, dim=-1)
    similarities = torch.einsum("qd,mtd->qmt", normalized_queries, normalized_segments)
    candidate_mask = mask.unsqueeze(0).expand(queries.shape[0], -1, -1)
    if top_k is not None and top_k < segments.shape[1]:
        requested = min(int(top_k), segments.shape[1])
        masked_scores = similarities.masked_fill(~candidate_mask, -torch.inf)
        top_indices = masked_scores.topk(requested, dim=-1).indices
        top_mask = torch.zeros_like(candidate_mask)
        top_mask.scatter_(-1, top_indices, True)
        candidate_mask = candidate_mask & top_mask

    logits = similarities / float(temperature)
    attention = torch.softmax(logits.masked_fill(~candidate_mask, -torch.inf), dim=-1)
    attention = attention.masked_fill(~candidate_mask, 0.0)
    scores = (attention * similarities).sum(dim=-1)
    return scores, attention
