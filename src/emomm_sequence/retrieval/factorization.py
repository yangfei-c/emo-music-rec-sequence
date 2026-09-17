"""Operations used to train affect/preference factors without manual pairs."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def mean_positive_label_embedding(
    targets: torch.Tensor,
    label_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean all positive label embeddings for each track.

    Returns normalized embeddings and a validity mask for rows with at least one
    positive label.
    """

    if targets.ndim != 2 or label_embeddings.ndim != 2:
        raise ValueError("targets and label embeddings must be rank-2")
    if targets.shape[1] != label_embeddings.shape[0]:
        raise ValueError("target vocabulary and label embeddings differ")
    weights = targets.to(label_embeddings.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    valid = counts[:, 0] > 0
    pooled = weights @ label_embeddings
    pooled = pooled / counts.clamp_min(1.0)
    return F.normalize(pooled, dim=-1), valid


def counterfactual_masks(
    preference_targets: torch.Tensor,
    affect_targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the two controlled hard-negative sets within a track batch.

    * affect negatives share a preference but have no affect label in common;
    * preference negatives share affect but have no preference label in common.
    """

    if preference_targets.ndim != 2 or affect_targets.ndim != 2:
        raise ValueError("factor targets must be rank-2")
    if preference_targets.shape[0] != affect_targets.shape[0]:
        raise ValueError("factor target batches have different sizes")
    preference = preference_targets.bool()
    affect = affect_targets.bool()
    shared_preference = (preference.float() @ preference.float().T) > 0
    shared_affect = (affect.float() @ affect.float().T) > 0
    has_preference = preference.any(dim=1)
    has_affect = affect.any(dim=1)
    valid_joint = has_preference & has_affect
    pair_valid = valid_joint[:, None] & valid_joint[None, :]
    identity = torch.eye(len(preference), dtype=torch.bool, device=preference.device)
    affect_negative = pair_valid & shared_preference & ~shared_affect & ~identity
    preference_negative = pair_valid & shared_affect & ~shared_preference & ~identity
    return affect_negative, preference_negative

