"""Current contrastive and semantic-distillation objectives."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def _cosine_logits_fp32(text, music, scale):
    # Small final batches can overflow scaled FP16 similarity gradients.
    # Casting alone is insufficient: autocast would downcast the matmul again.
    with torch.amp.autocast(device_type=text.device.type, enabled=False):
        text = F.normalize(text.float(), dim=-1)
        music = F.normalize(music.float(), dim=-1)
        scale = torch.as_tensor(scale, dtype=torch.float32, device=text.device)
        return (text @ music.T) * scale


def multi_positive_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    dim: int = 1,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Softmax loss whose numerator sums over every valid positive.

    This avoids treating a second caption or a second valid tag for the same
    track as a false negative.
    """

    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("logits and positive_mask must share a [Q,K] shape")
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")
    logits = logits.float()
    positive_mask = positive_mask.bool()
    if candidate_mask is None:
        candidate_mask = torch.ones_like(positive_mask)
    elif candidate_mask.shape != logits.shape:
        raise ValueError("candidate_mask must match logits")
    else:
        candidate_mask = candidate_mask.bool()
    if (positive_mask & ~candidate_mask).any().item():
        raise ValueError("candidate_mask cannot remove a known positive")
    valid = positive_mask.any(dim=dim)
    if not valid.any().item():
        return logits.sum() * 0.0
    # Remove queries with no positive before logsumexp: an all -inf row has
    # undefined gradients even when excluded from the final mean afterward.
    if dim == 0:
        logits, positive_mask, candidate_mask = logits.T, positive_mask.T, candidate_mask.T
    logits, positive_mask, candidate_mask = logits[valid], positive_mask[valid], candidate_mask[valid]
    denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, -torch.inf), dim=1)
    numerator = torch.logsumexp(logits.masked_fill(~positive_mask, -torch.inf), dim=1)
    return (denominator - numerator).mean()

def symmetric_multi_positive_info_nce(
    text_embeddings: torch.Tensor,
    music_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    logit_scale: torch.Tensor | float,
) -> torch.Tensor:
    if text_embeddings.ndim != 2 or music_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if text_embeddings.shape[1] != music_embeddings.shape[1]:
        raise ValueError("text/music embedding dimensions differ")
    if positive_mask.shape != (len(text_embeddings), len(music_embeddings)):
        raise ValueError("positive mask does not match text/music batch sizes")
    scale = torch.as_tensor(logit_scale, dtype=torch.float32, device=text_embeddings.device)
    logits = _cosine_logits_fp32(text_embeddings, music_embeddings, scale.clamp(max=100.0))
    return 0.5 * (
        multi_positive_cross_entropy(logits, positive_mask, dim=1)
        + multi_positive_cross_entropy(logits, positive_mask, dim=0)
    )

def symmetric_similarity_distillation_loss(
    student_text: torch.Tensor,
    student_music: torch.Tensor,
    teacher_text: torch.Tensor,
    teacher_music: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Preserve a frozen teacher's cross-modal neighbourhood distribution.

    The teacher is the unmodified frozen CLaMP3 geometry.  This is a compact
    similarity-distillation regularizer inspired by hybrid contrastive
    distillation in CLIP-Refine; it is not claimed as an exact reproduction of
    that vision-language training recipe.
    """

    tensors = (student_text, student_music, teacher_text, teacher_music)
    if any(value.ndim != 2 for value in tensors):
        raise ValueError("student and teacher embeddings must be rank-2")
    if student_text.shape != teacher_text.shape or student_music.shape != teacher_music.shape:
        raise ValueError("student and teacher batches must match")
    if student_text.shape[1] != student_music.shape[1]:
        raise ValueError("text/music embedding dimensions differ")
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")

    scale = 1.0 / float(temperature)
    student_logits = _cosine_logits_fp32(student_text, student_music, scale)
    with torch.no_grad():
        teacher_logits = _cosine_logits_fp32(teacher_text, teacher_music, scale)
        teacher_t2m = torch.softmax(teacher_logits, dim=1)
        teacher_m2t = torch.softmax(teacher_logits.T, dim=1)
    text_loss = F.kl_div(
        F.log_softmax(student_logits, dim=1), teacher_t2m, reduction="batchmean"
    )
    music_loss = F.kl_div(
        F.log_softmax(student_logits.T, dim=1), teacher_m2t, reduction="batchmean"
    )
    return 0.5 * (text_loss + music_loss)

from torch import nn
from ..models.music_va import QUANTILE_LEVELS

def pinball_loss(quantiles: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean pinball loss across songs, both VA axes, and three quantiles."""
    if quantiles.ndim != 3 or quantiles.shape[1:] != (2, 3):
        raise ValueError(f"quantiles must have shape [N,2,3], got {tuple(quantiles.shape)}")
    if targets.shape != quantiles.shape[:2]:
        raise ValueError(f"targets must have shape [N,2], got {tuple(targets.shape)}")
    levels = quantiles.new_tensor(QUANTILE_LEVELS)
    residual = targets.unsqueeze(-1) - quantiles
    return torch.maximum(levels * residual, (levels - 1.0) * residual).mean()


class VALoss(nn.Module):
    """Axis-balanced Smooth-L1 with an optional batch-CCC penalty."""

    def __init__(self, beta: float = 0.1, ccc_weight: float = 0.0, eps: float = 1.0e-8) -> None:
        super().__init__()
        if beta < 0 or ccc_weight < 0 or eps <= 0:
            raise ValueError("invalid VA loss parameter")
        self.beta, self.ccc_weight, self.eps = beta, ccc_weight, eps

    def ccc(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prediction_mean, target_mean = predictions.mean(0), targets.mean(0)
        prediction_centered = predictions - prediction_mean
        target_centered = targets - target_mean
        covariance = (prediction_centered * target_centered).mean(0)
        denominator = (
            prediction_centered.square().mean(0)
            + target_centered.square().mean(0)
            + (prediction_mean - target_mean).square()
        )
        return (2.0 * covariance / denominator.clamp_min(self.eps)).clamp(-1.0, 1.0)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, return_components: bool = False):
        if predictions.shape != targets.shape or predictions.ndim != 2 or predictions.shape[1] != 2:
            raise ValueError(
                f"predictions and targets must both be [B,2], got {tuple(predictions.shape)}/{tuple(targets.shape)}"
            )
        predictions, targets = predictions.float(), targets.float()
        huber = F.smooth_l1_loss(predictions, targets, beta=self.beta, reduction="none").mean(0)
        ccc_penalty = 1.0 - self.ccc(predictions, targets)
        axis = huber + self.ccc_weight * ccc_penalty
        total = axis.mean()
        if not return_components:
            return total
        return {
            "total": total,
            "valence": axis[0],
            "arousal": axis[1],
            "huber": huber.mean(),
            "ccc_penalty": ccc_penalty.mean(),
        }
