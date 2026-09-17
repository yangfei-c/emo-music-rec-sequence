"""Group-aware, long-tail, and counterfactual objectives for Ours."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    probability_margin: float = 0.05,
    reduction: str = "mean",
    detach_focal_weight: bool = True,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Asymmetric Loss (ASL) for independent multi-label logits.

    This follows Ridnik et al., ICCV 2021: positive and negative examples use
    different focusing exponents, and the negative probability is shifted by
    ``probability_margin``.  ``mean_samples_sum_labels`` matches the common
    multi-label convention of summing labels for each example and then
    averaging examples; it is useful when positives are extremely sparse.

    Reference implementation:
    https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
    """

    if logits.shape != targets.shape or logits.ndim < 1:
        raise ValueError("logits and targets must share a non-empty shape")
    if gamma_neg < 0 or gamma_pos < 0:
        raise ValueError("ASL focusing exponents must be non-negative")
    if not 0.0 <= probability_margin < 1.0:
        raise ValueError("ASL probability margin must be in [0, 1)")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if reduction not in {"none", "mean", "sum", "mean_samples_sum_labels"}:
        raise ValueError(f"unsupported ASL reduction: {reduction}")

    target = targets.to(dtype=logits.dtype)
    if ((target < 0) | (target > 1)).any().item():
        raise ValueError("ASL targets must be binary values in [0, 1]")

    positive_probability = torch.sigmoid(logits)
    negative_probability = 1.0 - positive_probability
    if probability_margin > 0:
        negative_probability = (negative_probability + probability_margin).clamp(max=1.0)

    log_likelihood = (
        target * torch.log(positive_probability.clamp_min(eps))
        + (1.0 - target) * torch.log(negative_probability.clamp_min(eps))
    )
    if gamma_neg > 0 or gamma_pos > 0:
        correct_probability = (
            target * positive_probability + (1.0 - target) * negative_probability
        )
        exponent = target * float(gamma_pos) + (1.0 - target) * float(gamma_neg)
        focal_weight = (1.0 - correct_probability).pow(exponent)
        if detach_focal_weight:
            focal_weight = focal_weight.detach()
        log_likelihood = log_likelihood * focal_weight

    loss = -log_likelihood
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean_samples_sum_labels":
        if loss.ndim < 2:
            raise ValueError("mean_samples_sum_labels requires [samples, labels, ...]")
        return loss.flatten(start_dim=1).sum(dim=1).mean()
    return loss.mean()


def distribution_balanced_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_frequency: torch.Tensor,
    *,
    train_samples: int,
    map_alpha: float = 0.1,
    map_beta: float = 10.0,
    map_gamma: float = 0.2,
    negative_scale: float = 2.0,
    initial_bias_scale: float = 0.05,
    focal_gamma: float = 2.0,
    focal_balance: float = 2.0,
    reduction: str = "mean",
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Distribution-Balanced Loss for long-tailed multi-label targets.

    This implements the re-balanced weighting, negative-tolerant logit
    regularization, and focal modulation used by Wu et al. (ECCV 2020).  The
    expected input is ``[samples, labels]``.  As in the paper's class-aware
    training setup, every included sample must have at least one positive
    label; callers should explicitly mask unlabelled samples rather than
    inventing a re-balancing weight for them.

    Reference implementation:
    https://github.com/wutong16/DistributionBalancedLoss
    """

    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("DB logits and targets must share a [samples, labels] shape")
    if class_frequency.shape != (logits.shape[1],):
        raise ValueError("class_frequency must contain one value per label")
    if train_samples <= 0 or eps <= 0:
        raise ValueError("train_samples and eps must be positive")
    if map_alpha < 0 or map_beta < 0 or negative_scale <= 0:
        raise ValueError("invalid Distribution-Balanced Loss parameters")
    if focal_gamma < 0 or focal_balance < 0:
        raise ValueError("focal parameters must be non-negative")
    if reduction not in {"none", "mean", "sum", "mean_samples_sum_labels"}:
        raise ValueError(f"unsupported DB reduction: {reduction}")

    target = targets.to(dtype=logits.dtype)
    if ((target < 0) | (target > 1)).any().item():
        raise ValueError("DB targets must be binary values in [0, 1]")
    if not target.bool().any(dim=1).all().item():
        raise ValueError("DB loss requires at least one positive label per sample")
    frequency = class_frequency.to(device=logits.device, dtype=logits.dtype)
    if (frequency <= 0).any().item() or (frequency >= train_samples).any().item():
        raise ValueError("class frequencies must be in (0, train_samples)")

    inverse_frequency = frequency.reciprocal()
    repeat_rate = (target * inverse_frequency.unsqueeze(0)).sum(dim=1, keepdim=True)
    raw_weight = inverse_frequency.unsqueeze(0) / repeat_rate.clamp_min(eps)
    weight = torch.sigmoid(float(map_beta) * (raw_weight - float(map_gamma)))
    weight = weight + float(map_alpha)

    prevalence = (frequency / float(train_samples)).clamp(eps, 1.0 - eps)
    initial_bias = torch.logit(prevalence) * float(initial_bias_scale)
    initial_bias = initial_bias / float(negative_scale)
    regularized_logits = logits + initial_bias.unsqueeze(0)
    regularized_logits = (
        target * regularized_logits
        + (1.0 - target) * regularized_logits * float(negative_scale)
    )
    weight = target * weight + (1.0 - target) * weight / float(negative_scale)

    unweighted = F.binary_cross_entropy_with_logits(
        regularized_logits, target, reduction="none"
    )
    correct_probability = torch.exp(-unweighted)
    loss = (
        float(focal_balance)
        * (1.0 - correct_probability).pow(float(focal_gamma))
        * unweighted
        * weight
    )
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean_samples_sum_labels":
        return loss.sum(dim=1).mean()
    return loss.mean()


def distributionally_robust_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pair_scale: float = 1.0,
    negative_scale: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Class-wise Distributionally Robust loss from Lin et al., ECCV 2024.

    For every label, the first term is an LSEP-style log-sum-exp over all
    positive/negative track pairs in the batch.  The second term constrains
    over-confident negatives.  Computing it class-wise follows the authors'
    official implementation and avoids indiscriminately coupling unrelated
    labels inside one track.

    Reference implementation:
    https://github.com/Kunmonkey/DR-Loss/blob/main/mllt/models/losses/dr_loss.py
    """

    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("DR logits and targets must share a [samples, labels] shape")
    if pair_scale <= 0 or negative_scale <= 0:
        raise ValueError("DR scales must be positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"unsupported DR reduction: {reduction}")
    target = targets.to(dtype=logits.dtype)
    if ((target < 0) | (target > 1)).any().item():
        raise ValueError("DR targets must be binary values in [0, 1]")

    positive_scores = (-logits).masked_fill(~target.bool(), -torch.inf)
    negative_scores = logits.masked_fill(target.bool(), -torch.inf)
    positive_term = torch.logsumexp(float(pair_scale) * positive_scores, dim=0)
    negative_term = torch.logsumexp(float(pair_scale) * negative_scores, dim=0)
    constraint_term = torch.logsumexp(
        float(negative_scale) * negative_scores, dim=0
    )
    zero = torch.zeros_like(positive_term)
    loss = torch.logsumexp(
        torch.stack((zero, positive_term + negative_term, constraint_term), dim=0),
        dim=0,
    )
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def asymmetric_probability_gap(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ASL's positive confidence, negative confidence, and their gap."""

    if logits.shape != targets.shape:
        raise ValueError("logits and targets must share a shape")
    mask = targets.bool()
    if not mask.any().item() or mask.all().item():
        raise ValueError("adaptive ASL needs both positive and negative targets")
    probabilities = torch.sigmoid(logits.detach())
    positive_confidence = probabilities[mask].mean()
    negative_confidence = (1.0 - probabilities[~mask]).mean()
    return (
        positive_confidence,
        negative_confidence,
        positive_confidence - negative_confidence,
    )


class AdaptiveAsymmetryController:
    """Batch-wise controller for the adaptive ASL negative exponent.

    Ridnik et al. write ``gamma_- += lambda * (gap - target)``.  Here
    ``step_size`` is the positive magnitude ``-lambda`` so that gamma grows
    when the observed positive-minus-negative confidence gap is too small.
    The clamp is an explicit numerical safeguard absent from the paper.
    """

    def __init__(
        self,
        *,
        initial_gamma_neg: float = 0.0,
        target_gap: float = 0.1,
        step_size: float = 0.01,
        min_gamma_neg: float = 0.0,
        max_gamma_neg: float = 20.0,
    ) -> None:
        if initial_gamma_neg < 0 or step_size < 0:
            raise ValueError("adaptive ASL gamma and step size must be non-negative")
        if min_gamma_neg < 0 or max_gamma_neg < min_gamma_neg:
            raise ValueError("invalid adaptive ASL gamma bounds")
        self.gamma_neg = float(initial_gamma_neg)
        self.target_gap = float(target_gap)
        self.step_size = float(step_size)
        self.min_gamma_neg = float(min_gamma_neg)
        self.max_gamma_neg = float(max_gamma_neg)
        self.last_positive_confidence = float("nan")
        self.last_negative_confidence = float("nan")
        self.last_gap = float("nan")

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> float:
        positive, negative, gap = asymmetric_probability_gap(logits, targets)
        self.last_positive_confidence = float(positive.item())
        self.last_negative_confidence = float(negative.item())
        self.last_gap = float(gap.item())
        updated = self.gamma_neg + self.step_size * (self.target_gap - self.last_gap)
        self.gamma_neg = min(self.max_gamma_neg, max(self.min_gamma_neg, updated))
        return self.gamma_neg

    def diagnostics(self) -> dict[str, float]:
        return {
            "gamma_neg": self.gamma_neg,
            "positive_confidence": self.last_positive_confidence,
            "negative_confidence": self.last_negative_confidence,
            "probability_gap": self.last_gap,
        }


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
    denominator = torch.logsumexp(
        logits.masked_fill(~candidate_mask, -torch.inf), dim=dim
    )
    masked = logits.masked_fill(~positive_mask, -torch.inf)
    numerator = torch.logsumexp(masked, dim=dim)
    return (denominator[valid] - numerator[valid]).mean()


def partial_label_mask(
    positive_mask: torch.Tensor,
    class_positive_frequency: torch.Tensor,
    *,
    total_samples: int,
    balance_strength: float = 1.0,
    minimum_keep_probability: float = 0.05,
    random_values: torch.Tensor | None = None,
) -> torch.Tensor:
    """Generate a stochastic class-wise Partial Label Masking view.

    PLM changes each class's effective positive-to-negative ratio by masking
    individual labels instead of discarding whole multi-label samples.  This
    stable transfer interpolates every observed class ratio toward the mean
    ratio: tail classes mask negatives and head classes mask positives.  A
    strength of zero is the unmodified dataset; one reaches the common target.

    Reference: Duarte et al., *PLM: Partial Label Masking for Imbalanced
    Multi-label Classification*, CVPRW 2021, Sec. 2.1.  The paper subsequently
    adapts class ratios with epoch-level KL estimates.  Here the target is
    fixed within a run so a retrieval experiment does not silently tune from
    its own validation behaviour; this distinction is recorded in configs.
    """

    if positive_mask.ndim != 2:
        raise ValueError("positive_mask must have shape [classes, samples]")
    frequency = torch.as_tensor(
        class_positive_frequency,
        device=positive_mask.device,
        dtype=torch.float32,
    )
    if frequency.ndim != 1 or len(frequency) != positive_mask.shape[0]:
        raise ValueError("class_positive_frequency must match the class dimension")
    if total_samples <= 0 or (frequency < 0).any().item():
        raise ValueError("invalid PLM class frequencies or sample count")
    if (frequency > int(total_samples)).any().item():
        raise ValueError("class frequency cannot exceed total_samples")
    if not 0.0 <= balance_strength <= 1.0:
        raise ValueError("PLM balance_strength must be in [0, 1]")
    if not 0.0 < minimum_keep_probability <= 1.0:
        raise ValueError("PLM minimum_keep_probability must be in (0, 1]")

    eps = torch.finfo(frequency.dtype).eps
    negative_frequency = (float(total_samples) - frequency).clamp_min(eps)
    current_ratio = frequency.clamp_min(eps) / negative_frequency
    common_ratio = current_ratio.mean()
    target_ratio = torch.lerp(
        current_ratio,
        common_ratio.expand_as(current_ratio),
        float(balance_strength),
    ).clamp_min(eps)
    positive_keep = (target_ratio / current_ratio).clamp(max=1.0)
    negative_keep = (current_ratio / target_ratio).clamp(max=1.0)
    probabilities = torch.where(
        positive_mask.bool(),
        positive_keep[:, None],
        negative_keep[:, None],
    ).clamp_min(float(minimum_keep_probability))
    if random_values is None:
        random_values = torch.rand(
            positive_mask.shape,
            device=positive_mask.device,
            dtype=probabilities.dtype,
        )
    elif random_values.shape != positive_mask.shape:
        raise ValueError("random_values must match positive_mask")
    else:
        random_values = random_values.to(
            device=positive_mask.device, dtype=probabilities.dtype
        )
    return random_values < probabilities


def multi_positive_sigmoid_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    bias: torch.Tensor | float,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pairwise multi-positive sigmoid loss used by SigLIP and FFF.

    ``logits`` are already temperature-scaled similarities.  A negative bias
    represents the low prior probability of a positive pair and prevents the
    numerous negatives from causing a large correction at initialization.
    The row-sum then row-mean reduction follows the official SigLIP code; it
    also supports rectangular label-by-track matrices used for MTG-Jamendo.

    References: Zhai et al., ICCV 2023, Eq. 1; Bulat et al., CVPR 2024,
    Eq. 2.  Known duplicate/tag positives can vary freely by row.
    """

    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("logits and positive_mask must share a [Q,K] shape")
    positive = positive_mask.bool()
    if candidate_mask is None:
        candidate = torch.ones_like(positive)
    elif candidate_mask.shape != logits.shape:
        raise ValueError("candidate_mask must match logits")
    else:
        candidate = candidate_mask.bool()
    if (positive & ~candidate).any().item():
        raise ValueError("candidate_mask cannot remove a known positive")
    valid = candidate.any(dim=1) & positive.any(dim=1)
    if not valid.any().item():
        return logits.sum() * 0.0
    signed_targets = positive.to(logits.dtype).mul(2.0).sub(1.0)
    shifted = logits + torch.as_tensor(
        bias, device=logits.device, dtype=logits.dtype
    )
    pair_loss = -F.logsigmoid(signed_targets * shifted)
    pair_loss = pair_loss.masked_fill(~candidate, 0.0)
    return pair_loss.sum(dim=1)[valid].mean()


def lambda_ndcg_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    cutoff: int = 10,
    sigma: float = 1.0,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Macro query-wise LambdaRank loss for binary relevance at NDCG@k.

    Each row is one retrieval query and each column is a candidate item.  The
    logistic error for a known positive/negative pair is weighted by the
    absolute change in NDCG that would result from swapping their current
    (detached) ranks.  Per-query Delta-NDCG weights are normalized before
    averaging so rare and frequent MTG tags receive equal macro weight, which
    matches the project's validation metric.

    This is the LambdaRank configuration formalized as Eq. 11 in Wang et al.,
    *The LambdaLoss Framework for Ranking Metric Optimization*, CIKM 2018.
    Mini-batches replace a complete corpus ranking during stochastic training;
    that transfer is recorded explicitly rather than claimed as a replication.
    """

    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("logits and positive_mask must share a [Q,K] shape")
    if cutoff <= 0:
        raise ValueError("LambdaNDCG cutoff must be positive")
    if sigma <= 0:
        raise ValueError("LambdaNDCG sigma must be positive")
    positive = positive_mask.bool()
    if candidate_mask is None:
        candidate = torch.ones_like(positive)
    elif candidate_mask.shape != logits.shape:
        raise ValueError("candidate_mask must match logits")
    else:
        candidate = candidate_mask.bool()
    if (positive & ~candidate).any().item():
        raise ValueError("candidate_mask cannot remove a known positive")

    negative = candidate & ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any().item():
        return logits.sum() * 0.0
    with torch.no_grad():
        ranked_scores = logits.detach().masked_fill(~candidate, -torch.inf)
        order = torch.argsort(ranked_scores, dim=1, descending=True, stable=True)
        ranks = torch.empty_like(order)
        ordinal = torch.arange(
            1, logits.shape[1] + 1, device=logits.device, dtype=order.dtype
        ).expand_as(order)
        ranks.scatter_(1, order, ordinal)
        rank_values = ranks.to(dtype=logits.dtype)
        discounts = torch.where(
            candidate & (ranks <= int(cutoff)),
            torch.log2(rank_values + 1.0).reciprocal(),
            torch.zeros_like(rank_values),
        )
        ideal_discounts = torch.log2(
            torch.arange(
                1,
                min(int(cutoff), logits.shape[1]) + 1,
                device=logits.device,
                dtype=logits.dtype,
            )
            + 1.0
        ).reciprocal()
        ideal_prefix = torch.cat(
            (torch.zeros(1, device=logits.device, dtype=logits.dtype), ideal_discounts)
        ).cumsum(dim=0)
        positive_count = positive.sum(dim=1).clamp(max=len(ideal_discounts))
        ideal_dcg = ideal_prefix[positive_count]
        delta_ndcg = (
            discounts[:, :, None] - discounts[:, None, :]
        ).abs() / ideal_dcg[:, None, None].clamp_min(
            torch.finfo(logits.dtype).eps
        )
        pair_mask = positive[:, :, None] & negative[:, None, :]
        weights = delta_ndcg * pair_mask.to(delta_ndcg.dtype)
        weight_sum = weights.sum(dim=(1, 2))
        valid = valid & (weight_sum > 0)
    if not valid.any().item():
        return logits.sum() * 0.0
    pair_error = F.softplus(
        -float(sigma) * (logits[:, :, None] - logits[:, None, :])
    )
    per_query = (pair_error * weights).sum(dim=(1, 2)) / weight_sum.clamp_min(
        torch.finfo(logits.dtype).eps
    )
    return per_query[valid].mean()


def smooth_average_precision_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    temperature: float = 0.01,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Macro Smooth-AP loss for a batch of binary-relevance query lists.

    For every positive candidate, its hard rank among all candidates and among
    positives is relaxed by replacing the comparison indicator with a sigmoid.
    The resulting precision values are averaged first over positives and then
    equally over valid queries, matching macro average precision evaluation.

    Reference: Brown et al., *Smooth-AP: Smoothing the Path Towards
    Large-Scale Image Retrieval*, ECCV 2020, Eqs. 5--7.  Callers should pass
    cosine similarities (not temperature-scaled CLIP logits), because the
    paper's default smoothing temperature is defined in similarity space.
    """

    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive_mask must share a [Q,K] shape")
    if temperature <= 0:
        raise ValueError("Smooth-AP temperature must be positive")
    positive = positive_mask.bool()
    if candidate_mask is None:
        candidate = torch.ones_like(positive)
    elif candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scores")
    else:
        candidate = candidate_mask.bool()
    if (positive & ~candidate).any().item():
        raise ValueError("candidate_mask cannot remove a known positive")
    negative = candidate & ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any().item():
        return scores.sum() * 0.0

    # difference[q, i, j] is s_j - s_i: how strongly candidate j is
    # predicted above candidate i.  Self-comparisons are the explicit +1 rank
    # term in Eq. 5 and are therefore removed from the sigmoid sums.
    difference = scores[:, None, :] - scores[:, :, None]
    smooth_above = torch.sigmoid(difference / float(temperature))
    not_self = ~torch.eye(
        scores.shape[1], device=scores.device, dtype=torch.bool
    ).unsqueeze(0)
    positive_anchor = positive[:, :, None]
    positive_pairs = positive_anchor & positive[:, None, :] & not_self
    candidate_pairs = positive_anchor & candidate[:, None, :] & not_self
    positive_rank = 1.0 + (
        smooth_above * positive_pairs.to(smooth_above.dtype)
    ).sum(dim=2)
    overall_rank = 1.0 + (
        smooth_above * candidate_pairs.to(smooth_above.dtype)
    ).sum(dim=2)
    precision = positive_rank / overall_rank.clamp_min(
        torch.finfo(scores.dtype).eps
    )
    positive_count = positive.sum(dim=1).clamp_min(1)
    approximate_ap = (
        precision * positive.to(precision.dtype)
    ).sum(dim=1) / positive_count
    return (1.0 - approximate_ap[valid]).mean()


def circle_pair_similarity_loss(
    scores: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    dim: int = 1,
    margin: float = 0.25,
    gamma: float = 32.0,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Circle Loss for rectangular multi-positive retrieval matrices.

    Every valid query contributes one log-sum-exp over its positive
    similarities and one over its negative similarities.  Detached adaptive
    weights emphasize positives below their optimum and negatives above their
    optimum, while already satisfactory pairs receive little or no gradient.
    ``dim=0`` applies the same objective in the reverse retrieval direction.

    This is the pair-similarity form in Sun et al., *Circle Loss: A Unified
    Perspective of Pair Similarity Optimization*, CVPR 2020, Eqs. 6--8.  The
    caller should pass cosine similarities rather than temperature-scaled
    logits because the margin and convergence targets are defined on cosine
    space.  The rectangular/multi-positive reduction is the project's MTG
    transfer, not a claim of reproducing the original benchmark protocol.
    """

    if scores.shape != positive_mask.shape or scores.ndim != 2:
        raise ValueError("scores and positive_mask must share a [Q,K] shape")
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")
    if not 0.0 <= margin < 1.0:
        raise ValueError("Circle Loss margin must be in [0, 1)")
    if gamma <= 0:
        raise ValueError("Circle Loss gamma must be positive")
    positive = positive_mask.bool()
    if candidate_mask is None:
        candidate = torch.ones_like(positive)
    elif candidate_mask.shape != scores.shape:
        raise ValueError("candidate_mask must match scores")
    else:
        candidate = candidate_mask.bool()
    if (positive & ~candidate).any().item():
        raise ValueError("candidate_mask cannot remove a known positive")
    if dim == 0:
        scores = scores.T
        positive = positive.T
        candidate = candidate.T

    negative = candidate & ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any().item():
        return scores.sum() * 0.0
    scores = scores[valid]
    positive = positive[valid]
    negative = negative[valid]

    # Pair-specific weights are detached in the authors' reference
    # implementation.  For cosine similarity, O_p=1+m, O_n=-m,
    # Delta_p=1-m, and Delta_n=m.
    alpha_positive = (1.0 + float(margin) - scores.detach()).clamp_min(0.0)
    alpha_negative = (scores.detach() + float(margin)).clamp_min(0.0)
    positive_logits = (
        -float(gamma)
        * alpha_positive
        * (scores - (1.0 - float(margin)))
    ).masked_fill(~positive, -torch.inf)
    negative_logits = (
        float(gamma)
        * alpha_negative
        * (scores - float(margin))
    ).masked_fill(~negative, -torch.inf)
    positive_term = torch.logsumexp(positive_logits, dim=1)
    negative_term = torch.logsumexp(negative_logits, dim=1)
    return F.softplus(positive_term + negative_term).mean()


def multi_positive_margin_cross_entropy(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    dim: int = 1,
    margin: float = 0.05,
    logit_scale: torch.Tensor | float = 1.0,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-Positive Margin NCE cross-entropy from R-Adapter.

    Unlike :func:`multi_positive_cross_entropy`, which uses a log-sum-exp over
    positives, this objective gives every known positive equal target mass.
    A cosine-space margin is added to negative logits before softmax, making
    negative separation deliberately harder.  For retrieval R-Adapter uses no
    label smoothing; the argument is retained to reproduce its general form.

    Reference: Kim et al., ECCV 2024, equations 10--11.
    """

    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("logits and positive_mask must share a [Q,K] shape")
    if dim not in (0, 1):
        raise ValueError("dim must be 0 or 1")
    if margin < 0:
        raise ValueError("MPM-NCE margin must be non-negative")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("MPM-NCE label smoothing must be in [0, 1)")
    if dim == 0:
        logits = logits.T
        positive_mask = positive_mask.T
    positive = positive_mask.bool()
    positive_count = positive.sum(dim=1)
    negative = ~positive
    negative_count = negative.sum(dim=1)
    valid = positive_count > 0
    if label_smoothing > 0:
        valid &= negative_count > 0
    if not valid.any().item():
        return logits.sum() * 0.0

    scale = torch.as_tensor(
        logit_scale, device=logits.device, dtype=logits.dtype
    ).clamp(max=100.0)
    adjusted_logits = logits + negative.to(logits.dtype) * float(margin) * scale
    targets = positive.to(logits.dtype) * (
        (1.0 - float(label_smoothing))
        / positive_count.clamp_min(1).to(logits.dtype).unsqueeze(1)
    )
    if label_smoothing > 0:
        targets = targets + negative.to(logits.dtype) * (
            float(label_smoothing)
            / negative_count.clamp_min(1).to(logits.dtype).unsqueeze(1)
        )
    loss = -(targets * F.log_softmax(adjusted_logits, dim=1)).sum(dim=1)
    return loss[valid].mean()


def multi_label_supervised_contrastive_loss(
    embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.1,
    overlap_weighting: str = "binary",
) -> torch.Tensor:
    """Cluster music samples that share one or more observed labels.

    Unlabelled samples are excluded from both anchors and the denominator.
    With ``binary`` weighting this is the one-level supervised contrastive
    special case of HiMulCon.  ``jaccard`` gives stronger target mass to pairs
    sharing a larger fraction of their observed multi-label sets.

    Reference: Zhang et al., CVPR 2022, equations 2--4.
    """

    if embeddings.ndim != 2 or targets.ndim != 2:
        raise ValueError("embeddings and targets must both be rank-2")
    if len(embeddings) != len(targets):
        raise ValueError("embeddings and targets must describe the same samples")
    if temperature <= 0:
        raise ValueError("supervised contrastive temperature must be positive")
    if overlap_weighting not in {"binary", "jaccard"}:
        raise ValueError("overlap_weighting must be binary or jaccard")
    target = targets.bool()
    labelled = target.any(dim=1)
    if int(labelled.sum().item()) < 2:
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings[labelled], dim=-1)
    target = target[labelled]
    intersection = target.float() @ target.float().T
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    positive = (intersection > 0) & ~eye
    valid = positive.any(dim=1)
    if not valid.any().item():
        return embeddings.sum() * 0.0
    if overlap_weighting == "jaccard":
        cardinality = target.sum(dim=1).to(intersection.dtype)
        union = cardinality[:, None] + cardinality[None, :] - intersection
        positive_weight = (intersection / union.clamp_min(1.0)) * positive
    else:
        positive_weight = positive.to(intersection.dtype)
    soft_target = positive_weight / positive_weight.sum(dim=1, keepdim=True).clamp_min(
        1.0e-8
    )
    logits = (z @ z.T) / float(temperature)
    logits = logits.masked_fill(eye, -torch.inf)
    log_probabilities = F.log_softmax(logits, dim=1)
    # Multiplication by an explicit mask avoids the undefined 0 * -inf case.
    loss = -(soft_target * log_probabilities.masked_fill(~positive, 0.0)).sum(dim=1)
    return loss[valid].mean()


def multi_label_cross_view_contrastive_loss(
    query_embeddings: torch.Tensor,
    key_embeddings: torch.Tensor,
    targets: torch.Tensor,
    *,
    temperature: float = 0.07,
    overlap_weighting: str = "jaccard",
    normalize_targets: bool = False,
) -> torch.Tensor:
    """Align adapted and frozen music views using multi-label overlap.

    The query and key rows must refer to the same tracks.  Tracks without an
    observed label for the requested factor are excluded from both sides so
    that missing factor annotations are not treated as reliable negatives.
    The diagonal is retained: it is the same-track positive used by the
    audio-to-audio objective in TP-CLAP.  With ``jaccard`` weighting and
    ``normalize_targets=False`` this is the direct global-embedding transfer
    of TP-CLAP equations 14--15.  Target normalization is exposed only as an
    ablation because it changes the paper's per-anchor weighting.

    Reference: Li et al., Text-Prompted CLAP, arXiv:2607.25085v2,
    equations 14--15.
    """

    if query_embeddings.ndim != 2 or key_embeddings.ndim != 2:
        raise ValueError("query_embeddings and key_embeddings must be rank-2")
    if query_embeddings.shape != key_embeddings.shape:
        raise ValueError("query_embeddings and key_embeddings must share shape")
    if targets.ndim != 2 or len(targets) != len(query_embeddings):
        raise ValueError("targets must be [tracks, labels] for the embedding rows")
    if temperature <= 0:
        raise ValueError("cross-view contrastive temperature must be positive")
    if overlap_weighting not in {"binary", "jaccard"}:
        raise ValueError("overlap_weighting must be binary or jaccard")

    target = targets.bool()
    labelled = target.any(dim=1)
    if not labelled.any().item():
        return query_embeddings.sum() * 0.0
    query = F.normalize(query_embeddings[labelled], dim=-1)
    key = F.normalize(key_embeddings[labelled], dim=-1)
    target = target[labelled]
    intersection = target.float() @ target.float().T
    if overlap_weighting == "jaccard":
        cardinality = target.sum(dim=1).to(intersection.dtype)
        union = cardinality[:, None] + cardinality[None, :] - intersection
        weights = intersection / union.clamp_min(1.0)
    else:
        weights = (intersection > 0).to(intersection.dtype)
    if normalize_targets:
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    log_probabilities = F.log_softmax((query @ key.T) / float(temperature), dim=1)
    return -(weights * log_probabilities).sum(dim=1).mean()


def two_way_multi_label_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_temperature: float = 4.0,
    negative_temperature: float = 1.0,
) -> torch.Tensor:
    """Relative positive-vs-negative comparison across samples and classes.

    ``logits`` and ``targets`` use the conventional ``[samples, classes]``
    layout.  The first term discriminates tracks for each observed label; the
    second discriminates labels for each labelled track.  Rows or columns that
    cannot form a positive-negative comparison are excluded.

    Reference: Kobayashi, Two-Way Multi-Label Loss, CVPR 2023.  This follows
    the authors' public ``TwoWayLoss`` implementation with explicit validity
    masks for sparse mini-batches.
    """

    if logits.ndim != 2 or targets.ndim != 2 or logits.shape != targets.shape:
        raise ValueError("logits and targets must share a [samples, classes] shape")
    if positive_temperature <= 0 or negative_temperature <= 0:
        raise ValueError("two-way temperatures must be positive")
    positive = targets > 0
    negative = targets == 0
    valid_classes = positive.any(dim=0) & negative.any(dim=0)
    valid_samples = positive.any(dim=1) & negative.any(dim=1)
    if not valid_classes.any().item() and not valid_samples.any().item():
        return logits.sum() * 0.0

    positive_logits = torch.logsumexp(
        (-logits / float(positive_temperature)).masked_fill(~positive, -torch.inf),
        dim=0,
    ) * float(positive_temperature)
    negative_logits = torch.logsumexp(
        (logits / float(negative_temperature)).masked_fill(~negative, -torch.inf),
        dim=0,
    ) * float(negative_temperature)
    terms = []
    if valid_classes.any().item():
        terms.append(F.softplus(negative_logits + positive_logits)[valid_classes].mean())

    positive_logits = torch.logsumexp(
        (-logits / float(positive_temperature)).masked_fill(~positive, -torch.inf),
        dim=1,
    ) * float(positive_temperature)
    negative_logits = torch.logsumexp(
        (logits / float(negative_temperature)).masked_fill(~negative, -torch.inf),
        dim=1,
    ) * float(negative_temperature)
    if valid_samples.any().item():
        terms.append(F.softplus(negative_logits + positive_logits)[valid_samples].mean())
    return sum(terms)


def large_loss_negative_candidate_mask(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    ignore_fraction: float,
) -> torch.Tensor:
    """Exclude the highest-scoring unobserved tracks for each label query.

    In partially annotated multi-label data, an unobserved label is unknown,
    not a confirmed negative. Once the model has warmed up, the unobserved
    candidates with the largest logits also have the largest negative loss and
    are the most plausible false negatives. The returned detached boolean mask
    retains every known positive and all remaining reliable negatives.

    This is a retrieval adaptation of large-loss rejection for weakly
    supervised multi-label learning (Kim et al., CVPR 2022), not a claim of an
    exact reproduction of their image-classification algorithm.
    """

    if logits.shape != positive_mask.shape or logits.ndim != 2:
        raise ValueError("logits and positive_mask must share a [Q,K] shape")
    if not 0.0 <= ignore_fraction < 1.0:
        raise ValueError("ignore_fraction must be in [0, 1)")
    positive = positive_mask.bool()
    candidate = torch.ones_like(positive)
    if ignore_fraction == 0.0:
        return candidate
    with torch.no_grad():
        detached = logits.detach()
        for row in range(len(detached)):
            negative = ~positive[row]
            negative_count = int(negative.sum().item())
            ignore_count = int(negative_count * float(ignore_fraction))
            if ignore_count <= 0:
                continue
            scores = detached[row].masked_fill(~negative, -torch.inf)
            indices = torch.topk(scores, k=ignore_count, largest=True).indices
            candidate[row, indices] = False
    return candidate | positive


def cross_group_label_association(
    source_targets: torch.Tensor,
    target_targets: torch.Tensor,
    *,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Cosine-normalized train-set co-occurrence between two label groups."""

    if source_targets.ndim != 2 or target_targets.ndim != 2:
        raise ValueError("label targets must be rank-2")
    if len(source_targets) != len(target_targets):
        raise ValueError("source and target labels must describe the same samples")
    if eps <= 0:
        raise ValueError("eps must be positive")
    source = source_targets.float()
    target = target_targets.float()
    if ((source < 0) | (source > 1)).any().item() or (
        (target < 0) | (target > 1)
    ).any().item():
        raise ValueError("label targets must be binary values in [0, 1]")
    joint = source.T @ target
    denominator = torch.sqrt(
        source.sum(dim=0)[:, None] * target.sum(dim=0)[None, :]
    )
    return joint / denominator.clamp_min(float(eps))


def cross_group_pseudo_label_loss(
    target_logits: torch.Tensor,
    source_targets: torch.Tensor,
    observed_target_targets: torch.Tensor,
    association: torch.Tensor,
    *,
    top_k: int = 3,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Teach missing target groups from sparse cross-group co-occurrence.

    Only samples with at least one source-group label and no observed target
    label are used. Their source labels vote through a train-only association
    matrix; the strongest ``top_k`` target labels form a soft target. This is a
    lightweight training-only adaptation of classifier group chains for music
    tagging, so inference retains the original single-retriever architecture.
    """

    if target_logits.ndim != 2:
        raise ValueError("target_logits must be [samples, target_labels]")
    if source_targets.ndim != 2 or observed_target_targets.shape != target_logits.shape:
        raise ValueError("target tensors must match logits and be rank-2")
    if len(source_targets) != len(target_logits):
        raise ValueError("source targets must match the logits batch")
    if association.shape != (source_targets.shape[1], target_logits.shape[1]):
        raise ValueError("association has incompatible label dimensions")
    if not 1 <= top_k <= target_logits.shape[1]:
        raise ValueError("top_k must be within the target label count")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    source = source_targets.to(dtype=target_logits.dtype)
    observed = observed_target_targets.bool()
    valid = source.bool().any(dim=1) & ~observed.any(dim=1)
    if not valid.any().item():
        return target_logits.sum() * 0.0
    votes = source[valid] @ association.to(
        device=target_logits.device, dtype=target_logits.dtype
    )
    votes = votes / source[valid].sum(dim=1, keepdim=True).clamp_min(1.0)
    informative = votes.max(dim=1).values > 0
    if not informative.any().item():
        return target_logits.sum() * 0.0
    votes = votes[informative]
    logits = target_logits[valid][informative]
    top_values, top_indices = torch.topk(votes, k=int(top_k), dim=1)
    sparse_votes = torch.full_like(votes, -torch.inf)
    sparse_votes.scatter_(1, top_indices, top_values / float(temperature))
    pseudo_target = torch.softmax(sparse_votes, dim=1).detach()
    return -(pseudo_target * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


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
    text = F.normalize(text_embeddings, dim=-1)
    music = F.normalize(music_embeddings, dim=-1)
    scale = torch.as_tensor(logit_scale, dtype=text.dtype, device=text.device)
    logits = (text @ music.T) * scale.clamp(max=100.0)
    return 0.5 * (
        multi_positive_cross_entropy(logits, positive_mask, dim=1)
        + multi_positive_cross_entropy(logits, positive_mask, dim=0)
    )


def symmetric_modulated_multi_positive_info_nce(
    text_embeddings: torch.Tensor,
    music_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    minimum_temperature: float = 0.01,
    modulation_range: float = 0.04,
) -> torch.Tensor:
    """TeMo-style per-pair temperature for cross-modal InfoNCE.

    This transfers equations 1 and 5--9 of TeMo to the project's
    multi-positive identity mask.  A single-positive mask reduces to the
    paper's objective exactly.  The progressive mixture with standard
    InfoNCE is applied by the training loop, rather than inside this function.

    Reference: Duka et al., TeMo, arXiv:2609.07540v1 (2026).
    """

    if text_embeddings.ndim != 2 or music_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if text_embeddings.shape[1] != music_embeddings.shape[1]:
        raise ValueError("text/music embedding dimensions differ")
    if positive_mask.shape != (len(text_embeddings), len(music_embeddings)):
        raise ValueError("positive mask does not match text/music batch sizes")
    if minimum_temperature <= 0 or modulation_range < 0:
        raise ValueError("invalid modulated-temperature range")
    text = F.normalize(text_embeddings, dim=-1)
    music = F.normalize(music_embeddings, dim=-1)
    similarities = text @ music.T
    unit_similarities = ((1.0 + similarities) * 0.5).clamp(0.0, 1.0)
    temperatures = float(minimum_temperature) + float(modulation_range) * torch.sqrt(
        unit_similarities
    )
    logits = similarities / temperatures
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
    student_logits = scale * (
        F.normalize(student_text, dim=-1) @ F.normalize(student_music, dim=-1).T
    )
    with torch.no_grad():
        teacher_logits = scale * (
            F.normalize(teacher_text, dim=-1) @ F.normalize(teacher_music, dim=-1).T
        )
        teacher_t2m = torch.softmax(teacher_logits, dim=1)
        teacher_m2t = torch.softmax(teacher_logits.T, dim=1)
    text_loss = F.kl_div(
        F.log_softmax(student_logits, dim=1), teacher_t2m, reduction="batchmean"
    )
    music_loss = F.kl_div(
        F.log_softmax(student_logits.T, dim=1), teacher_m2t, reduction="batchmean"
    )
    return 0.5 * (text_loss + music_loss)


def symmetric_intra_modal_soft_target_loss(
    text_embeddings: torch.Tensor,
    music_embeddings: torch.Tensor,
    teacher_text_embeddings: torch.Tensor,
    teacher_music_embeddings: torch.Tensor,
    positive_mask: torch.Tensor,
    *,
    logit_scale: torch.Tensor | float,
    teacher_temperature: float = 0.07,
    soft_target_fraction: float = 0.1,
    text_teacher_weight: float = 0.5,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """SmoothCLAP-inspired soft supervision from intra-modal geometry.

    Frozen CLaMP3 text-text and music-music similarities replace the local
    speech-feature teacher used by SmoothCLAP.  The hard component is uniform
    over every known same-identity positive, which preserves this project's
    duplicate-aware semantics.  The loss is symmetric KL in both retrieval
    directions as in SmoothCLAP equation 11.

    Reference: Jing et al., SmoothCLAP, arXiv:2601.12591v1 (2026),
    equations 5--11.  This is a global-embedding transfer, not a reproduction
    of their speech-specific local feature extractor.
    """

    tensors = (
        text_embeddings,
        music_embeddings,
        teacher_text_embeddings,
        teacher_music_embeddings,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("all embedding inputs must be rank-2")
    if not all(tensor.shape == text_embeddings.shape for tensor in tensors):
        raise ValueError("student and teacher views must share a square paired batch")
    batch = len(text_embeddings)
    if positive_mask.shape != (batch, batch):
        raise ValueError("positive_mask must be square and match the paired batch")
    if teacher_temperature <= 0 or eps <= 0:
        raise ValueError("soft-target temperatures and eps must be positive")
    if not 0.0 < soft_target_fraction <= 1.0:
        raise ValueError("soft_target_fraction must be in (0, 1]")
    if not 0.0 <= text_teacher_weight <= 1.0:
        raise ValueError("text_teacher_weight must be in [0, 1]")

    positive = positive_mask.bool()
    valid = positive.any(dim=1) & positive.any(dim=0)
    if not valid.any().item():
        return text_embeddings.sum() * 0.0
    hard_target = positive.to(text_embeddings.dtype)
    hard_target = hard_target / hard_target.sum(dim=1, keepdim=True).clamp_min(1.0)
    with torch.no_grad():
        teacher_text = F.normalize(teacher_text_embeddings, dim=-1)
        teacher_music = F.normalize(teacher_music_embeddings, dim=-1)
        text_target = F.softmax(
            (teacher_text @ teacher_text.T) / float(teacher_temperature), dim=1
        )
        music_target = F.softmax(
            (teacher_music @ teacher_music.T) / float(teacher_temperature), dim=1
        )
        relation_target = (
            float(text_teacher_weight) * text_target
            + (1.0 - float(text_teacher_weight)) * music_target
        )
        target = (
            (1.0 - float(soft_target_fraction)) * hard_target
            + float(soft_target_fraction) * relation_target
        ).clamp_min(eps)
        target = target / target.sum(dim=1, keepdim=True)
        log_target = target.log()

    text = F.normalize(text_embeddings, dim=-1)
    music = F.normalize(music_embeddings, dim=-1)
    scale = torch.as_tensor(logit_scale, device=text.device, dtype=text.dtype).clamp(
        max=100.0
    )
    similarities = (text @ music.T) * scale

    def symmetric_kl(logits: torch.Tensor) -> torch.Tensor:
        log_prediction = F.log_softmax(logits, dim=1)
        prediction = log_prediction.exp()
        forward = (target * (log_target - log_prediction)).sum(dim=1)
        reverse = (prediction * (log_prediction - log_target)).sum(dim=1)
        return (forward + reverse)[valid].mean()

    return 0.5 * (symmetric_kl(similarities) + symmetric_kl(similarities.T))


def counterfactual_hard_negative_loss(
    scores: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    margin: float = 0.1,
) -> torch.Tensor:
    """Margin loss against the hardest valid counterfactual per query."""

    if scores.ndim != 2 or scores.shape != negative_mask.shape:
        raise ValueError("scores and negative_mask must share a [B,B] shape")
    if scores.shape[0] != scores.shape[1]:
        raise ValueError("counterfactual score matrix must be square")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    mask = negative_mask.bool()
    valid = mask.any(dim=1)
    if not valid.any().item():
        return scores.sum() * 0.0
    hardest = scores.masked_fill(~mask, -torch.inf).max(dim=1).values
    positive = scores.diagonal()
    return F.relu(float(margin) + hardest[valid] - positive[valid]).mean()


def factor_separation_loss(
    affect_embeddings: torch.Tensor,
    preference_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Penalize per-sample collapse of the two supervised factor spaces."""

    if affect_embeddings.shape != preference_embeddings.shape or affect_embeddings.ndim != 2:
        raise ValueError("factor embeddings must share a [B,D] shape")
    affect = F.normalize(affect_embeddings, dim=-1)
    preference = F.normalize(preference_embeddings, dim=-1)
    return (affect * preference).sum(dim=-1).square().mean()


def soft_cross_modal_va_alignment_loss(
    text_embeddings: torch.Tensor,
    music_embeddings: torch.Tensor,
    text_va: torch.Tensor,
    music_va: torch.Tensor,
    *,
    logit_scale: torch.Tensor | float,
    va_temperature: float = 0.1,
) -> torch.Tensor:
    """Align unpaired text/music batches through continuous VA soft positives."""

    if text_embeddings.ndim != 2 or music_embeddings.ndim != 2:
        raise ValueError("embeddings must be rank-2")
    if text_va.shape != (len(text_embeddings), 2):
        raise ValueError("text_va must have shape [text_batch, 2]")
    if music_va.shape != (len(music_embeddings), 2):
        raise ValueError("music_va must have shape [music_batch, 2]")
    if va_temperature <= 0:
        raise ValueError("va_temperature must be positive")
    squared_distance = torch.cdist(text_va.float(), music_va.float(), p=2).square()
    text_targets = torch.softmax(-squared_distance / va_temperature, dim=1).detach()
    music_targets = torch.softmax(-squared_distance.T / va_temperature, dim=1).detach()
    text = F.normalize(text_embeddings, dim=-1)
    music = F.normalize(music_embeddings, dim=-1)
    scale = torch.as_tensor(
        logit_scale, dtype=text_embeddings.dtype, device=text_embeddings.device
    ).clamp(max=100.0)
    logits = scale * (text @ music.T)
    text_loss = -(text_targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    music_loss = -(music_targets * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    return 0.5 * (text_loss + music_loss)
