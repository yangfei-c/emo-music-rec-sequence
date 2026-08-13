import torch
from torch import nn
from torch.nn import functional as F


class GroupBalancedASL(nn.Module):
    """Genre、instrument、mood/theme 分组等权 ASL。"""

    GROUPS = {
        "genre": slice(0, 87),
        "instrument": slice(87, 127),
        "mood": slice(127, 183),
    }

    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1.0e-8,
    ) -> None:
        super().__init__()
        if gamma_pos < 0 or gamma_neg < 0 or clip < 0 or eps <= 0:
            raise ValueError("ASL 参数错误")
        self.gamma_pos, self.gamma_neg, self.clip, self.eps = (
            gamma_pos,
            gamma_neg,
            clip,
            eps,
        )

    def asl(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos = logits.float().sigmoid()
        neg = 1.0 - pos
        if self.clip:
            neg = (neg + self.clip).clamp(max=1.0)

        targets = targets.float()
        probability = pos * targets + neg * (1.0 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
        log_probability = (
            targets * pos.clamp_min(self.eps).log()
            + (1.0 - targets) * neg.clamp_min(self.eps).log()
        )
        return -(log_probability * (1.0 - probability).pow(gamma)).mean()

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ):
        if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 183:
            raise ValueError(
                "logits 和 targets 应为 [B,183]，实际为 "
                f"{tuple(logits.shape)}、{tuple(targets.shape)}"
            )

        losses = {
            name: self.asl(logits[:, group], targets[:, group])
            for name, group in self.GROUPS.items()
        }
        total = torch.stack(tuple(losses.values())).mean()
        return {"total": total, **losses} if return_components else total


class TagSubsetWeightedBCE(nn.Module):
    """只训练一个 MTG 标签组，其余标签头不参与梯度。"""

    def __init__(self, label_slice: slice, positive_weights: torch.Tensor) -> None:
        super().__init__()
        size = label_slice.stop - label_slice.start
        if label_slice.step not in (None, 1) or label_slice.start < 0 or size <= 0:
            raise ValueError(f"无效标签切片：{label_slice}")
        if (
            positive_weights.shape != (size,)
            or not torch.isfinite(positive_weights).all()
            or (positive_weights <= 0).any()
        ):
            raise ValueError(f"positive_weights 应为 {size} 维正有限值")
        self.label_slice = label_slice
        self.register_buffer("positive_weights", positive_weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 183:
            raise ValueError("TagSubsetWeightedBCE 输入应为 [B,183]")
        return F.binary_cross_entropy_with_logits(
            logits[:, self.label_slice].float(),
            targets[:, self.label_slice].float(),
            pos_weight=self.positive_weights,
        )


class VALoss(nn.Module):
    """轴等权 Smooth L1，可选小权重 batch-CCC 正则。"""

    def __init__(
        self, beta: float = 0.1, ccc_weight: float = 0.0, eps: float = 1.0e-8
    ) -> None:
        super().__init__()
        if beta < 0 or ccc_weight < 0 or eps <= 0:
            raise ValueError("VA loss 参数错误")
        self.beta, self.ccc_weight, self.eps = beta, ccc_weight, eps

    def ccc(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prediction_mean, target_mean = predictions.mean(0), targets.mean(0)
        prediction_centered, target_centered = (
            predictions - prediction_mean,
            targets - target_mean,
        )
        covariance = (prediction_centered * target_centered).mean(0)
        denominator = (
            prediction_centered.square().mean(0)
            + target_centered.square().mean(0)
            + (prediction_mean - target_mean).square()
        )
        return (2.0 * covariance / denominator.clamp_min(self.eps)).clamp(-1.0, 1.0)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False,
    ):
        if (
            predictions.shape != targets.shape
            or predictions.ndim != 2
            or predictions.shape[1] != 2
        ):
            raise ValueError(
                "predictions 和 targets 应为 [B,2]，实际为 "
                f"{tuple(predictions.shape)}、{tuple(targets.shape)}"
            )

        predictions, targets = predictions.float(), targets.float()
        huber = F.smooth_l1_loss(
            predictions, targets, beta=self.beta, reduction="none"
        ).mean(0)
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
