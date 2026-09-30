"""Score-function estimator: no gradient is claimed through discrete search."""
import torch


def policy_loss(selection, cost, baseline, warm_loss=None, warm_weight=0.1, entropy_weight=0.01):
    advantage = selection.log_prob.new_tensor(float(cost) - float(baseline)).detach()
    loss = advantage * selection.log_prob - entropy_weight * selection.entropy
    if warm_loss is not None:
        loss = loss + warm_weight * warm_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite policy loss")
    return loss
