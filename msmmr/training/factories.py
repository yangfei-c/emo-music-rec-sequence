"""由 YAML 配置构造模型、损失和优化器。"""

import torch

from msmmr.losses import GroupBalancedASL, VALoss
from msmmr.models import MusicMultiTaskModel, XLMRobertaVAModel


def build_music_model(config: dict, device: torch.device) -> MusicMultiTaskModel:
    model = config["model"]
    return MusicMultiTaskModel(
        layer_indices=config["feature"]["layer_indices"],
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        pooling=str(config["feature"].get("pooling", "mean_std")),
        pooling_eps=float(model.get("pooling_eps", 1.0e-5)),
    ).to(device)


def build_text_model(config: dict, device: torch.device) -> XLMRobertaVAModel:
    model, data = config["model"], config["data"]
    return XLMRobertaVAModel(
        pretrained_name=str(model["pretrained_name"]),
        hidden_dim=int(model["hidden_dim"]),
        dropout=float(model["dropout"]),
        max_length=int(data["max_length"]),
        trainable_encoder_layers=int(model["trainable_encoder_layers"]),
        local_files_only=bool(model.get("local_files_only", False)),
    ).to(device)


def build_tag_loss(config: dict) -> GroupBalancedASL:
    loss = config["loss"]["tag"]
    if str(loss.get("name", "asymmetric_loss")).lower() not in {
        "asymmetric_loss",
        "asl",
    }:
        raise ValueError("标签训练目前只支持 group-balanced ASL")
    return GroupBalancedASL(
        gamma_pos=float(loss["gamma_pos"]),
        gamma_neg=float(loss["gamma_neg"]),
        clip=float(loss["clip"]),
        eps=float(loss["eps"]),
    )


def build_va_loss(config: dict) -> torch.nn.Module:
    configured = config["loss"]
    loss = configured.get("va", configured)
    name = str(loss.get("name", "smooth_l1")).lower()
    if name in {"smooth_l1", "huber", "smooth_l1_ccc", "huber_ccc"}:
        return VALoss(
            beta=float(loss.get("beta", 0.1)),
            ccc_weight=float(loss.get("ccc_weight", 0.0)) if "ccc" in name else 0.0,
        )
    if name == "mse":
        return torch.nn.MSELoss()
    if name in {"mae", "l1"}:
        return torch.nn.L1Loss()
    raise ValueError("VA loss 只支持 smooth_l1、smooth_l1_ccc、mse、mae")


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    if str(optimizer.get("name", "adamw")).lower() != "adamw":
        raise ValueError("当前训练基础设施只支持 AdamW")
    parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return torch.optim.AdamW(
        parameters,
        lr=float(optimizer["learning_rate"]),
        weight_decay=float(optimizer["weight_decay"]),
    )


def build_text_optimizer(
    model: XLMRobertaVAModel, config: dict
) -> torch.optim.Optimizer:
    optimizer = config["optimizer"]
    if str(optimizer.get("name", "adamw")).lower() != "adamw":
        raise ValueError("文本训练只支持 AdamW")
    encoder = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    head = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not name.startswith("encoder.")
    ]
    groups = []
    if encoder:
        groups.append(
            {"params": encoder, "lr": float(optimizer["encoder_learning_rate"])}
        )
    groups.append({"params": head, "lr": float(optimizer["head_learning_rate"])})
    return torch.optim.AdamW(groups, weight_decay=float(optimizer["weight_decay"]))
