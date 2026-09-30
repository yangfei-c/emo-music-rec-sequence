"""Train Ours on semantic pairs, MTG factors, and held-out VA supervision."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence.losses.va import VALoss
from emomm_sequence.evaluation.va_metrics import compute_va_metrics
from emomm_sequence.evaluation.pairs import grouped_retrieval_metrics
from emomm_sequence.models.retriever import RetrieverModel
from emomm_sequence.evaluation.retrieval import (
    counterfactual_factor_metrics,
    label_to_track_metrics,
)
from emomm_sequence.retrieval.factorization import (
    counterfactual_masks,
    mean_positive_label_embedding,
)
from emomm_sequence.losses.retrieval import (
    AdaptiveAsymmetryController,
    asymmetric_loss_with_logits,
    circle_pair_similarity_loss,
    counterfactual_hard_negative_loss,
    cross_group_label_association,
    cross_group_pseudo_label_loss,
    distribution_balanced_loss_with_logits,
    distributionally_robust_multilabel_loss,
    factor_separation_loss,
    large_loss_negative_candidate_mask,
    lambda_ndcg_loss,
    multi_positive_cross_entropy,
    multi_positive_margin_cross_entropy,
    multi_positive_sigmoid_loss,
    partial_label_mask,
    multi_label_cross_view_contrastive_loss,
    multi_label_supervised_contrastive_loss,
    soft_cross_modal_va_alignment_loss,
    smooth_average_precision_loss,
    symmetric_multi_positive_info_nce,
    symmetric_modulated_multi_positive_info_nce,
    symmetric_similarity_distillation_loss,
    symmetric_intra_modal_soft_target_loss,
    two_way_multi_label_loss,
)
from emomm_sequence.training.utilities import load_yaml, set_seed
from emomm_sequence.training.selection import (
    retrieval_selection_score,
    tag_balanced_selection_score,
)
from emomm_sequence.training.temperature import controlled_temperature
from emomm_sequence.data.loaders import load_pair_data, load_va_data, make_pair_loader, make_va_loader, make_mtg_loader
from emomm_sequence.training.records import check_identity_splits, sha256, state_sha256, training_provenance, write_json


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def cycle(loader: DataLoader):
    while True:
        yield from loader


def evenly_spaced_steps(active_steps: int, total_steps: int) -> set[int]:
    """Distribute a smaller task's batches across a longer multi-task epoch."""

    if not 0 <= active_steps <= total_steps or total_steps <= 0:
        raise ValueError("active_steps must be in [0, total_steps]")
    if active_steps == 0:
        return set()
    if active_steps == total_steps:
        return set(range(total_steps))
    return {
        min(total_steps - 1, (index * total_steps) // active_steps)
        for index in range(active_steps)
    }


def label_prior_logits(targets: np.ndarray, eps: float = 1.0e-4) -> torch.Tensor:
    """Empirical train-label priors used only to calibrate ASL probabilities."""

    values = np.asarray(targets, dtype=np.float64)
    if values.ndim != 2 or not 0 < eps < 0.5:
        raise ValueError("label priors require [tracks, labels] targets and valid eps")
    prevalence = np.clip(values.mean(axis=0), eps, 1.0 - eps)
    return torch.from_numpy(np.log(prevalence / (1.0 - prevalence)).astype(np.float32))


@torch.inference_mode()
def pair_evaluation(
    model: RetrieverModel,
    data: dict[str, np.ndarray],
    split: str,
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...],
) -> dict:
    mask = data["splits"].astype(str) == split
    text_raw = data["text_embeddings"][mask].astype(np.float32)
    music_raw = data["audio_embeddings"][mask].astype(np.float32)
    identities = data["global_identities"][mask].astype(str)
    datasets = data["datasets"][mask].astype(str)
    text_parts, music_parts = [], []
    model.eval()
    for start in range(0, len(text_raw), batch_size):
        text = torch.from_numpy(text_raw[start : start + batch_size]).to(device)
        music = torch.from_numpy(music_raw[start : start + batch_size]).to(device)
        text_parts.append(model.encode_factor("text", "semantic", text).cpu().numpy())
        music_parts.append(model.encode_factor("music", "semantic", music).cpu().numpy())
    text, music = np.concatenate(text_parts), np.concatenate(music_parts)
    overall = grouped_retrieval_metrics(text, music, identities, ks)
    overall["centroid_l2_gap"] = float(
        np.linalg.norm(text.mean(axis=0) - music.mean(axis=0))
    )
    per_dataset, selection = {}, []
    for name in sorted(set(datasets.tolist())):
        subset = datasets == name
        metrics = grouped_retrieval_metrics(text[subset], music[subset], identities[subset], ks)
        per_dataset[name] = metrics
        selection.append(
            0.5 * (metrics["text_to_music"]["mrr"] + metrics["music_to_text"]["mrr"])
        )
    return {"overall": overall, "per_dataset": per_dataset, "selection_score": float(np.mean(selection))}


@torch.inference_mode()
def va_evaluation(
    model: RetrieverModel,
    modality: str,
    loader: DataLoader,
    device: torch.device,
    domain_names: list[str],
) -> dict:
    predictions, targets, domains = [], [], []
    model.eval()
    for features, target, domain in loader:
        predictions.append(model.predict_va(modality, features.to(device)).cpu().numpy())
        targets.append(target.numpy())
        domains.append(domain.numpy())
    prediction, target, domain = map(np.concatenate, (predictions, targets, domains))
    overall = compute_va_metrics(prediction, target)
    per_dataset, selection = {}, []
    for index, name in enumerate(domain_names):
        subset = domain == index
        metrics = compute_va_metrics(prediction[subset], target[subset])
        per_dataset[name] = metrics
        selection.append(metrics["va_score"])
    return {"overall": overall, "per_dataset": per_dataset, "selection_score": float(np.mean(selection))}


@torch.inference_mode()
def tag_evaluation(
    model: RetrieverModel,
    factor: str,
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    split: str,
    device: torch.device,
    batch_size: int,
    ks: tuple[int, ...],
) -> dict:
    mask = mtg["splits"].astype(str) == split
    raw_text = mtg[f"{factor}_label_embeddings"].astype(np.float32)
    raw_music = music_index[mtg["index_positions"][mask]].astype(np.float32)
    targets = mtg[f"{factor}_targets"][mask].astype(bool).T
    text_parts, music_parts = [], []
    model.eval()
    for start in range(0, len(raw_text), batch_size):
        value = torch.from_numpy(raw_text[start : start + batch_size]).to(device)
        text_parts.append(model.encode_factor("text", factor, value).cpu().numpy())
    for start in range(0, len(raw_music), batch_size):
        value = torch.from_numpy(raw_music[start : start + batch_size]).to(device)
        music_parts.append(model.encode_factor("music", factor, value).cpu().numpy())
    scores = np.concatenate(text_parts) @ np.concatenate(music_parts).T
    result = label_to_track_metrics(scores, targets, ks)
    result.pop("per_query")
    return result


def gate_prior_loss(model: RetrieverModel, prior: torch.Tensor) -> torch.Tensor:
    weights = model.factor_weights().clamp_min(1.0e-8)
    prior = prior.to(weights)
    return torch.sum(weights * (weights.log() - prior.log()))


def capture_factor_state(model: RetrieverModel, factor: str) -> dict[str, torch.Tensor]:
    """Copy one factor and every head whose inputs depend on that factor."""

    if factor not in {"semantic", "affect", "preference"}:
        raise ValueError(f"unknown factor: {factor}")
    prefixes = [
        f"text_adapters.{factor}.",
        f"music_adapters.{factor}.",
        f"logit_scales.{factor}",
    ]
    if factor == "affect":
        # The VA heads must remain paired with the affect adapters from the
        # same validation-selected epoch.  Mixing epochs changes their input
        # distribution and caused avoidable CCC variance.
        prefixes.extend(("text_va_head.", "music_va_head."))
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(tuple(prefixes))
    }


def assembled_factor_state(
    model: RetrieverModel, factor_states: dict[str, dict[str, torch.Tensor]]
) -> dict[str, torch.Tensor]:
    """Assemble independently validation-selected factor heads."""

    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    for factor in ("semantic", "affect", "preference"):
        if factor not in factor_states:
            raise ValueError(f"missing validation-selected state for {factor}")
        state.update(factor_states[factor])
    return state


def freeze_factor_adapters(
    model: RetrieverModel, training_config: dict
) -> dict[str, list[str]]:
    """Freeze explicitly selected factor adapters without changing architecture."""

    valid = {"semantic", "affect", "preference"}
    frozen = {
        "text": [str(value) for value in training_config.get("freeze_text_adapters", [])],
        "music": [str(value) for value in training_config.get("freeze_music_adapters", [])],
    }
    for modality, factors in frozen.items():
        unknown = set(factors).difference(valid)
        if unknown:
            raise ValueError(
                f"unknown frozen {modality} adapter factors: "
                + ", ".join(sorted(unknown))
            )
        if len(factors) != len(set(factors)):
            raise ValueError(f"duplicate frozen {modality} adapter factor")
        if model.adapter_sharing == "across_factors" and factors and set(factors) != valid:
            raise ValueError("a shared adapter can only be frozen for every factor")
        adapters = model.text_adapters if modality == "text" else model.music_adapters
        for factor in factors:
            adapters[factor].requires_grad_(False)
    return frozen


def train_seed(
    config: dict,
    pair_data: dict[str, np.ndarray],
    text_va_data: dict[str, np.ndarray],
    music_va_data: dict[str, np.ndarray],
    mtg: dict[str, np.ndarray],
    music_index: np.ndarray,
    seed: int,
    output: Path,
    evaluate_test: bool,
) -> dict:
    set_seed(seed)
    random.seed(seed)
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model_config = config["model"]
    model = RetrieverModel(
        embedding_dim=int(config["backbone"]["embedding_dim"]),
        bottleneck_dim=int(model_config["bottleneck_dim"]),
        dropout=float(model_config["dropout"]),
        adapter_drop_probability=float(
            model_config.get("adapter_drop_probability", 0.0)
        ),
        adapter_drop_probabilities=model_config.get("adapter_drop_probabilities"),
        initial_adapter_scale=float(model_config["initial_adapter_scale"]),
        initial_factor_weights=tuple(float(x) for x in model_config["initial_factor_weights"]),
        initial_temperature=float(model_config["initial_temperature"]),
        train_factor_weights=bool(model_config["train_factor_weights"]),
        adapter_sharing=str(model_config.get("adapter_sharing", "none")),
        adapter_kind=str(model_config.get("adapter_kind", "residual")),
        adapter_blend=float(model_config.get("adapter_blend", 0.2)),
        semantic_output_blend=float(model_config.get("semantic_output_blend", 1.0)),
        affect_output_blend=float(model_config.get("affect_output_blend", 1.0)),
        preference_output_blend=float(model_config.get("preference_output_blend", 1.0)),
    ).to(device)
    training, loss_config = config["training"], config["loss"]
    frozen_adapters = freeze_factor_adapters(model, training)
    max_epochs = int(training["max_epochs"])
    raw_temperature_control = training.get("temperature_control", {})
    factor_temperature_control = {}
    for factor in ("semantic", "affect", "preference"):
        values = raw_temperature_control.get(factor, {})
        mode = str(values.get("mode", "learned"))
        start = float(values.get("start", model_config["initial_temperature"]))
        end = float(values.get("end", start))
        controlled = controlled_temperature(
            mode,
            start=start,
            end=end,
            epoch=1,
            total_epochs=max_epochs,
        )
        factor_temperature_control[factor] = {
            "mode": mode,
            "start": start,
            "end": end,
        }
        if controlled is not None:
            model.logit_scales[factor].requires_grad_(False)
            with torch.no_grad():
                model.logit_scales[factor].fill_(float(np.log(1.0 / controlled)))
    semantic_temperature_modulation = bool(
        loss_config.get("semantic_temperature_modulation", False)
    )
    semantic_modulation_minimum_temperature = float(
        loss_config.get("semantic_modulation_minimum_temperature", 0.01)
    )
    semantic_modulation_range = float(
        loss_config.get("semantic_modulation_range", 0.04)
    )
    semantic_modulation_schedule_power = float(
        loss_config.get("semantic_modulation_schedule_power", 2.0)
    )
    if semantic_modulation_minimum_temperature <= 0:
        raise ValueError("semantic modulation minimum temperature must be positive")
    if semantic_modulation_range < 0:
        raise ValueError("semantic modulation range must be non-negative")
    if semantic_modulation_schedule_power <= 0:
        raise ValueError("semantic modulation schedule power must be positive")
    semantic_smooth_target_mix = float(
        loss_config.get("semantic_smooth_target_mix", 0.0)
    )
    semantic_smooth_target_fraction = float(
        loss_config.get("semantic_smooth_target_fraction", 0.1)
    )
    semantic_smooth_target_text_weight = float(
        loss_config.get("semantic_smooth_target_text_weight", 0.5)
    )
    semantic_smooth_target_temperature = float(
        loss_config.get("semantic_smooth_target_temperature", 0.07)
    )
    if not 0.0 <= semantic_smooth_target_mix <= 1.0:
        raise ValueError("semantic smooth-target mix must be in [0, 1]")
    if not 0.0 < semantic_smooth_target_fraction <= 1.0:
        raise ValueError("semantic smooth-target fraction must be in (0, 1]")
    if not 0.0 <= semantic_smooth_target_text_weight <= 1.0:
        raise ValueError("semantic smooth-target text weight must be in [0, 1]")
    if semantic_smooth_target_temperature <= 0:
        raise ValueError("semantic smooth-target temperature must be positive")
    pair_loaders = {
        split: make_pair_loader(pair_data, split, int(training["batch_size_pairs"]), seed)
        for split in ("train", "validation", "test")
    }
    text_va_loaders, text_domains = {}, {}
    music_va_loaders, music_domains = {}, {}
    for split in ("train", "validation", "test"):
        text_va_loaders[split], text_domains[split] = make_va_loader(
            text_va_data, "text_embeddings", split, int(training["batch_size_va"]), seed
        )
        music_va_loaders[split], music_domains[split] = make_va_loader(
            music_va_data, "audio_embeddings", split, int(training["batch_size_va"]), seed
        )
    mtg_loaders = {
        split: make_mtg_loader(mtg, music_index, split, int(training["batch_size_mtg"]), seed)
        for split in ("train", "validation", "test")
    }
    preference_labels = torch.from_numpy(
        mtg["preference_label_embeddings"].astype(np.float32)
    ).to(device)
    affect_labels = torch.from_numpy(mtg["affect_label_embeddings"].astype(np.float32)).to(device)
    va_criterion = VALoss(
        beta=float(loss_config["huber_beta"]), ccc_weight=float(loss_config["ccc_weight"])
    )
    tag_loss_mode = str(loss_config.get("tag_loss", "contrastive"))
    if tag_loss_mode not in {"contrastive", "asl", "hybrid"}:
        raise ValueError("loss.tag_loss must be contrastive, asl, or hybrid")
    default_asl_mix = float(loss_config.get("asl_mix", 0.5))
    if tag_loss_mode == "contrastive":
        default_asl_mix = 0.0
    elif tag_loss_mode == "asl":
        default_asl_mix = 1.0
    factor_asl_mix = {
        "preference": float(
            loss_config.get("preference_asl_mix", default_asl_mix)
        ),
        "affect": float(loss_config.get("affect_asl_mix", default_asl_mix)),
    }
    if any(not 0.0 <= value <= 1.0 for value in factor_asl_mix.values()):
        raise ValueError("factor ASL mixing weights must be in [0, 1]")
    factor_db_mix = {
        "preference": float(loss_config.get("preference_db_mix", 0.0)),
        "affect": float(loss_config.get("affect_db_mix", 0.0)),
    }
    factor_dr_mix = {
        "preference": float(loss_config.get("preference_dr_mix", 0.0)),
        "affect": float(loss_config.get("affect_dr_mix", 0.0)),
    }
    factor_mpm_mix = {
        "preference": float(loss_config.get("preference_mpm_mix", 0.0)),
        "affect": float(loss_config.get("affect_mpm_mix", 0.0)),
    }
    mpm_margin = float(loss_config.get("mpm_margin", 0.05))
    mpm_label_smoothing = float(loss_config.get("mpm_label_smoothing", 0.0))
    factor_two_way_mix = {
        "preference": float(loss_config.get("preference_two_way_mix", 0.0)),
        "affect": float(loss_config.get("affect_two_way_mix", 0.0)),
    }
    two_way_positive_temperature = float(
        loss_config.get("two_way_positive_temperature", 4.0)
    )
    two_way_negative_temperature = float(
        loss_config.get("two_way_negative_temperature", 1.0)
    )
    factor_sigmoid_mix = {
        "preference": float(loss_config.get("preference_sigmoid_mix", 0.0)),
        "affect": float(loss_config.get("affect_sigmoid_mix", 0.0)),
    }
    factor_sigmoid_bias = {
        "preference": float(loss_config.get("preference_sigmoid_bias", -3.5)),
        "affect": float(loss_config.get("affect_sigmoid_bias", -4.5)),
    }
    factor_lambda_ndcg_mix = {
        "preference": float(loss_config.get("preference_lambda_ndcg_mix", 0.0)),
        "affect": float(loss_config.get("affect_lambda_ndcg_mix", 0.0)),
    }
    lambda_ndcg_cutoff = int(loss_config.get("lambda_ndcg_cutoff", 10))
    lambda_ndcg_sigma = float(loss_config.get("lambda_ndcg_sigma", 1.0))
    factor_smooth_ap_mix = {
        "preference": float(loss_config.get("preference_smooth_ap_mix", 0.0)),
        "affect": float(loss_config.get("affect_smooth_ap_mix", 0.0)),
    }
    smooth_ap_temperature = float(loss_config.get("smooth_ap_temperature", 0.01))
    factor_circle_mix = {
        "preference": float(loss_config.get("preference_circle_mix", 0.0)),
        "affect": float(loss_config.get("affect_circle_mix", 0.0)),
    }
    circle_margin = float(loss_config.get("circle_margin", 0.25))
    circle_gamma = float(loss_config.get("circle_gamma", 32.0))
    factor_plm_mix = {
        "preference": float(loss_config.get("preference_plm_mix", 0.0)),
        "affect": float(loss_config.get("affect_plm_mix", 0.0)),
    }
    plm_balance_strength = float(loss_config.get("plm_balance_strength", 1.0))
    plm_minimum_keep_probability = float(
        loss_config.get("plm_minimum_keep_probability", 0.05)
    )
    factor_track_contrastive_weight = {
        "preference": float(
            loss_config.get("preference_track_contrastive_weight", 0.0)
        ),
        "affect": float(loss_config.get("affect_track_contrastive_weight", 0.0)),
    }
    track_contrastive_temperature = float(
        loss_config.get("track_contrastive_temperature", 0.1)
    )
    track_contrastive_overlap = str(
        loss_config.get("track_contrastive_overlap", "binary")
    )
    factor_frozen_track_contrastive_weight = {
        "preference": float(
            loss_config.get("preference_frozen_track_contrastive_weight", 0.0)
        ),
        "affect": float(
            loss_config.get("affect_frozen_track_contrastive_weight", 0.0)
        ),
    }
    frozen_track_contrastive_temperature = float(
        loss_config.get("frozen_track_contrastive_temperature", 0.07)
    )
    frozen_track_contrastive_overlap = str(
        loss_config.get("frozen_track_contrastive_overlap", "jaccard")
    )
    frozen_track_contrastive_normalize_targets = bool(
        loss_config.get("frozen_track_contrastive_normalize_targets", False)
    )
    factor_labeled_only = {
        "preference": bool(loss_config.get("preference_labeled_tracks_only", False)),
        "affect": bool(loss_config.get("affect_labeled_tracks_only", False)),
    }
    factor_label_to_track_weight = {
        "preference": float(
            loss_config.get("preference_label_to_track_weight", 0.5)
        ),
        "affect": float(loss_config.get("affect_label_to_track_weight", 0.5)),
    }
    factor_false_negative_fraction = {
        "preference": float(
            loss_config.get("preference_false_negative_fraction", 0.0)
        ),
        "affect": float(loss_config.get("affect_false_negative_fraction", 0.0)),
    }
    false_negative_warmup_epochs = int(
        loss_config.get("false_negative_warmup_epochs", 1)
    )
    if any(not 0.0 <= value <= 1.0 for value in factor_db_mix.values()):
        raise ValueError("factor DB mixing weights must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in factor_dr_mix.values()):
        raise ValueError("factor DR mixing weights must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in factor_mpm_mix.values()):
        raise ValueError("factor MPM-NCE mixing weights must be in [0, 1]")
    if mpm_margin < 0 or not 0.0 <= mpm_label_smoothing < 1.0:
        raise ValueError("invalid MPM-NCE margin or label smoothing")
    if any(not 0.0 <= value <= 1.0 for value in factor_two_way_mix.values()):
        raise ValueError("factor two-way mixing weights must be in [0, 1]")
    if two_way_positive_temperature <= 0 or two_way_negative_temperature <= 0:
        raise ValueError("two-way temperatures must be positive")
    if any(not 0.0 <= value <= 1.0 for value in factor_sigmoid_mix.values()):
        raise ValueError("factor sigmoid mixing weights must be in [0, 1]")
    if any(not np.isfinite(value) for value in factor_sigmoid_bias.values()):
        raise ValueError("factor sigmoid biases must be finite")
    if any(not 0.0 <= value <= 1.0 for value in factor_lambda_ndcg_mix.values()):
        raise ValueError("factor LambdaNDCG mixing weights must be in [0, 1]")
    if lambda_ndcg_cutoff <= 0 or lambda_ndcg_sigma <= 0:
        raise ValueError("LambdaNDCG cutoff and sigma must be positive")
    if any(not 0.0 <= value <= 1.0 for value in factor_smooth_ap_mix.values()):
        raise ValueError("factor Smooth-AP mixing weights must be in [0, 1]")
    if smooth_ap_temperature <= 0:
        raise ValueError("Smooth-AP temperature must be positive")
    if any(not 0.0 <= value <= 1.0 for value in factor_circle_mix.values()):
        raise ValueError("factor Circle Loss mixing weights must be in [0, 1]")
    if not 0.0 <= circle_margin < 1.0 or circle_gamma <= 0:
        raise ValueError("invalid Circle Loss margin or gamma")
    if any(not 0.0 <= value <= 1.0 for value in factor_plm_mix.values()):
        raise ValueError("factor PLM mixing weights must be in [0, 1]")
    if not 0.0 <= plm_balance_strength <= 1.0:
        raise ValueError("PLM balance strength must be in [0, 1]")
    if not 0.0 < plm_minimum_keep_probability <= 1.0:
        raise ValueError("PLM minimum keep probability must be in (0, 1]")
    if any(value < 0 for value in factor_track_contrastive_weight.values()):
        raise ValueError("track supervised-contrastive weights must be non-negative")
    if track_contrastive_temperature <= 0:
        raise ValueError("track supervised-contrastive temperature must be positive")
    if track_contrastive_overlap not in {"binary", "jaccard"}:
        raise ValueError("track contrastive overlap must be binary or jaccard")
    if any(value < 0 for value in factor_frozen_track_contrastive_weight.values()):
        raise ValueError("frozen-track contrastive weights must be non-negative")
    if frozen_track_contrastive_temperature <= 0:
        raise ValueError("frozen-track contrastive temperature must be positive")
    if frozen_track_contrastive_overlap not in {"binary", "jaccard"}:
        raise ValueError("frozen-track contrastive overlap must be binary or jaccard")
    if any(
        not 0.0 <= value <= 1.0 for value in factor_label_to_track_weight.values()
    ):
        raise ValueError("label-to-track ranking weights must be in [0, 1]")
    if any(
        not 0.0 <= value < 1.0 for value in factor_false_negative_fraction.values()
    ):
        raise ValueError("false-negative rejection fractions must be in [0, 1)")
    if false_negative_warmup_epochs < 0:
        raise ValueError("false-negative warmup epochs must be non-negative")
    if any(
        factor_db_mix[factor] > 0 and not factor_labeled_only[factor]
        for factor in ("preference", "affect")
    ):
        raise ValueError(
            "Distribution-Balanced Loss requires factor_labeled_tracks_only"
        )
    mtg_train_mask = mtg["splits"].astype(str) == "train"
    factor_class_frequency = {
        factor: torch.from_numpy(
            mtg[f"{factor}_targets"][mtg_train_mask].sum(axis=0).astype(np.float32)
        ).to(device)
        for factor in ("preference", "affect")
    }
    factor_db_train_samples = {
        factor: int(
            mtg[f"{factor}_targets"][mtg_train_mask].astype(bool).any(axis=1).sum()
        )
        for factor in ("preference", "affect")
    }
    affect_group_chain_weight = float(
        loss_config.get("affect_group_chain_weight", 0.0)
    )
    affect_group_chain_top_k = int(loss_config.get("affect_group_chain_top_k", 3))
    affect_group_chain_temperature = float(
        loss_config.get("affect_group_chain_temperature", 0.1)
    )
    if affect_group_chain_weight < 0:
        raise ValueError("affect group-chain weight must be non-negative")
    if not 1 <= affect_group_chain_top_k <= mtg["affect_targets"].shape[1]:
        raise ValueError("affect group-chain top-k is outside the label count")
    if affect_group_chain_temperature <= 0:
        raise ValueError("affect group-chain temperature must be positive")
    affect_group_association = cross_group_label_association(
        torch.from_numpy(
            mtg["preference_targets"][mtg_train_mask].astype(np.float32)
        ),
        torch.from_numpy(mtg["affect_targets"][mtg_train_mask].astype(np.float32)),
    ).to(device)
    preference_prior = label_prior_logits(
        mtg["preference_targets"][mtg_train_mask]
    ).to(device)
    affect_prior = label_prior_logits(mtg["affect_targets"][mtg_train_mask]).to(device)
    asl_dynamic = bool(loss_config.get("asl_dynamic", False))
    asl_controllers = {
        factor: AdaptiveAsymmetryController(
            initial_gamma_neg=float(
                loss_config.get(
                    "asl_dynamic_initial_gamma_neg", loss_config.get("asl_gamma_neg", 4.0)
                )
            ),
            target_gap=float(loss_config.get("asl_target_gap", 0.1)),
            step_size=float(loss_config.get("asl_step_size", 0.01)),
            min_gamma_neg=float(loss_config.get("asl_min_gamma_neg", 0.0)),
            max_gamma_neg=float(loss_config.get("asl_max_gamma_neg", 20.0)),
        )
        for factor in ("preference", "affect")
    }

    base_learning_rate = float(training["learning_rate"])
    semantic_learning_rate = float(training.get("semantic_learning_rate", base_learning_rate))
    if semantic_learning_rate != base_learning_rate:
        if model.adapter_sharing != "none":
            raise ValueError("a separate semantic learning rate requires independent adapters")
        semantic_parameters = [
            *model.text_adapters["semantic"].parameters(),
            *model.music_adapters["semantic"].parameters(),
            model.logit_scales["semantic"],
        ]
        semantic_ids = {id(parameter) for parameter in semantic_parameters}
        other_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in semantic_ids
        ]
        optimizer_parameters: list[dict] = [
            {"params": semantic_parameters, "lr": semantic_learning_rate},
            {"params": other_parameters, "lr": base_learning_rate},
        ]
    else:
        optimizer_parameters = [{"params": list(model.parameters()), "lr": base_learning_rate}]
    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        weight_decay=float(training["weight_decay"]),
    )
    gradient_clip_mode = str(training.get("gradient_clip_mode", "global"))
    if gradient_clip_mode not in {"global", "factorwise"}:
        raise ValueError("training.gradient_clip_mode must be global or factorwise")
    if gradient_clip_mode == "factorwise":
        if model.adapter_sharing != "none":
            raise ValueError("factorwise clipping requires independent adapters")
        gradient_parameter_groups = [
            [
                *model.text_adapters["semantic"].parameters(),
                *model.music_adapters["semantic"].parameters(),
                model.logit_scales["semantic"],
            ],
            [
                *model.text_adapters["affect"].parameters(),
                *model.music_adapters["affect"].parameters(),
                *model.text_va_head.parameters(),
                *model.music_va_head.parameters(),
                model.logit_scales["affect"],
            ],
            [
                *model.text_adapters["preference"].parameters(),
                *model.music_adapters["preference"].parameters(),
                model.logit_scales["preference"],
            ],
            [model.factor_logits],
        ]
    else:
        gradient_parameter_groups = [list(model.parameters())]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(training["amp"]) and device.type == "cuda"
    )
    initial_prior = torch.tensor(model_config["initial_factor_weights"], dtype=torch.float32)
    initial_prior /= initial_prior.sum()
    output.mkdir(parents=True, exist_ok=True)
    provenance = training_provenance(model, seed, config)
    write_json(output / "provenance.json", provenance)
    torch.save({"model": model.state_dict(), "seed": seed, "initial_checkpoint": None}, output / "initial.pt")
    selection_metric = str(
        config["evaluation"].get("selection_metric", "validation_composite")
    )
    factorwise_metrics = {"factorwise_retrieval", "factorwise_balanced_retrieval"}
    if selection_metric in factorwise_metrics and model.adapter_sharing != "none":
        raise ValueError("factorwise checkpoint assembly requires six independent adapters")
    history, best_score, patience_reference, best_epoch, bad_epochs = [], -float("inf"), -float("inf"), 0, 0
    factor_best_scores = {
        factor: -float("inf") for factor in ("semantic", "affect", "preference")
    }
    factor_best_epochs = {factor: 0 for factor in factor_best_scores}
    factor_best_states: dict[str, dict[str, torch.Tensor]] = {}
    retrieval_ks = tuple(int(k) for k in config["evaluation"]["retrieval_ks"])
    tag_ks = tuple(int(k) for k in config["evaluation"]["tag_ks"])
    for epoch in range(1, max_epochs + 1):
        with torch.no_grad():
            for factor, values in factor_temperature_control.items():
                temperature = controlled_temperature(
                    values["mode"],
                    start=values["start"],
                    end=values["end"],
                    epoch=epoch,
                    total_epochs=max_epochs,
                )
                if temperature is not None:
                    model.logit_scales[factor].fill_(
                        float(np.log(1.0 / temperature))
                    )
        model.train()
        mtg_iter = cycle(mtg_loaders["train"])
        text_va_iter = cycle(text_va_loaders["train"])
        music_va_iter = cycle(music_va_loaders["train"])
        steps = max(
            len(pair_loaders["train"]),
            len(mtg_loaders["train"]),
            len(text_va_loaders["train"]),
            len(music_va_loaders["train"]),
        )
        pair_sampling = str(training.get("semantic_pair_sampling", "cycle"))
        if pair_sampling == "cycle":
            pair_iter = cycle(pair_loaders["train"])
            pair_steps = set(range(steps))
        elif pair_sampling == "once":
            pair_iter = iter(pair_loaders["train"])
            pair_steps = evenly_spaced_steps(len(pair_loaders["train"]), steps)
        else:
            raise ValueError("training.semantic_pair_sampling must be cycle or once")
        totals = {
            key: 0.0
            for key in (
                "total",
                "semantic",
                "semantic_standard",
                "semantic_modulated",
                "semantic_smooth_target",
                "semantic_distillation",
                "preference",
                "preference_ranking",
                "preference_mpm",
                "preference_two_way",
                "preference_sigmoid",
                "preference_lambda_ndcg",
                "preference_smooth_ap",
                "preference_circle",
                "preference_plm",
                "preference_track_contrastive",
                "preference_frozen_track_contrastive",
                "preference_asl",
                "preference_db",
                "preference_dr",
                "affect_tag",
                "affect_ranking",
                "affect_mpm",
                "affect_two_way",
                "affect_sigmoid",
                "affect_lambda_ndcg",
                "affect_smooth_ap",
                "affect_circle",
                "affect_plm",
                "affect_track_contrastive",
                "affect_frozen_track_contrastive",
                "affect_asl",
                "affect_db",
                "affect_dr",
                "affect_group_chain",
                "text_va",
                "music_va",
                "counterfactual",
                "separation",
                "gate_prior",
                "cross_modal_va",
            )
        }
        counts = {key: 0 for key in totals}
        for step in range(steps):
            mtg_music, preference_target, affect_target = next(mtg_iter)
            text_va_features, text_va_target, _ = next(text_va_iter)
            music_va_features, music_va_target, _ = next(music_va_iter)
            mtg_music = mtg_music.to(device)
            preference_target = preference_target.to(device)
            affect_target = affect_target.to(device)
            text_va_features = text_va_features.to(device)
            text_va_target = text_va_target.to(device)
            music_va_features = music_va_features.to(device)
            music_va_target = music_va_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=bool(training["amp"]) and device.type == "cuda"
            ):
                zero = mtg_music.sum() * 0.0
                semantic_loss = zero
                semantic_standard_loss = zero
                semantic_modulated_loss = zero
                semantic_smooth_target_loss = zero
                semantic_distillation_loss = zero
                semantic_active = step in pair_steps
                if semantic_active:
                    pair_text, pair_music, pair_identity, _ = next(pair_iter)
                    pair_text = pair_text.to(device)
                    pair_music = pair_music.to(device)
                    pair_identity = pair_identity.to(device)
                    semantic_text = model.encode_factor("text", "semantic", pair_text)
                    semantic_music = model.encode_factor("music", "semantic", pair_music)
                    pair_positive = pair_identity[:, None] == pair_identity[None, :]
                    semantic_standard_loss = symmetric_multi_positive_info_nce(
                        semantic_text,
                        semantic_music,
                        pair_positive,
                        logit_scale=model.logit_scales["semantic"].exp(),
                    )
                    if semantic_temperature_modulation:
                        semantic_modulated_loss = (
                            symmetric_modulated_multi_positive_info_nce(
                                semantic_text,
                                semantic_music,
                                pair_positive,
                                minimum_temperature=(
                                    semantic_modulation_minimum_temperature
                                ),
                                modulation_range=semantic_modulation_range,
                            )
                        )
                        total_semantic_steps = max(max_epochs * steps - 1, 1)
                        normalized_step = (
                            (epoch - 1) * steps + step
                        ) / total_semantic_steps
                        fixed_weight = (1.0 - normalized_step) ** (
                            semantic_modulation_schedule_power
                        )
                        modulated_weight = normalized_step ** (
                            semantic_modulation_schedule_power
                        )
                        semantic_loss = (
                            fixed_weight * semantic_standard_loss
                            + modulated_weight * semantic_modulated_loss
                        )
                    else:
                        semantic_loss = semantic_standard_loss
                    if semantic_smooth_target_mix > 0:
                        semantic_smooth_target_loss = (
                            symmetric_intra_modal_soft_target_loss(
                                semantic_text,
                                semantic_music,
                                pair_text,
                                pair_music,
                                pair_positive,
                                logit_scale=model.logit_scales["semantic"].exp(),
                                teacher_temperature=(
                                    semantic_smooth_target_temperature
                                ),
                                soft_target_fraction=(
                                    semantic_smooth_target_fraction
                                ),
                                text_teacher_weight=(
                                    semantic_smooth_target_text_weight
                                ),
                            )
                        )
                        semantic_loss = (
                            (1.0 - semantic_smooth_target_mix) * semantic_loss
                            + semantic_smooth_target_mix
                            * semantic_smooth_target_loss
                        )
                    semantic_distillation_loss = symmetric_similarity_distillation_loss(
                        semantic_text,
                        semantic_music,
                        pair_text,
                        pair_music,
                        temperature=float(
                            loss_config.get("semantic_distillation_temperature", 0.07)
                        ),
                    )

                preference_text = model.encode_factor("text", "preference", preference_labels)
                affect_text = model.encode_factor("text", "affect", affect_labels)
                music_factors = model.encode_music(mtg_music)
                preference_logits = (
                    preference_text @ music_factors["preference"].T
                ) * model.logit_scales["preference"].exp().clamp(max=100.0)
                affect_logits = (
                    affect_text @ music_factors["affect"].T
                ) * model.logit_scales["affect"].exp().clamp(max=100.0)
                preference_positive = preference_target.T
                affect_positive = affect_target.T
                preference_valid_tracks = preference_target.bool().any(dim=1)
                affect_valid_tracks = affect_target.bool().any(dim=1)
                preference_loss_logits = (
                    preference_logits[:, preference_valid_tracks]
                    if factor_labeled_only["preference"]
                    else preference_logits
                )
                affect_loss_logits = (
                    affect_logits[:, affect_valid_tracks]
                    if factor_labeled_only["affect"]
                    else affect_logits
                )
                preference_loss_target = (
                    preference_target[preference_valid_tracks]
                    if factor_labeled_only["preference"]
                    else preference_target
                )
                affect_loss_target = (
                    affect_target[affect_valid_tracks]
                    if factor_labeled_only["affect"]
                    else affect_target
                )
                preference_loss_positive = preference_loss_target.T
                affect_loss_positive = affect_loss_target.T
                preference_direction = factor_label_to_track_weight["preference"]
                affect_direction = factor_label_to_track_weight["affect"]
                preference_candidate_mask = None
                affect_candidate_mask = None
                if epoch > false_negative_warmup_epochs:
                    if factor_false_negative_fraction["preference"] > 0:
                        preference_candidate_mask = large_loss_negative_candidate_mask(
                            preference_loss_logits,
                            preference_loss_positive,
                            ignore_fraction=factor_false_negative_fraction["preference"],
                        )
                    if factor_false_negative_fraction["affect"] > 0:
                        affect_candidate_mask = large_loss_negative_candidate_mask(
                            affect_loss_logits,
                            affect_loss_positive,
                            ignore_fraction=factor_false_negative_fraction["affect"],
                        )
                preference_standard_ranking_loss = (
                    preference_direction
                    * multi_positive_cross_entropy(
                        preference_loss_logits,
                        preference_loss_positive,
                        dim=1,
                        candidate_mask=preference_candidate_mask,
                    )
                    + (1.0 - preference_direction)
                    * multi_positive_cross_entropy(
                        preference_loss_logits, preference_loss_positive, dim=0
                    )
                )
                affect_standard_ranking_loss = (
                    affect_direction
                    * multi_positive_cross_entropy(
                        affect_loss_logits,
                        affect_loss_positive,
                        dim=1,
                        candidate_mask=affect_candidate_mask,
                    )
                    + (1.0 - affect_direction)
                    * multi_positive_cross_entropy(
                        affect_loss_logits, affect_loss_positive, dim=0
                    )
                )
                preference_mpm_loss = (
                    preference_direction
                    * multi_positive_margin_cross_entropy(
                        preference_loss_logits,
                        preference_loss_positive,
                        dim=1,
                        margin=mpm_margin,
                        logit_scale=model.logit_scales["preference"].exp(),
                        label_smoothing=mpm_label_smoothing,
                    )
                    + (1.0 - preference_direction)
                    * multi_positive_margin_cross_entropy(
                        preference_loss_logits,
                        preference_loss_positive,
                        dim=0,
                        margin=mpm_margin,
                        logit_scale=model.logit_scales["preference"].exp(),
                        label_smoothing=mpm_label_smoothing,
                    )
                )
                affect_mpm_loss = (
                    affect_direction
                    * multi_positive_margin_cross_entropy(
                        affect_loss_logits,
                        affect_loss_positive,
                        dim=1,
                        margin=mpm_margin,
                        logit_scale=model.logit_scales["affect"].exp(),
                        label_smoothing=mpm_label_smoothing,
                    )
                    + (1.0 - affect_direction)
                    * multi_positive_margin_cross_entropy(
                        affect_loss_logits,
                        affect_loss_positive,
                        dim=0,
                        margin=mpm_margin,
                        logit_scale=model.logit_scales["affect"].exp(),
                        label_smoothing=mpm_label_smoothing,
                    )
                )
                preference_base_ranking_loss = (
                    (1.0 - factor_mpm_mix["preference"])
                    * preference_standard_ranking_loss
                    + factor_mpm_mix["preference"] * preference_mpm_loss
                )
                affect_base_ranking_loss = (
                    (1.0 - factor_mpm_mix["affect"])
                    * affect_standard_ranking_loss
                    + factor_mpm_mix["affect"] * affect_mpm_loss
                )
                preference_two_way_loss = two_way_multi_label_loss(
                    preference_loss_logits.T,
                    preference_loss_target,
                    positive_temperature=two_way_positive_temperature,
                    negative_temperature=two_way_negative_temperature,
                )
                affect_two_way_loss = two_way_multi_label_loss(
                    affect_loss_logits.T,
                    affect_loss_target,
                    positive_temperature=two_way_positive_temperature,
                    negative_temperature=two_way_negative_temperature,
                )
                preference_ranking_loss = (
                    (1.0 - factor_two_way_mix["preference"])
                    * preference_base_ranking_loss
                    + factor_two_way_mix["preference"] * preference_two_way_loss
                )
                affect_ranking_loss = (
                    (1.0 - factor_two_way_mix["affect"])
                    * affect_base_ranking_loss
                    + factor_two_way_mix["affect"] * affect_two_way_loss
                )
                preference_sigmoid_loss = multi_positive_sigmoid_loss(
                    preference_loss_logits,
                    preference_loss_positive,
                    bias=factor_sigmoid_bias["preference"],
                    candidate_mask=preference_candidate_mask,
                )
                affect_sigmoid_loss = multi_positive_sigmoid_loss(
                    affect_loss_logits,
                    affect_loss_positive,
                    bias=factor_sigmoid_bias["affect"],
                    candidate_mask=affect_candidate_mask,
                )
                preference_ranking_loss = (
                    (1.0 - factor_sigmoid_mix["preference"])
                    * preference_ranking_loss
                    + factor_sigmoid_mix["preference"] * preference_sigmoid_loss
                )
                affect_ranking_loss = (
                    (1.0 - factor_sigmoid_mix["affect"])
                    * affect_ranking_loss
                    + factor_sigmoid_mix["affect"] * affect_sigmoid_loss
                )
                preference_lambda_ndcg_loss = zero
                if factor_lambda_ndcg_mix["preference"] > 0:
                    preference_lambda_ndcg_loss = lambda_ndcg_loss(
                        preference_loss_logits,
                        preference_loss_positive,
                        cutoff=lambda_ndcg_cutoff,
                        sigma=lambda_ndcg_sigma,
                        candidate_mask=preference_candidate_mask,
                    )
                    preference_ranking_loss = (
                        (1.0 - factor_lambda_ndcg_mix["preference"])
                        * preference_ranking_loss
                        + factor_lambda_ndcg_mix["preference"]
                        * preference_lambda_ndcg_loss
                    )
                affect_lambda_ndcg_loss = zero
                if factor_lambda_ndcg_mix["affect"] > 0:
                    affect_lambda_ndcg_loss = lambda_ndcg_loss(
                        affect_loss_logits,
                        affect_loss_positive,
                        cutoff=lambda_ndcg_cutoff,
                        sigma=lambda_ndcg_sigma,
                        candidate_mask=affect_candidate_mask,
                    )
                    affect_ranking_loss = (
                        (1.0 - factor_lambda_ndcg_mix["affect"])
                        * affect_ranking_loss
                        + factor_lambda_ndcg_mix["affect"]
                        * affect_lambda_ndcg_loss
                    )
                preference_smooth_ap_loss = zero
                if factor_smooth_ap_mix["preference"] > 0:
                    preference_scale = model.logit_scales[
                        "preference"
                    ].exp().clamp(max=100.0)
                    preference_smooth_ap_loss = smooth_average_precision_loss(
                        preference_loss_logits / preference_scale,
                        preference_loss_positive,
                        temperature=smooth_ap_temperature,
                        candidate_mask=preference_candidate_mask,
                    )
                    preference_ranking_loss = (
                        (1.0 - factor_smooth_ap_mix["preference"])
                        * preference_ranking_loss
                        + factor_smooth_ap_mix["preference"]
                        * preference_smooth_ap_loss
                    )
                affect_smooth_ap_loss = zero
                if factor_smooth_ap_mix["affect"] > 0:
                    affect_scale = model.logit_scales["affect"].exp().clamp(max=100.0)
                    affect_smooth_ap_loss = smooth_average_precision_loss(
                        affect_loss_logits / affect_scale,
                        affect_loss_positive,
                        temperature=smooth_ap_temperature,
                        candidate_mask=affect_candidate_mask,
                    )
                    affect_ranking_loss = (
                        (1.0 - factor_smooth_ap_mix["affect"])
                        * affect_ranking_loss
                        + factor_smooth_ap_mix["affect"] * affect_smooth_ap_loss
                    )
                preference_circle_loss = zero
                if factor_circle_mix["preference"] > 0:
                    preference_scale = model.logit_scales[
                        "preference"
                    ].exp().clamp(max=100.0)
                    preference_circle_scores = preference_loss_logits / preference_scale
                    preference_circle_loss = (
                        preference_direction
                        * circle_pair_similarity_loss(
                            preference_circle_scores,
                            preference_loss_positive,
                            dim=1,
                            margin=circle_margin,
                            gamma=circle_gamma,
                            candidate_mask=preference_candidate_mask,
                        )
                        + (1.0 - preference_direction)
                        * circle_pair_similarity_loss(
                            preference_circle_scores,
                            preference_loss_positive,
                            dim=0,
                            margin=circle_margin,
                            gamma=circle_gamma,
                        )
                    )
                    preference_ranking_loss = (
                        (1.0 - factor_circle_mix["preference"])
                        * preference_ranking_loss
                        + factor_circle_mix["preference"] * preference_circle_loss
                    )
                affect_circle_loss = zero
                if factor_circle_mix["affect"] > 0:
                    affect_scale = model.logit_scales["affect"].exp().clamp(max=100.0)
                    affect_circle_scores = affect_loss_logits / affect_scale
                    affect_circle_loss = (
                        affect_direction
                        * circle_pair_similarity_loss(
                            affect_circle_scores,
                            affect_loss_positive,
                            dim=1,
                            margin=circle_margin,
                            gamma=circle_gamma,
                            candidate_mask=affect_candidate_mask,
                        )
                        + (1.0 - affect_direction)
                        * circle_pair_similarity_loss(
                            affect_circle_scores,
                            affect_loss_positive,
                            dim=0,
                            margin=circle_margin,
                            gamma=circle_gamma,
                        )
                    )
                    affect_ranking_loss = (
                        (1.0 - factor_circle_mix["affect"])
                        * affect_ranking_loss
                        + factor_circle_mix["affect"] * affect_circle_loss
                    )
                preference_plm_loss = zero
                if factor_plm_mix["preference"] > 0:
                    preference_plm_candidates = partial_label_mask(
                        preference_loss_positive,
                        factor_class_frequency["preference"],
                        total_samples=int(mtg_train_mask.sum()),
                        balance_strength=plm_balance_strength,
                        minimum_keep_probability=plm_minimum_keep_probability,
                    )
                    if preference_candidate_mask is not None:
                        preference_plm_candidates &= preference_candidate_mask
                    preference_plm_positive = (
                        preference_loss_positive.bool() & preference_plm_candidates
                    )
                    preference_plm_loss = (
                        preference_direction
                        * multi_positive_cross_entropy(
                            preference_loss_logits,
                            preference_plm_positive,
                            dim=1,
                            candidate_mask=preference_plm_candidates,
                        )
                        + (1.0 - preference_direction)
                        * multi_positive_cross_entropy(
                            preference_loss_logits,
                            preference_loss_positive,
                            dim=0,
                        )
                    )
                    preference_ranking_loss = (
                        (1.0 - factor_plm_mix["preference"])
                        * preference_ranking_loss
                        + factor_plm_mix["preference"] * preference_plm_loss
                    )
                affect_plm_loss = zero
                if factor_plm_mix["affect"] > 0:
                    affect_plm_candidates = partial_label_mask(
                        affect_loss_positive,
                        factor_class_frequency["affect"],
                        total_samples=int(mtg_train_mask.sum()),
                        balance_strength=plm_balance_strength,
                        minimum_keep_probability=plm_minimum_keep_probability,
                    )
                    if affect_candidate_mask is not None:
                        affect_plm_candidates &= affect_candidate_mask
                    affect_plm_positive = affect_loss_positive.bool() & affect_plm_candidates
                    affect_plm_loss = (
                        affect_direction
                        * multi_positive_cross_entropy(
                            affect_loss_logits,
                            affect_plm_positive,
                            dim=1,
                            candidate_mask=affect_plm_candidates,
                        )
                        + (1.0 - affect_direction)
                        * multi_positive_cross_entropy(
                            affect_loss_logits,
                            affect_loss_positive,
                            dim=0,
                        )
                    )
                    affect_ranking_loss = (
                        (1.0 - factor_plm_mix["affect"]) * affect_ranking_loss
                        + factor_plm_mix["affect"] * affect_plm_loss
                    )
                prior_strength = float(loss_config.get("asl_prior_strength", 1.0))
                preference_asl_logits = (
                    preference_loss_logits + prior_strength * preference_prior[:, None]
                ).T
                affect_asl_logits = (
                    affect_loss_logits + prior_strength * affect_prior[:, None]
                ).T
                preference_gamma = (
                    asl_controllers["preference"].gamma_neg
                    if asl_dynamic
                    else float(loss_config.get("asl_gamma_neg", 4.0))
                )
                affect_gamma = (
                    asl_controllers["affect"].gamma_neg
                    if asl_dynamic
                    else float(loss_config.get("asl_gamma_neg", 4.0))
                )
                asl_kwargs = {
                    "gamma_pos": float(loss_config.get("asl_gamma_pos", 0.0)),
                    "probability_margin": float(loss_config.get("asl_probability_margin", 0.05)),
                    "reduction": str(
                        loss_config.get("asl_reduction", "mean_samples_sum_labels")
                    ),
                    "detach_focal_weight": bool(
                        loss_config.get("asl_detach_focal_weight", True)
                    ),
                }
                preference_asl_loss = asymmetric_loss_with_logits(
                    preference_asl_logits,
                    preference_loss_target,
                    gamma_neg=preference_gamma,
                    **asl_kwargs,
                )
                affect_asl_loss = asymmetric_loss_with_logits(
                    affect_asl_logits,
                    affect_loss_target,
                    gamma_neg=affect_gamma,
                    **asl_kwargs,
                )
                preference_hybrid_loss = (
                    (1.0 - factor_asl_mix["preference"]) * preference_ranking_loss
                    + factor_asl_mix["preference"] * preference_asl_loss
                )
                affect_hybrid_loss = (
                    (1.0 - factor_asl_mix["affect"]) * affect_ranking_loss
                    + factor_asl_mix["affect"] * affect_asl_loss
                )
                preference_db_loss = zero
                if factor_db_mix["preference"] > 0:
                    preference_db_loss = distribution_balanced_loss_with_logits(
                        preference_loss_logits.T,
                        preference_loss_target,
                        factor_class_frequency["preference"],
                        train_samples=factor_db_train_samples["preference"],
                        map_alpha=float(loss_config.get("db_map_alpha", 0.1)),
                        map_beta=float(loss_config.get("db_map_beta", 10.0)),
                        map_gamma=float(loss_config.get("db_map_gamma", 0.2)),
                        negative_scale=float(loss_config.get("db_negative_scale", 2.0)),
                        initial_bias_scale=float(
                            loss_config.get("db_initial_bias_scale", 0.05)
                        ),
                        focal_gamma=float(loss_config.get("db_focal_gamma", 2.0)),
                        focal_balance=float(loss_config.get("db_focal_balance", 2.0)),
                        reduction=str(loss_config.get("db_reduction", "mean")),
                    )
                affect_db_loss = zero
                if factor_db_mix["affect"] > 0:
                    affect_db_loss = distribution_balanced_loss_with_logits(
                        affect_loss_logits.T,
                        affect_loss_target,
                        factor_class_frequency["affect"],
                        train_samples=factor_db_train_samples["affect"],
                        map_alpha=float(loss_config.get("db_map_alpha", 0.1)),
                        map_beta=float(loss_config.get("db_map_beta", 10.0)),
                        map_gamma=float(loss_config.get("db_map_gamma", 0.2)),
                        negative_scale=float(loss_config.get("db_negative_scale", 2.0)),
                        initial_bias_scale=float(
                            loss_config.get("db_initial_bias_scale", 0.05)
                        ),
                        focal_gamma=float(loss_config.get("db_focal_gamma", 2.0)),
                        focal_balance=float(loss_config.get("db_focal_balance", 2.0)),
                        reduction=str(loss_config.get("db_reduction", "mean")),
                    )
                preference_balanced_loss = (
                    (1.0 - factor_db_mix["preference"]) * preference_hybrid_loss
                    + factor_db_mix["preference"] * preference_db_loss
                )
                affect_balanced_loss = (
                    (1.0 - factor_db_mix["affect"]) * affect_hybrid_loss
                    + factor_db_mix["affect"] * affect_db_loss
                )
                preference_dr_loss = zero
                if factor_dr_mix["preference"] > 0:
                    preference_dr_loss = distributionally_robust_multilabel_loss(
                        preference_loss_logits.T,
                        preference_loss_target,
                        pair_scale=float(loss_config.get("dr_pair_scale", 1.0)),
                        negative_scale=float(
                            loss_config.get("dr_negative_scale", 1.0)
                        ),
                        reduction=str(loss_config.get("dr_reduction", "mean")),
                    )
                affect_dr_loss = zero
                if factor_dr_mix["affect"] > 0:
                    affect_dr_loss = distributionally_robust_multilabel_loss(
                        affect_loss_logits.T,
                        affect_loss_target,
                        pair_scale=float(loss_config.get("dr_pair_scale", 1.0)),
                        negative_scale=float(
                            loss_config.get("dr_negative_scale", 1.0)
                        ),
                        reduction=str(loss_config.get("dr_reduction", "mean")),
                    )
                preference_loss = (
                    (1.0 - factor_dr_mix["preference"])
                    * preference_balanced_loss
                    + factor_dr_mix["preference"] * preference_dr_loss
                )
                affect_tag_loss = (
                    (1.0 - factor_dr_mix["affect"]) * affect_balanced_loss
                    + factor_dr_mix["affect"] * affect_dr_loss
                )
                affect_group_chain_loss = cross_group_pseudo_label_loss(
                    affect_logits.T,
                    preference_target,
                    affect_target,
                    affect_group_association,
                    top_k=affect_group_chain_top_k,
                    temperature=affect_group_chain_temperature,
                )
                preference_track_contrastive_loss = zero
                if factor_track_contrastive_weight["preference"] > 0:
                    preference_track_contrastive_loss = multi_label_supervised_contrastive_loss(
                        music_factors["preference"],
                        preference_target,
                        temperature=track_contrastive_temperature,
                        overlap_weighting=track_contrastive_overlap,
                    )
                affect_track_contrastive_loss = zero
                if factor_track_contrastive_weight["affect"] > 0:
                    affect_track_contrastive_loss = multi_label_supervised_contrastive_loss(
                        music_factors["affect"],
                        affect_target,
                        temperature=track_contrastive_temperature,
                        overlap_weighting=track_contrastive_overlap,
                    )
                preference_frozen_track_contrastive_loss = zero
                if factor_frozen_track_contrastive_weight["preference"] > 0:
                    preference_frozen_track_contrastive_loss = (
                        multi_label_cross_view_contrastive_loss(
                            music_factors["preference"],
                            mtg_music,
                            preference_target,
                            temperature=frozen_track_contrastive_temperature,
                            overlap_weighting=frozen_track_contrastive_overlap,
                            normalize_targets=frozen_track_contrastive_normalize_targets,
                        )
                    )
                affect_frozen_track_contrastive_loss = zero
                if factor_frozen_track_contrastive_weight["affect"] > 0:
                    affect_frozen_track_contrastive_loss = (
                        multi_label_cross_view_contrastive_loss(
                            music_factors["affect"],
                            mtg_music,
                            affect_target,
                            temperature=frozen_track_contrastive_temperature,
                            overlap_weighting=frozen_track_contrastive_overlap,
                            normalize_targets=frozen_track_contrastive_normalize_targets,
                        )
                    )

                preference_query, preference_valid = mean_positive_label_embedding(
                    preference_target, preference_text
                )
                affect_query, affect_valid = mean_positive_label_embedding(affect_target, affect_text)
                factor_weights = model.factor_weights()
                affect_preference_weights = factor_weights[1:] / factor_weights[1:].sum()
                joint_scores = (
                    affect_preference_weights[0]
                    * (affect_query @ music_factors["affect"].T)
                    + affect_preference_weights[1]
                    * (preference_query @ music_factors["preference"].T)
                )
                affect_negative, preference_negative = counterfactual_masks(
                    preference_target, affect_target
                )
                valid_joint = preference_valid & affect_valid
                valid_pairs = valid_joint[:, None] & valid_joint[None, :]
                affect_negative &= valid_pairs
                preference_negative &= valid_pairs
                counterfactual_loss = 0.5 * (
                    counterfactual_hard_negative_loss(
                        joint_scores,
                        affect_negative,
                        margin=float(loss_config["counterfactual_margin"]),
                    )
                    + counterfactual_hard_negative_loss(
                        joint_scores,
                        preference_negative,
                        margin=float(loss_config["counterfactual_margin"]),
                    )
                )
                separation_loss = factor_separation_loss(
                    music_factors["affect"], music_factors["preference"]
                )
                text_va_loss = va_criterion(
                    model.predict_va("text", text_va_features), text_va_target
                )
                music_va_loss = va_criterion(
                    model.predict_va("music", music_va_features), music_va_target
                )
                cross_modal_va_loss = soft_cross_modal_va_alignment_loss(
                    model.encode_factor("text", "affect", text_va_features),
                    model.encode_factor("music", "affect", music_va_features),
                    text_va_target,
                    music_va_target,
                    logit_scale=model.logit_scales["affect"].exp(),
                    va_temperature=float(
                        loss_config.get("cross_modal_va_temperature", 0.1)
                    ),
                )
                prior_loss = gate_prior_loss(model, initial_prior)
                losses = {
                    "semantic": semantic_loss,
                    "semantic_standard": semantic_standard_loss,
                    "semantic_modulated": semantic_modulated_loss,
                    "semantic_smooth_target": semantic_smooth_target_loss,
                    "semantic_distillation": semantic_distillation_loss,
                    "preference": preference_loss,
                    "preference_ranking": preference_ranking_loss,
                    "preference_mpm": preference_mpm_loss,
                    "preference_two_way": preference_two_way_loss,
                    "preference_sigmoid": preference_sigmoid_loss,
                    "preference_lambda_ndcg": preference_lambda_ndcg_loss,
                    "preference_smooth_ap": preference_smooth_ap_loss,
                    "preference_circle": preference_circle_loss,
                    "preference_plm": preference_plm_loss,
                    "preference_track_contrastive": preference_track_contrastive_loss,
                    "preference_frozen_track_contrastive": preference_frozen_track_contrastive_loss,
                    "preference_asl": preference_asl_loss,
                    "preference_db": preference_db_loss,
                    "preference_dr": preference_dr_loss,
                    "affect_tag": affect_tag_loss,
                    "affect_ranking": affect_ranking_loss,
                    "affect_mpm": affect_mpm_loss,
                    "affect_two_way": affect_two_way_loss,
                    "affect_sigmoid": affect_sigmoid_loss,
                    "affect_lambda_ndcg": affect_lambda_ndcg_loss,
                    "affect_smooth_ap": affect_smooth_ap_loss,
                    "affect_circle": affect_circle_loss,
                    "affect_plm": affect_plm_loss,
                    "affect_track_contrastive": affect_track_contrastive_loss,
                    "affect_frozen_track_contrastive": affect_frozen_track_contrastive_loss,
                    "affect_asl": affect_asl_loss,
                    "affect_db": affect_db_loss,
                    "affect_dr": affect_dr_loss,
                    "affect_group_chain": affect_group_chain_loss,
                    "text_va": text_va_loss,
                    "music_va": music_va_loss,
                    "counterfactual": counterfactual_loss,
                    "separation": separation_loss,
                    "gate_prior": prior_loss,
                    "cross_modal_va": cross_modal_va_loss,
                }
                loss_weight_keys = {
                    "semantic": "semantic_weight",
                    "semantic_distillation": "semantic_distillation_weight",
                    "preference": "preference_weight",
                    "preference_track_contrastive": "preference_track_contrastive_weight",
                    "preference_frozen_track_contrastive": "preference_frozen_track_contrastive_weight",
                    "affect_tag": "affect_tag_weight",
                    "affect_track_contrastive": "affect_track_contrastive_weight",
                    "affect_frozen_track_contrastive": "affect_frozen_track_contrastive_weight",
                    "affect_group_chain": "affect_group_chain_weight",
                    "text_va": "text_va_weight",
                    "music_va": "music_va_weight",
                    "counterfactual": "counterfactual_weight",
                    "separation": "factor_separation_weight",
                    "gate_prior": "gate_prior_weight",
                    "cross_modal_va": "cross_modal_va_weight",
                }
                total = sum(
                    float(loss_config.get(weight_key, 0.0)) * losses[name]
                    for name, weight_key in loss_weight_keys.items()
                    if float(loss_config.get(weight_key, 0.0)) != 0.0
                )
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite retriever loss seed={seed} epoch={epoch}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            for parameter_group in gradient_parameter_groups:
                torch.nn.utils.clip_grad_norm_(
                    parameter_group, float(training["gradient_clip_norm"])
                )
            scaler.step(optimizer)
            scaler.update()
            if asl_dynamic and factor_asl_mix["preference"] > 0:
                asl_controllers["preference"].update(
                    preference_asl_logits, preference_loss_target
                )
            if asl_dynamic and factor_asl_mix["affect"] > 0:
                asl_controllers["affect"].update(affect_asl_logits, affect_loss_target)
            totals["total"] += float(total.item())
            counts["total"] += 1
            for name, value in losses.items():
                totals[name] += float(value.item())
                if name in {
                    "semantic",
                    "semantic_standard",
                    "semantic_modulated",
                    "semantic_smooth_target",
                    "semantic_distillation",
                }:
                    counts[name] += int(semantic_active)
                else:
                    counts[name] += 1
        scheduler.step()

        pair_val = pair_evaluation(
            model, pair_data, "validation", device, int(training["batch_size_pairs"]), retrieval_ks
        )
        preference_val = tag_evaluation(
            model,
            "preference",
            mtg,
            music_index,
            "validation",
            device,
            int(training["batch_size_mtg"]),
            tag_ks,
        )
        affect_val = tag_evaluation(
            model,
            "affect",
            mtg,
            music_index,
            "validation",
            device,
            int(training["batch_size_mtg"]),
            tag_ks,
        )
        text_va_val = va_evaluation(
            model, "text", text_va_loaders["validation"], device, text_domains["validation"]
        )
        music_va_val = va_evaluation(
            model, "music", music_va_loaders["validation"], device, music_domains["validation"]
        )
        counterfactual_val = counterfactual_factor_metrics(
            model,
            mtg,
            music_index,
            "validation",
            device,
            int(training["batch_size_mtg"]),
            int(config["evaluation"]["counterfactual_queries"]),
            int(config["evaluation"]["counterfactual_seed"]),
        )
        counterfactual_score = float(
            np.mean(
                [
                    counterfactual_val["affect_control_auc"],
                    counterfactual_val["preference_control_auc"],
                ]
            )
        )
        ndcg_key = "ndcg_at_10"
        selection = float(
            np.mean(
                [
                    pair_val["selection_score"],
                    preference_val["macro"][ndcg_key],
                    affect_val["macro"][ndcg_key],
                    text_va_val["selection_score"],
                    music_va_val["selection_score"],
                    counterfactual_score,
                ]
            )
        )
        preference_balanced_score = tag_balanced_selection_score(
            preference_val["macro"][ndcg_key],
            preference_val["macro"]["average_precision"],
        )
        affect_balanced_score = tag_balanced_selection_score(
            affect_val["macro"][ndcg_key],
            affect_val["macro"]["average_precision"],
        )
        if selection_metric in {"retrieval_only", "factorwise_retrieval"}:
            selection = retrieval_selection_score(
                pair_val["selection_score"], preference_val["macro"][ndcg_key],
                affect_val["macro"][ndcg_key],
            )
        elif selection_metric == "factorwise_balanced_retrieval":
            selection = retrieval_selection_score(
                pair_val["selection_score"],
                preference_balanced_score,
                affect_balanced_score,
            )
        elif selection_metric != "validation_composite":
            raise ValueError(f"Unknown validation selection metric: {selection_metric}")
        row = {
            "epoch": epoch,
            **{
                f"loss_{name}": value / max(counts[name], 1)
                for name, value in totals.items()
            },
            "semantic_updates": counts["semantic"],
            "semantic_temperature_modulation": int(semantic_temperature_modulation),
            "semantic_modulation_minimum_temperature": (
                semantic_modulation_minimum_temperature
            ),
            "semantic_modulation_range": semantic_modulation_range,
            "semantic_modulation_schedule_power": semantic_modulation_schedule_power,
            "semantic_smooth_target_mix": semantic_smooth_target_mix,
            "semantic_smooth_target_fraction": semantic_smooth_target_fraction,
            "semantic_smooth_target_text_weight": semantic_smooth_target_text_weight,
            "semantic_smooth_target_temperature": semantic_smooth_target_temperature,
            "pair_bi_mrr": pair_val["selection_score"],
            "preference_ndcg_at_10": preference_val["macro"][ndcg_key],
            "affect_ndcg_at_10": affect_val["macro"][ndcg_key],
            "preference_average_precision": preference_val["macro"][
                "average_precision"
            ],
            "affect_average_precision": affect_val["macro"]["average_precision"],
            "preference_balanced_score": preference_balanced_score,
            "affect_balanced_score": affect_balanced_score,
            "text_va_ccc": text_va_val["selection_score"],
            "music_va_ccc": music_va_val["selection_score"],
            "counterfactual_auc": counterfactual_score,
            "selection_score": selection,
            "semantic_weight": float(model.factor_weights()[0].item()),
            "affect_weight": float(model.factor_weights()[1].item()),
            "preference_weight": float(model.factor_weights()[2].item()),
            "preference_asl_gamma_neg": float(
                asl_controllers["preference"].gamma_neg
                if asl_dynamic
                else loss_config.get("asl_gamma_neg", 4.0)
            ),
            "affect_asl_gamma_neg": float(
                asl_controllers["affect"].gamma_neg
                if asl_dynamic
                else loss_config.get("asl_gamma_neg", 4.0)
            ),
            "preference_asl_gap": float(asl_controllers["preference"].last_gap),
            "affect_asl_gap": float(asl_controllers["affect"].last_gap),
            "preference_asl_mix": factor_asl_mix["preference"],
            "affect_asl_mix": factor_asl_mix["affect"],
            "preference_db_mix": factor_db_mix["preference"],
            "affect_db_mix": factor_db_mix["affect"],
            "preference_dr_mix": factor_dr_mix["preference"],
            "affect_dr_mix": factor_dr_mix["affect"],
            "preference_mpm_mix": factor_mpm_mix["preference"],
            "affect_mpm_mix": factor_mpm_mix["affect"],
            "mpm_margin": mpm_margin,
            "mpm_label_smoothing": mpm_label_smoothing,
            "preference_two_way_mix": factor_two_way_mix["preference"],
            "affect_two_way_mix": factor_two_way_mix["affect"],
            "two_way_positive_temperature": two_way_positive_temperature,
            "two_way_negative_temperature": two_way_negative_temperature,
            "preference_sigmoid_mix": factor_sigmoid_mix["preference"],
            "affect_sigmoid_mix": factor_sigmoid_mix["affect"],
            "preference_lambda_ndcg_mix": factor_lambda_ndcg_mix["preference"],
            "affect_lambda_ndcg_mix": factor_lambda_ndcg_mix["affect"],
            "lambda_ndcg_cutoff": lambda_ndcg_cutoff,
            "lambda_ndcg_sigma": lambda_ndcg_sigma,
            "preference_smooth_ap_mix": factor_smooth_ap_mix["preference"],
            "affect_smooth_ap_mix": factor_smooth_ap_mix["affect"],
            "smooth_ap_temperature": smooth_ap_temperature,
            "preference_circle_mix": factor_circle_mix["preference"],
            "affect_circle_mix": factor_circle_mix["affect"],
            "circle_margin": circle_margin,
            "circle_gamma": circle_gamma,
            "preference_plm_mix": factor_plm_mix["preference"],
            "affect_plm_mix": factor_plm_mix["affect"],
            "plm_balance_strength": plm_balance_strength,
            "plm_minimum_keep_probability": plm_minimum_keep_probability,
            "preference_sigmoid_bias": factor_sigmoid_bias["preference"],
            "affect_sigmoid_bias": factor_sigmoid_bias["affect"],
            "preference_track_contrastive_weight": factor_track_contrastive_weight[
                "preference"
            ],
            "affect_track_contrastive_weight": factor_track_contrastive_weight[
                "affect"
            ],
            "track_contrastive_temperature": track_contrastive_temperature,
            "track_contrastive_overlap": track_contrastive_overlap,
            "preference_frozen_track_contrastive_weight": (
                factor_frozen_track_contrastive_weight["preference"]
            ),
            "affect_frozen_track_contrastive_weight": (
                factor_frozen_track_contrastive_weight["affect"]
            ),
            "frozen_track_contrastive_temperature": (
                frozen_track_contrastive_temperature
            ),
            "frozen_track_contrastive_overlap": frozen_track_contrastive_overlap,
            "frozen_track_contrastive_normalize_targets": int(
                frozen_track_contrastive_normalize_targets
            ),
            "preference_label_to_track_weight": factor_label_to_track_weight[
                "preference"
            ],
            "affect_label_to_track_weight": factor_label_to_track_weight["affect"],
            "preference_false_negative_fraction": factor_false_negative_fraction[
                "preference"
            ],
            "affect_false_negative_fraction": factor_false_negative_fraction["affect"],
            "false_negative_rejection_active": int(
                epoch > false_negative_warmup_epochs
            ),
            "affect_group_chain_weight": affect_group_chain_weight,
            "affect_group_chain_top_k": affect_group_chain_top_k,
            "affect_group_chain_temperature": affect_group_chain_temperature,
            **{
                f"{factor}_temperature": float(
                    model.logit_scales[factor].detach().exp().reciprocal().item()
                )
                for factor in ("semantic", "affect", "preference")
            },
            **{
                f"{factor}_temperature_mode": factor_temperature_control[factor][
                    "mode"
                ]
                for factor in ("semantic", "affect", "preference")
            },
            "preference_labeled_tracks_only": int(
                factor_labeled_only["preference"]
            ),
            "affect_labeled_tracks_only": int(factor_labeled_only["affect"]),
        }
        history.append(row)
        with (output / "history.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"seed={seed} epoch={epoch:03d} loss={row['loss_total']:.4f} "
            f"pair={row['pair_bi_mrr']:.4f} pref={row['preference_ndcg_at_10']:.4f} "
            f"affect={row['affect_ndcg_at_10']:.4f} textVA={row['text_va_ccc']:.4f} "
            f"musicVA={row['music_va_ccc']:.4f} cfAUC={row['counterfactual_auc']:.4f} "
            f"selection={selection:.4f}",
            flush=True,
        )
        factor_scores = {
            "semantic": float(pair_val["selection_score"]),
            "preference": float(
                preference_balanced_score
                if selection_metric == "factorwise_balanced_retrieval"
                else preference_val["macro"][ndcg_key]
            ),
            "affect": float(
                affect_balanced_score
                if selection_metric == "factorwise_balanced_retrieval"
                else affect_val["macro"][ndcg_key]
            ),
        }
        if selection_metric in factorwise_metrics:
            improved = False
            for factor, score in factor_scores.items():
                if score > factor_best_scores[factor]:
                    factor_best_scores[factor] = score
                    factor_best_epochs[factor] = epoch
                    factor_best_states[factor] = capture_factor_state(model, factor)
                    improved = True
            best_score = retrieval_selection_score(
                factor_best_scores["semantic"],
                factor_best_scores["preference"],
                factor_best_scores["affect"],
            )
            best_epoch = max(factor_best_epochs.values())
            if improved:
                torch.save(
                    {
                        "model": assembled_factor_state(model, factor_best_states),
                        "model_config": model_config,
                        "embedding_dim": int(config["backbone"]["embedding_dim"]),
                        "selection_metric": selection_metric,
                        "epoch": best_epoch,
                        "selection_score": best_score,
                        "factor_best_epochs": factor_best_epochs.copy(),
                        "factor_best_validation_scores": factor_best_scores.copy(),
                        "seed": seed,
                        "run_id": config["run_id"],
                        "initial_state_sha256": provenance["initial_state_sha256"],
                    },
                    output / "best.pt",
                )
            bad_epochs = 0 if improved else bad_epochs + 1
        else:
            if selection > best_score:
                best_score, best_epoch = selection, epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "model_config": model_config,
                        "embedding_dim": int(config["backbone"]["embedding_dim"]),
                        "selection_metric": selection_metric,
                        "epoch": epoch,
                        "selection_score": selection,
                        "seed": seed,
                        "run_id": config["run_id"],
                        "initial_state_sha256": provenance["initial_state_sha256"],
                    },
                    output / "best.pt",
                )
            if selection > patience_reference + float(training["early_stopping_min_delta"]):
                patience_reference, bad_epochs = selection, 0
            else:
                bad_epochs += 1
        if bad_epochs >= int(training["early_stopping_patience"]):
            break

    with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    state = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    provenance.update({"best_epoch": best_epoch, "best_validation_score": best_score,
                       "factor_best_epochs": state.get("factor_best_epochs"),
                       "factor_best_validation_scores": state.get("factor_best_validation_scores"),
                       "checkpoint_sha256": sha256(output / "best.pt"),
                       "final_state_sha256": state_sha256(model)})
    if provenance["initial_state_sha256"] == provenance["final_state_sha256"]:
        raise RuntimeError("Training did not change model parameters")
    write_json(output / "provenance.json", provenance)
    splits = ("validation", "test") if evaluate_test else ("validation",)
    results = {}
    for split in splits:
        results[split] = {
            "semantic_pairs": pair_evaluation(
                model, pair_data, split, device, int(training["batch_size_pairs"]), retrieval_ks
            ),
            "preference_tags": tag_evaluation(
                model,
                "preference",
                mtg,
                music_index,
                split,
                device,
                int(training["batch_size_mtg"]),
                tag_ks,
            ),
            "affect_tags": tag_evaluation(
                model,
                "affect",
                mtg,
                music_index,
                split,
                device,
                int(training["batch_size_mtg"]),
                tag_ks,
            ),
            "text_va": va_evaluation(
                model, "text", text_va_loaders[split], device, text_domains[split]
            ),
            "music_va": va_evaluation(
                model, "music", music_va_loaders[split], device, music_domains[split]
            ),
            "counterfactual": counterfactual_factor_metrics(
                model,
                mtg,
                music_index,
                split,
                device,
                int(training["batch_size_mtg"]),
                int(config["evaluation"]["counterfactual_queries"]),
                int(config["evaluation"]["counterfactual_seed"]),
            ),
        }
    result = {
        "model": str(config.get("name", "retriever")),
        "adapter_sharing": str(model_config.get("adapter_sharing", "none")),
        "adapter_kind": str(model_config.get("adapter_kind", "residual")),
        "frozen_adapters": frozen_adapters,
        "selection_metric": str(config["evaluation"]["selection_metric"]),
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "factor_best_epochs": state.get("factor_best_epochs"),
        "factor_best_validation_scores": state.get("factor_best_validation_scores"),
        "test_evaluated": evaluate_test,
        "factor_weights": {
            name: float(value)
            for name, value in zip(
                ("semantic", "affect", "preference"), model.factor_weights().cpu().tolist(), strict=True
            )
        },
        "results": results,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "training" / "current_retriever.yaml"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument(
        "--loss-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Validation-only hyperparameter override recorded in effective_config.json.",
    )
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts" / "runs" / "retriever"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = load_yaml(config_path)
    if args.max_epochs is not None:
        if args.max_epochs <= 0:
            parser.error("--max-epochs must be positive")
        config["training"]["max_epochs"] = int(args.max_epochs)
    for item in args.loss_override:
        if "=" not in item:
            parser.error(f"invalid --loss-override: {item}")
        key, raw_value = item.split("=", 1)
        if key not in config["loss"]:
            parser.error(f"unknown loss key: {key}")
        config["loss"][key] = float(raw_value)
    supervision = config["supervision"]
    pair_data = load_pair_data([resolve(path) for path in supervision["semantic_pairs"]])
    text_va_data = load_va_data([resolve(supervision["text_va"])], "text_embeddings")
    music_va_data = load_va_data(
        [resolve(path) for path in supervision["music_va"]], "audio_embeddings"
    )
    for name, values in (("semantic", pair_data), ("text_va", text_va_data), ("music_va", music_va_data)):
        check_identity_splits(values["global_identities"], values["splits"], name)
    mtg_path = resolve(supervision["mtg_factors"])
    with np.load(mtg_path, allow_pickle=False) as data:
        mtg = {key: data[key] for key in data.files}
    index_path = resolve(supervision["music_index"])
    with np.load(index_path, allow_pickle=False) as data:
        music_index = data["embeddings"].astype(np.float32)
        music_ids = data["track_ids"].astype(str)
    if not np.array_equal(music_ids[mtg["index_positions"]], mtg["track_ids"].astype(str)):
        raise ValueError("Tag supervision does not align with music index track IDs")
    mtg_splits = {str(int(i.removeprefix("track_"))): str(s) for i, s in zip(mtg["track_ids"], mtg["splits"], strict=True)}
    for identity, split in zip(pair_data["global_identities"], pair_data["splits"], strict=True):
        if str(identity).startswith("mtg:"):
            catalog_split = mtg_splits.get(str(int(str(identity)[4:])))
            if catalog_split is None:
                if split != "test":
                    raise ValueError(f"Unindexed Song Describer track must be held out: {identity}")
            elif catalog_split != split:
                raise ValueError(f"Song Describer / MTG split disagreement: {identity}")
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    if any((output / f"seed_{seed}" / "best.pt").exists() for seed in seeds):
        raise FileExistsError("Training output already has checkpoints. Choose a new --output directory.")
    (output / "effective_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    data_paths = [*supervision["semantic_pairs"], supervision["text_va"], *supervision["music_va"], supervision["mtg_factors"], supervision["music_index"]]
    write_json(output / "input_manifest.json", {"files": [{"path": p, "sha256": sha256(resolve(p))} for p in data_paths], "test_used_for_selection": False})
    runs = []
    for seed in seeds:
        runs.append(
            train_seed(
                config,
                pair_data,
                text_va_data,
                music_va_data,
                mtg,
                music_index,
                seed,
                output / f"seed_{seed}",
                evaluate_test=not args.validation_only,
            )
        )
    aggregate = {
        "model": str(config.get("name", "retriever")),
        "seeds": seeds,
        "test_evaluated": not args.validation_only,
        "validation_score_mean": float(np.mean([run["best_validation_score"] for run in runs])),
        "validation_score_std": float(np.std([run["best_validation_score"] for run in runs], ddof=1))
        if len(runs) > 1
        else 0.0,
        "runs": runs,
    }
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "runs"}, indent=2))


if __name__ == "__main__":
    main()
