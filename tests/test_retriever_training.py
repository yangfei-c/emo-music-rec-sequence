from __future__ import annotations

import pytest
import torch

from emomm_sequence.models.retriever import RetrieverModel, FACTORS, ResidualAdapter
from emomm_sequence.losses.retrieval import (
    AdaptiveAsymmetryController,
    asymmetric_loss_with_logits,
    circle_pair_similarity_loss,
    counterfactual_hard_negative_loss,
    cross_group_label_association,
    cross_group_pseudo_label_loss,
    distribution_balanced_loss_with_logits,
    distributionally_robust_multilabel_loss,
    large_loss_negative_candidate_mask,
    lambda_ndcg_loss,
    multi_positive_cross_entropy,
    multi_positive_margin_cross_entropy,
    multi_positive_sigmoid_loss,
    partial_label_mask,
    multi_label_supervised_contrastive_loss,
    multi_label_cross_view_contrastive_loss,
    two_way_multi_label_loss,
    soft_cross_modal_va_alignment_loss,
    smooth_average_precision_loss,
    symmetric_multi_positive_info_nce,
    symmetric_modulated_multi_positive_info_nce,
    symmetric_similarity_distillation_loss,
    symmetric_intra_modal_soft_target_loss,
)


def test_zero_initialized_adapter_preserves_cosine_geometry() -> None:
    adapter = ResidualAdapter(embedding_dim=8, bottleneck_dim=3, dropout=0.0)
    features = torch.randn(5, 8)
    expected = torch.nn.functional.normalize(features, dim=-1)
    assert torch.allclose(adapter(features), expected, atol=1.0e-7)


def test_adapter_drop_is_training_only_dynamic_ensemble() -> None:
    torch.manual_seed(4)
    adapter = ResidualAdapter(
        embedding_dim=8,
        bottleneck_dim=3,
        dropout=0.0,
        adapter_drop_probability=0.5,
    )
    torch.nn.init.normal_(adapter.up.weight)
    features = torch.randn(4, 8)
    adapter.train()
    outputs = [adapter(features) for _ in range(20)]
    assert any(not torch.allclose(outputs[0], value) for value in outputs[1:])
    adapter.eval()
    first, second = adapter(features), adapter(features)
    torch.testing.assert_close(first, second)


def test_adapter_drop_can_be_configured_per_factor() -> None:
    model = RetrieverModel(
        embedding_dim=8,
        bottleneck_dim=3,
        dropout=0.0,
        adapter_drop_probability=0.4,
        adapter_drop_probabilities={
            "semantic": 0.0,
            "affect": 0.1,
            "preference": 0.05,
        },
    )
    for modality in (model.text_adapters, model.music_adapters):
        assert modality["semantic"].adapter_drop_probability == 0.0
        assert modality["affect"].adapter_drop_probability == 0.1
        assert modality["preference"].adapter_drop_probability == 0.05


def test_multi_positive_margin_nce_uses_uniform_positive_targets() -> None:
    logits = torch.tensor([[2.0, 1.0, -1.0], [0.5, -0.5, 1.5]])
    positives = torch.tensor([[True, True, False], [False, True, False]])
    loss = multi_positive_margin_cross_entropy(
        logits,
        positives,
        margin=0.2,
        logit_scale=2.0,
    )
    adjusted = logits + (~positives).float() * 0.4
    targets = torch.tensor([[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]])
    expected = -(targets * torch.log_softmax(adjusted, dim=1)).sum(dim=1).mean()
    torch.testing.assert_close(loss, expected)


def test_multi_positive_margin_nce_supports_reverse_direction() -> None:
    logits = torch.randn(3, 5, requires_grad=True)
    positives = torch.tensor(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    loss = multi_positive_margin_cross_entropy(
        logits, positives, dim=0, margin=0.05, logit_scale=10.0
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_multi_label_supervised_contrastive_excludes_unlabelled_tracks() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [[1, 0], [1, 0], [0, 1], [0, 0]], dtype=torch.float32
    )
    loss = multi_label_supervised_contrastive_loss(
        embeddings, targets, temperature=0.2
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None
    torch.testing.assert_close(embeddings.grad[-1], torch.zeros(2))


def test_multi_label_supervised_contrastive_supports_jaccard_targets() -> None:
    embeddings = torch.randn(5, 8, requires_grad=True)
    targets = torch.tensor(
        [[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]],
        dtype=torch.float32,
    )
    loss = multi_label_supervised_contrastive_loss(
        embeddings,
        targets,
        temperature=0.1,
        overlap_weighting="jaccard",
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(embeddings.grad).all()


def test_multi_label_cross_view_contrastive_matches_manual_jaccard_loss() -> None:
    query = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], requires_grad=True)
    key = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    targets = torch.tensor([[1, 1], [1, 0], [0, 1]], dtype=torch.float32)
    actual = multi_label_cross_view_contrastive_loss(
        query,
        key,
        targets,
        temperature=0.2,
        overlap_weighting="jaccard",
        normalize_targets=False,
    )
    normalized_query = torch.nn.functional.normalize(query, dim=-1)
    normalized_key = torch.nn.functional.normalize(key, dim=-1)
    log_probabilities = torch.nn.functional.log_softmax(
        normalized_query @ normalized_key.T / 0.2, dim=1
    )
    weights = torch.tensor(
        [[1.0, 0.5, 0.5], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]]
    )
    expected = -(weights * log_probabilities).sum(dim=1).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(query.grad).all()


def test_multi_label_cross_view_contrastive_ignores_unlabelled_tracks() -> None:
    query = torch.randn(4, 6, requires_grad=True)
    key = torch.randn(4, 6)
    targets = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=torch.float32)
    loss = multi_label_cross_view_contrastive_loss(
        query, key, targets, normalize_targets=True
    )
    assert torch.isfinite(loss)
    loss.backward()
    torch.testing.assert_close(query.grad[-1], torch.zeros(6))


def test_two_way_multi_label_loss_matches_author_reference_formula() -> None:
    logits = torch.tensor(
        [[1.0, -0.2, 0.4], [-0.5, 0.8, 0.1], [0.2, -0.3, 1.1]],
        requires_grad=True,
    )
    targets = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.float32)
    actual = two_way_multi_label_loss(
        logits, targets, positive_temperature=4.0, negative_temperature=1.0
    )
    negative_infinity = -100.0
    positive_mask = targets.masked_fill(targets <= 0, negative_infinity).masked_fill(
        targets > 0, 0.0
    )
    negative_mask = targets.masked_fill(targets != 0, negative_infinity).masked_fill(
        targets == 0, 0.0
    )
    positive_class = torch.logsumexp(-logits / 4.0 + positive_mask, dim=0) * 4.0
    negative_class = torch.logsumexp(logits + negative_mask, dim=0)
    positive_sample = torch.logsumexp(-logits / 4.0 + positive_mask, dim=1) * 4.0
    negative_sample = torch.logsumexp(logits + negative_mask, dim=1)
    expected = torch.nn.functional.softplus(negative_class + positive_class).mean()
    expected += torch.nn.functional.softplus(negative_sample + positive_sample).mean()
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()


def test_two_way_multi_label_loss_excludes_unlabelled_rows_and_classes() -> None:
    logits = torch.randn(4, 3, requires_grad=True)
    targets = torch.tensor(
        [[1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=torch.float32
    )
    loss = two_way_multi_label_loss(logits, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    # The third class has no positives and the fourth sample has no positives.
    # They still serve as valid negatives in the orthogonal direction.
    assert logits.grad[3, :2].abs().sum() > 0
    assert logits.grad[:3, 2].abs().sum() > 0


def test_modulated_info_nce_matches_manual_single_positive_formula() -> None:
    text = torch.tensor([[1.0, 0.0], [0.2, 0.8]], requires_grad=True)
    music = torch.tensor([[0.9, 0.1], [0.0, 1.0]])
    positives = torch.eye(2, dtype=torch.bool)
    actual = symmetric_modulated_multi_positive_info_nce(
        text,
        music,
        positives,
        minimum_temperature=0.01,
        modulation_range=0.04,
    )
    normalized_text = torch.nn.functional.normalize(text, dim=-1)
    normalized_music = torch.nn.functional.normalize(music, dim=-1)
    similarities = normalized_text @ normalized_music.T
    temperatures = 0.01 + 0.04 * torch.sqrt((1.0 + similarities) * 0.5)
    logits = similarities / temperatures
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(logits, torch.arange(2))
        + torch.nn.functional.cross_entropy(logits.T, torch.arange(2))
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(text.grad).all()


def test_modulated_info_nce_supports_multiple_positives() -> None:
    text = torch.randn(4, 8, requires_grad=True)
    music = torch.randn(4, 8)
    identities = torch.tensor([0, 0, 1, 2])
    positives = identities[:, None] == identities[None, :]
    loss = symmetric_modulated_multi_positive_info_nce(
        text, music, positives, minimum_temperature=0.03, modulation_range=0.04
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(text.grad).all()


def test_intra_modal_soft_target_loss_is_finite_and_duplicate_aware() -> None:
    text = torch.randn(4, 8, requires_grad=True)
    music = torch.randn(4, 8, requires_grad=True)
    teacher_text = torch.randn(4, 8)
    teacher_music = torch.randn(4, 8)
    identities = torch.tensor([0, 0, 1, 2])
    positives = identities[:, None] == identities[None, :]
    loss = symmetric_intra_modal_soft_target_loss(
        text,
        music,
        teacher_text,
        teacher_music,
        positives,
        logit_scale=12.0,
        teacher_temperature=0.07,
        soft_target_fraction=0.1,
        text_teacher_weight=0.5,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(music.grad).all()


def test_intra_modal_soft_target_loss_rejects_zero_soft_fraction() -> None:
    embeddings = torch.randn(2, 4)
    with pytest.raises(ValueError, match="soft_target_fraction"):
        symmetric_intra_modal_soft_target_loss(
            embeddings,
            embeddings,
            embeddings,
            embeddings,
            torch.eye(2, dtype=torch.bool),
            logit_scale=10.0,
            soft_target_fraction=0.0,
        )


def test_multi_positive_sigmoid_loss_supports_rectangular_masks() -> None:
    logits = torch.randn(3, 7, requires_grad=True)
    positives = torch.zeros(3, 7, dtype=torch.bool)
    positives[0, (0, 2)] = True
    positives[1, (1, 4, 6)] = True
    positives[2, 3] = True
    loss = multi_positive_sigmoid_loss(logits, positives, bias=-3.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_multi_positive_sigmoid_loss_candidate_mask_keeps_positives() -> None:
    logits = torch.zeros(2, 3)
    positives = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=torch.bool)
    candidates = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    masked = multi_positive_sigmoid_loss(
        logits, positives, bias=-2.0, candidate_mask=candidates
    )
    full = multi_positive_sigmoid_loss(logits, positives, bias=-2.0)
    assert masked < full


def test_partial_label_mask_moves_head_and_tail_ratios_toward_mean() -> None:
    positives = torch.tensor(
        [[True, False, False, False], [True, True, True, False]]
    )
    frequency = torch.tensor([10.0, 90.0])
    # Ratios are 1/9 and 9, with a common arithmetic ratio of 41/9.
    # Tail negatives are retained with probability 1/41; head positives with
    # probability 41/81.  The chosen draws expose both decisions exactly.
    draws = torch.tensor(
        [[0.99, 0.01, 0.03, 0.50], [0.40, 0.60, 0.50, 0.99]]
    )
    mask = partial_label_mask(
        positives,
        frequency,
        total_samples=100,
        balance_strength=1.0,
        minimum_keep_probability=0.001,
        random_values=draws,
    )
    expected = torch.tensor(
        [[True, True, False, False], [True, False, True, True]]
    )
    torch.testing.assert_close(mask, expected)


def test_partial_label_mask_strength_zero_keeps_every_label() -> None:
    positives = torch.tensor([[True, False], [False, True]])
    mask = partial_label_mask(
        positives,
        torch.tensor([2.0, 8.0]),
        total_samples=10,
        balance_strength=0.0,
        random_values=torch.full((2, 2), 0.999),
    )
    assert mask.all()


def test_lambda_ndcg_loss_matches_single_pair_logistic_error() -> None:
    logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    positives = torch.tensor([[True, False]])
    actual = lambda_ndcg_loss(logits, positives, cutoff=10, sigma=1.0)
    expected = torch.nn.functional.softplus(torch.tensor(-0.3))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, 1] > 0


def test_lambda_ndcg_loss_ignores_candidates_outside_mask() -> None:
    logits = torch.tensor([[0.3, 0.1, 8.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    candidate = torch.tensor([[True, True, False]])
    actual = lambda_ndcg_loss(
        logits, positives, cutoff=1, candidate_mask=candidate
    )
    expected = torch.nn.functional.softplus(torch.tensor(-0.2))
    torch.testing.assert_close(actual, expected)
    actual.backward()
    torch.testing.assert_close(logits.grad[0, 2], torch.tensor(0.0))


def test_lambda_ndcg_loss_skips_queries_without_both_classes() -> None:
    logits = torch.tensor([[1.0, 0.0], [0.5, -0.5]], requires_grad=True)
    positives = torch.tensor([[True, True], [False, False]])
    loss = lambda_ndcg_loss(logits, positives)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_smooth_ap_matches_one_positive_one_negative_formula() -> None:
    scores = torch.tensor([[0.2, -0.1]], requires_grad=True)
    positives = torch.tensor([[True, False]])
    actual = smooth_average_precision_loss(
        scores, positives, temperature=0.1
    )
    rank = 1.0 + torch.sigmoid(torch.tensor((-0.1 - 0.2) / 0.1))
    expected = 1.0 - 1.0 / rank
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad is not None
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_smooth_ap_rewards_correct_ranking() -> None:
    positives = torch.tensor([[True, False, True, False]])
    correct = smooth_average_precision_loss(
        torch.tensor([[0.8, -0.2, 0.4, -0.5]]), positives, temperature=0.05
    )
    reversed_loss = smooth_average_precision_loss(
        torch.tensor([[-0.2, 0.8, -0.5, 0.4]]), positives, temperature=0.05
    )
    assert correct < reversed_loss


def test_smooth_ap_candidate_mask_zeroes_excluded_gradient() -> None:
    scores = torch.tensor([[0.3, 0.1, 4.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    candidate = torch.tensor([[True, True, False]])
    loss = smooth_average_precision_loss(
        scores, positives, temperature=0.1, candidate_mask=candidate
    )
    loss.backward()
    torch.testing.assert_close(scores.grad[0, 2], torch.tensor(0.0))


def test_circle_loss_matches_single_positive_negative_formula() -> None:
    scores = torch.tensor([[0.4, -0.2]], requires_grad=True)
    positives = torch.tensor([[True, False]])
    margin, gamma = 0.25, 16.0
    alpha_positive = 1.0 + margin - scores[0, 0].detach()
    alpha_negative = (scores[0, 1].detach() + margin).clamp_min(0.0)
    positive_term = -gamma * alpha_positive * (scores[0, 0] - (1.0 - margin))
    negative_term = gamma * alpha_negative * (scores[0, 1] - margin)
    expected = torch.nn.functional.softplus(positive_term + negative_term)
    actual = circle_pair_similarity_loss(
        scores, positives, margin=margin, gamma=gamma
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_circle_loss_supports_reverse_direction() -> None:
    scores = torch.tensor(
        [[0.5, -0.1, 0.2], [-0.2, 0.6, 0.1]], requires_grad=True
    )
    positives = torch.tensor(
        [[True, False, True], [False, True, False]], dtype=torch.bool
    )
    loss = circle_pair_similarity_loss(scores, positives, dim=0, gamma=24.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_circle_loss_candidate_mask_zeroes_excluded_gradient() -> None:
    scores = torch.tensor([[0.3, 0.1, 4.0]], requires_grad=True)
    positives = torch.tensor([[True, False, False]])
    candidate = torch.tensor([[True, True, False]])
    loss = circle_pair_similarity_loss(
        scores, positives, gamma=16.0, candidate_mask=candidate
    )
    loss.backward()
    torch.testing.assert_close(scores.grad[0, 2], torch.tensor(0.0))


def test_retriever_structured_score_shapes_and_weights() -> None:
    model = RetrieverModel(embedding_dim=8, bottleneck_dim=3, dropout=0.0)
    joint, affect, preference, music = (torch.randn(2, 8) for _ in range(4))
    music = torch.randn(7, 8)
    score, components = model.score_structured_query(
        joint, affect, preference, music, return_components=True
    )
    assert score.shape == (2, 7)
    assert set(components) == set(FACTORS)
    assert torch.allclose(model.factor_weights().sum(), torch.tensor(1.0))
    preference_score = model.score_structured_query(
        joint, affect, preference, music, preference_only=True
    )
    assert torch.allclose(preference_score, components["preference"])


def test_zero_semantic_blend_is_exact_frozen_geometry() -> None:
    model = RetrieverModel(
        embedding_dim=8,
        bottleneck_dim=3,
        dropout=0.0,
        semantic_output_blend=0.0,
    )
    features = torch.randn(5, 8)
    torch.testing.assert_close(
        model.encode_factor("text", "semantic", features),
        torch.nn.functional.normalize(features, dim=-1),
    )


def test_shared_adapter_ablation_ties_factor_parameters() -> None:
    model = RetrieverModel(
        embedding_dim=8,
        bottleneck_dim=3,
        dropout=0.0,
        adapter_sharing="across_factors",
    )
    assert model.text_adapters["semantic"] is model.text_adapters["affect"]
    assert model.text_adapters["affect"] is model.text_adapters["preference"]
    assert model.music_adapters["semantic"] is model.music_adapters["preference"]
    assert model.text_adapters["semantic"] is not model.music_adapters["semantic"]


def test_multi_positive_loss_does_not_treat_second_positive_as_negative() -> None:
    logits = torch.tensor([[5.0, 5.0, -5.0], [-5.0, -5.0, 5.0]])
    positives = torch.tensor([[True, True, False], [False, False, True]])
    loss = multi_positive_cross_entropy(logits, positives)
    assert loss.item() < 1.0e-3


def test_large_loss_rejection_masks_only_highest_unobserved_candidates() -> None:
    logits = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0]])
    positives = torch.tensor([[True, False, True, False, False]])
    candidates = large_loss_negative_candidate_mask(
        logits, positives, ignore_fraction=0.34
    )
    assert candidates.tolist() == [[True, False, True, True, True]]
    assert (positives & ~candidates).sum().item() == 0
    full = multi_positive_cross_entropy(logits, positives, dim=1)
    filtered = multi_positive_cross_entropy(
        logits, positives, dim=1, candidate_mask=candidates
    )
    assert filtered < full


def test_multi_positive_candidate_mask_cannot_remove_known_positive() -> None:
    logits = torch.tensor([[1.0, 0.0]])
    positives = torch.tensor([[True, False]])
    try:
        multi_positive_cross_entropy(
            logits,
            positives,
            dim=1,
            candidate_mask=torch.tensor([[False, True]]),
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("removing a known positive must fail")


def test_cross_group_association_is_cosine_normalized_cooccurrence() -> None:
    source = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    target = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
    association = cross_group_label_association(source, target)
    expected = torch.tensor([[1.0, 0.5], [0.5, 1.0]])
    torch.testing.assert_close(association, expected)


def test_cross_group_pseudo_loss_uses_only_missing_target_rows() -> None:
    source = torch.tensor([[1, 0], [0, 1]], dtype=torch.float32)
    observed = torch.tensor([[0, 0], [1, 0]], dtype=torch.float32)
    association = torch.eye(2)
    good = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    bad = torch.tensor([[-4.0, 4.0], [-4.0, 4.0]])
    good_loss = cross_group_pseudo_label_loss(
        good, source, observed, association, top_k=1
    )
    bad_loss = cross_group_pseudo_label_loss(
        bad, source, observed, association, top_k=1
    )
    assert good_loss < bad_loss


def test_asymmetric_loss_matches_reference_formula() -> None:
    logits = torch.tensor([[1.2, -0.7, 0.1], [-1.1, 0.4, 2.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    gamma_neg, gamma_pos, margin = 4.0, 0.0, 0.05
    probability = torch.sigmoid(logits)
    negative_probability = (1.0 - probability + margin).clamp(max=1.0)
    log_likelihood = targets * torch.log(probability.clamp_min(1.0e-8))
    log_likelihood += (1.0 - targets) * torch.log(
        negative_probability.clamp_min(1.0e-8)
    )
    correct_probability = targets * probability + (1.0 - targets) * negative_probability
    exponent = targets * gamma_pos + (1.0 - targets) * gamma_neg
    expected = -(log_likelihood * (1.0 - correct_probability).pow(exponent)).sum()
    actual = asymmetric_loss_with_logits(
        logits,
        targets,
        gamma_neg=gamma_neg,
        gamma_pos=gamma_pos,
        probability_margin=margin,
        reduction="sum",
    )
    torch.testing.assert_close(actual, expected)


def test_distribution_balanced_loss_matches_reference_operations() -> None:
    logits = torch.tensor([[0.4, -0.2, 1.1], [-0.7, 0.8, 0.2]])
    targets = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    frequency = torch.tensor([20.0, 10.0, 5.0])
    train_samples = 40
    alpha, beta, gamma = 0.1, 10.0, 0.2
    negative_scale, bias_scale = 2.0, 0.05

    inverse = frequency.reciprocal()
    repeat_rate = (targets * inverse).sum(dim=1, keepdim=True)
    weight = torch.sigmoid(beta * (inverse.unsqueeze(0) / repeat_rate - gamma)) + alpha
    prevalence = frequency / train_samples
    bias = torch.logit(prevalence) * bias_scale / negative_scale
    adjusted = logits + bias
    adjusted = targets * adjusted + (1.0 - targets) * adjusted * negative_scale
    weight = targets * weight + (1.0 - targets) * weight / negative_scale
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        adjusted, targets, reduction="none"
    )
    expected = (2.0 * (1.0 - torch.exp(-bce)).square() * bce * weight).mean()
    actual = distribution_balanced_loss_with_logits(
        logits,
        targets,
        frequency,
        train_samples=train_samples,
    )
    torch.testing.assert_close(actual, expected)


def test_distributionally_robust_loss_matches_classwise_lsep_formula() -> None:
    logits = torch.tensor([[1.0, -0.5], [-0.2, 0.7], [0.4, -1.1]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    per_class = []
    for column in range(logits.shape[1]):
        positive = logits[targets[:, column].bool(), column]
        negative = logits[~targets[:, column].bool(), column]
        pair_term = torch.exp(-positive).sum() * torch.exp(negative).sum()
        negative_constraint = torch.exp(negative).sum()
        per_class.append(torch.log1p(pair_term + negative_constraint))
    expected = torch.stack(per_class).mean()
    actual = distributionally_robust_multilabel_loss(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_distributionally_robust_loss_matches_classwise_reference() -> None:
    logits = torch.tensor([[0.8, -0.2], [-0.5, 1.1], [0.2, 0.4]])
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    positive = (-logits).masked_fill(~targets.bool(), -torch.inf)
    negative = logits.masked_fill(targets.bool(), -torch.inf)
    expected = torch.logsumexp(
        torch.stack(
            (
                torch.zeros(logits.shape[1]),
                torch.logsumexp(positive, dim=0)
                + torch.logsumexp(negative, dim=0),
                torch.logsumexp(negative, dim=0),
            )
        ),
        dim=0,
    ).mean()
    actual = distributionally_robust_multilabel_loss(logits, targets)
    torch.testing.assert_close(actual, expected)


def test_adaptive_asymmetry_increases_focus_when_gap_is_below_target() -> None:
    controller = AdaptiveAsymmetryController(
        initial_gamma_neg=1.0, target_gap=0.1, step_size=0.5
    )
    logits = torch.tensor([[-2.0, 2.0, 2.0, 2.0]])
    targets = torch.tensor([[1, 0, 0, 0]])
    before = controller.gamma_neg
    after = controller.update(logits, targets)
    assert controller.last_gap < controller.target_gap
    assert after > before


def test_symmetric_multi_positive_loss_and_counterfactual_margin_are_finite() -> None:
    text = torch.eye(3)
    music = torch.eye(3)
    positives = torch.eye(3, dtype=torch.bool)
    contrastive = symmetric_multi_positive_info_nce(
        text, music, positives, logit_scale=10.0
    )
    scores = text @ music.T
    negatives = ~positives
    counterfactual = counterfactual_hard_negative_loss(scores, negatives, margin=0.1)
    assert torch.isfinite(contrastive)
    assert contrastive.item() < 1.0e-3
    assert counterfactual.item() == 0.0


def test_soft_cross_modal_va_alignment_prefers_matching_geometry() -> None:
    text = torch.eye(3)
    music = torch.eye(3)
    va = torch.tensor([[-0.8, -0.8], [0.0, 0.0], [0.8, 0.8]])
    aligned = soft_cross_modal_va_alignment_loss(
        text,
        music,
        va,
        va,
        logit_scale=10.0,
        va_temperature=0.05,
    )
    reversed_music = music.flip(0)
    misaligned = soft_cross_modal_va_alignment_loss(
        text,
        reversed_music,
        va,
        va,
        logit_scale=10.0,
        va_temperature=0.05,
    )
    assert torch.isfinite(aligned)
    assert aligned.item() < misaligned.item()


def test_soft_cross_modal_va_alignment_validates_inputs() -> None:
    embeddings = torch.eye(2)
    va = torch.zeros(2, 2)
    try:
        soft_cross_modal_va_alignment_loss(
            embeddings,
            embeddings,
            va,
            va,
            logit_scale=1.0,
            va_temperature=0.0,
        )
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("non-positive temperature must fail")


def test_similarity_distillation_prefers_teacher_geometry() -> None:
    teacher = torch.eye(4)
    aligned = symmetric_similarity_distillation_loss(
        teacher, teacher, teacher, teacher, temperature=0.1
    )
    shuffled = symmetric_similarity_distillation_loss(
        teacher, teacher.flip(0), teacher, teacher, temperature=0.1
    )
    assert aligned.item() < 1.0e-7
    assert shuffled.item() > aligned.item()
