from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from emomm_sequence.encoding.clamp3 import Clamp3TextAudioEncoder
from emomm_sequence.retrieval.model import RetrieverModel
from emomm_sequence.retrieval.segments import query_conditioned_segment_similarity


class _IdentityAudioTransformer(nn.Module):
    def forward(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        del attention_mask
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def _small_encoder() -> Clamp3TextAudioEncoder:
    encoder = Clamp3TextAudioEncoder.__new__(Clamp3TextAudioEncoder)
    nn.Module.__init__(encoder)
    encoder.audio_model = _IdentityAudioTransformer()
    encoder.audio_proj = nn.Linear(768, 768)
    with torch.no_grad():
        encoder.audio_proj.weight.copy_(torch.eye(768))
        encoder.audio_proj.bias.fill_(0.01)
    encoder.max_audio_length = 8
    encoder.normalize = True
    return encoder


def test_segment_extraction_keeps_global_path_and_removes_boundaries() -> None:
    encoder = _small_encoder()
    tokens = torch.randn(2, 5, 768)
    mask = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    global_embedding = encoder.encode_audio(tokens, mask)
    manual_mean = torch.stack((tokens[0, :4].mean(0), tokens[1].mean(0)))
    expected = torch.nn.functional.normalize(encoder.audio_proj(manual_mean), dim=-1)
    torch.testing.assert_close(global_embedding, expected)

    segments, segment_mask = encoder.encode_audio_segments(tokens, mask)
    assert segment_mask.tolist() == [
        [False, True, True, False, False],
        [False, True, True, True, False],
    ]
    torch.testing.assert_close(
        torch.linalg.vector_norm(segments[segment_mask], dim=-1),
        torch.ones(int(segment_mask.sum())),
    )


def test_query_conditioned_pooling_finds_matching_window() -> None:
    queries = torch.tensor([[1.0, 0.0]])
    segments = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    scores, attention = query_conditioned_segment_similarity(
        queries, segments, mask, temperature=0.05, top_k=1
    )
    assert scores.shape == (1, 2)
    assert attention.shape == (1, 2, 2)
    assert scores[0, 0] > scores[0, 1]
    torch.testing.assert_close(attention.sum(-1), torch.ones(1, 2))


def test_dense_semantic_zero_weight_is_exact_historical_score() -> None:
    model = RetrieverModel(embedding_dim=4, bottleneck_dim=2, dropout=0.0)
    text = torch.randn(3, 4)
    music = torch.randn(5, 4)
    expected = model.factor_similarity("semantic", text, music)
    actual = model.semantic_similarity(text, music, local_weight=0.0)
    torch.testing.assert_close(actual, expected)


def test_dense_semantic_branch_reuses_the_semantic_adapters() -> None:
    model = RetrieverModel(embedding_dim=2, bottleneck_dim=1, dropout=0.0)
    text = torch.tensor([[1.0, 0.0]])
    music = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    segments = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [-1.0, 0.0]],
        ]
    )
    mask = torch.ones(2, 2, dtype=torch.bool)
    score, components = model.semantic_similarity(
        text,
        music,
        music_segments=segments,
        music_segment_mask=mask,
        local_weight=1.0,
        segment_temperature=0.05,
        segment_top_k=1,
        return_components=True,
    )
    torch.testing.assert_close(score, components["local"])
    assert score[0, 0] > score[0, 1]
