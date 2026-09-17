from __future__ import annotations

import numpy as np

from emomm_sequence.evaluation.pairs import grouped_retrieval_metrics_from_scores
from emomm_sequence.evaluation.retrieval import label_to_track_metrics


def test_label_to_track_metrics_support_multiple_positives() -> None:
    scores = np.asarray([[0.9, 0.8, 0.1], [0.1, 0.2, 0.9]])
    positives = np.asarray([[True, True, False], [False, False, True]])
    result = label_to_track_metrics(scores, positives, ks=(1, 2))
    assert result["queries"] == 2
    assert result["macro"]["average_precision"] == 1.0
    assert result["macro"]["reciprocal_rank"] == 1.0
    assert result["macro"]["ndcg_at_2"] == 1.0


def test_grouped_score_metrics_support_multiple_captions() -> None:
    scores = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
        ]
    )
    result = grouped_retrieval_metrics_from_scores(
        scores,
        np.asarray(["track-a", "track-a", "track-b"]),
        np.asarray(["track-a", "track-b"]),
        ks=(1, 2),
    )
    assert result["captions"] == 3
    assert result["tracks"] == 2
    assert result["text_to_music"]["recall_at_1"] == 1.0
    assert result["music_to_text"]["recall_at_1"] == 1.0
