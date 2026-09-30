"""Song candidates shared by retrieval and Route2 sequence planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Candidate:
    track_id: str
    artist_id: str
    audio_path: str
    start_relevance: float
    target_relevance: float
    preference_percentile: float
    embedding: np.ndarray
    start_anchor: bool
    target_anchor: bool
