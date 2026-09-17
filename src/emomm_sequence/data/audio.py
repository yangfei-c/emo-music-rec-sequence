"""Shared, deterministic full-track window selection."""

from __future__ import annotations

import numpy as np


def select_front_middle_end(
    waveform: np.ndarray, sample_rate: int, window_seconds: int = 30
) -> np.ndarray:
    """Concatenate up to three fixed front/middle/end windows.

    Tracks shorter than one window are kept intact.  Longer tracks always use
    position-derived starts and never use affect labels or retrieval scores.
    """

    waveform = np.asarray(waveform)
    width = int(sample_rate) * int(window_seconds)
    if waveform.ndim != 1 or width <= 0:
        raise ValueError("waveform must be mono and window size must be positive")
    if len(waveform) <= width:
        return waveform
    starts = [0, max((len(waveform) - width) // 2, 0), max(len(waveform) - width, 0)]
    starts = list(dict.fromkeys(starts))
    return np.concatenate([waveform[start : start + width] for start in starts])
