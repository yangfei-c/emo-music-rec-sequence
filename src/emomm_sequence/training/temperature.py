"""Temperature-control schedules for contrastive retrieval training."""

from __future__ import annotations


def controlled_temperature(
    mode: str,
    *,
    start: float,
    end: float,
    epoch: int,
    total_epochs: int,
) -> float | None:
    """Return a fixed/scheduled temperature, or ``None`` when it is learned."""

    if mode not in {"learned", "fixed", "linear"}:
        raise ValueError("temperature mode must be learned, fixed, or linear")
    if start <= 0 or end <= 0:
        raise ValueError("temperature endpoints must be positive")
    if total_epochs <= 0 or not 1 <= epoch <= total_epochs:
        raise ValueError("epoch must be in [1, total_epochs]")
    if mode == "learned":
        return None
    if mode == "fixed" or total_epochs == 1:
        return float(start)
    progress = (epoch - 1) / (total_epochs - 1)
    return float(start + progress * (end - start))
