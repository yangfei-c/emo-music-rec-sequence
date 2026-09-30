"""Resolve feature paths recorded in experiment manifests."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LEGACY_MERT_ROOT = ROOT / "features" / "mert_clamp3"
MERT_ROOT = ROOT / "artifacts" / "features" / "mert_clamp3"


def resolve_feature_path(value: str | Path) -> Path:
    """Keep historical manifest bytes valid after moving local MERT tokens."""
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        return MERT_ROOT / path.relative_to(LEGACY_MERT_ROOT)
    except ValueError:
        return path
