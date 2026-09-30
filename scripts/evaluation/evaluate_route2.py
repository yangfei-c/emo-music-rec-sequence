"""Run held-out Route 2 sequence evaluation."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.evaluation.sequence import main


if __name__ == "__main__":
    main()
