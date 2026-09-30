"""Command entry point for retrieval-adapter training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.training.retriever import capture_factor_state, main


if __name__ == "__main__":
    main()
