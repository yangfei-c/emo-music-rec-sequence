"""Command entry point for EmoBank text-VA training."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.training.text_va import main


if __name__ == "__main__":
    main()
