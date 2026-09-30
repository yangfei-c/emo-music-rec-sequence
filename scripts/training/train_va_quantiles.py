"""Command entry point for music-VA quantile training and CQR calibration."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from emomm_sequence.training.music_va import main


if __name__ == "__main__":
    main()
