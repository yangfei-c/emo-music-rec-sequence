"""MTG 音乐标签训练入口"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.experiments import run_music_tagging
from msmmr.training.engine import RunOptions


TAG_MODE = "full183"  # full183 | genre87 | instrument40 | mood56
RUN = RunOptions(evaluate_test=False)


def main() -> None:
    run_music_tagging(TAG_MODE, RUN)


if __name__ == "__main__":
    main()
