"""DEAM、PMEmo 或联合音乐 VA 训练入口。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msmmr.experiments import run_music_emotion
from msmmr.training.engine import RunOptions


DATASET_MODE = "joint"  # deam | pmemo | joint
RUN_NAME: str | None = None
RUN = RunOptions(evaluate_test=False)


def main() -> None:
    run_music_emotion(DATASET_MODE, RUN, RUN_NAME)


if __name__ == "__main__":
    main()
