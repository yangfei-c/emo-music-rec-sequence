"""Build the MTG inference catalog from trained tag and emotion checkpoints."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from msmmr.inference import build_music_catalog


CONFIG_PATH = ROOT / "configs" / "local_catalog.yaml"


def main() -> None:
    build_music_catalog(CONFIG_PATH)


if __name__ == "__main__":
    main()
