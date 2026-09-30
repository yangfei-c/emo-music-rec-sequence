"""Download the small official MTG consensus-label TSV used for endpoint checks.

The project catalog already contains the relevant MTG audio tracks; this
script fetches only independent human mood labels and verifies their hash.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
URL = (
    "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/"
    "derived/music-classification-annotations/music-classification-annotations-clean.tsv"
)
SHA256 = "9b060b3db785d475870ceb0682241782bf20053d751863fbd42e02cf11088e1d"
DEFAULT_PATH = ROOT / "data/raw/mtg-jamendo/music-classification-annotations-clean.tsv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != SHA256:
            raise ValueError(f"existing annotation file has an unexpected SHA-256: {target}")
        print(f"Already verified: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "EmoMM-Sequence-research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
    actual = hashlib.sha256(content).hexdigest()
    if actual != SHA256:
        raise ValueError(f"downloaded annotation SHA-256 differs: {actual}")
    target.write_bytes(content)
    print(f"Downloaded and verified {len(content)} bytes: {target}")


if __name__ == "__main__":
    main()
