"""Restore the two public paired-text datasets required for retrieval training."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    ("13939205", "MERGE_Bimodal_Complete.zip", "data/raw/paired/MERGE", "4b4dbe24083a5987e0f37ce4eb1df771"),
    ("10072001", "song_describer.csv", "data/raw/paired/song_describer", "e90e9459c22bfbe69f5462dc1434d573"),
)


def digest(path: Path, algorithm: str = "md5") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main() -> None:
    report = []
    for record, filename, directory, checksum in SOURCES:
        folder = ROOT / directory
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        url = f"https://zenodo.org/api/records/{record}/files/{filename}/content"
        if not path.exists() or digest(path) != checksum:
            partial = path.with_suffix(path.suffix + ".partial")
            started = time.monotonic()
            with requests.get(url, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                written, last = 0, 0.0
                with partial.open("wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        stream.write(block)
                        written += len(block)
                        elapsed = time.monotonic() - started
                        if elapsed - last >= 20:
                            print(f"{filename}: {written / 2**20:.1f} MiB, {elapsed:.0f}s", flush=True)
                            last = elapsed
            actual = digest(partial)
            if actual != checksum:
                raise ValueError(f"Checksum mismatch for {filename}: {actual}")
            partial.replace(path)
        if path.suffix == ".zip":
            destination = (folder / "extracted").resolve()
            destination.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    target = (destination / member.filename).resolve()
                    if not target.is_relative_to(destination):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                archive.extractall(destination)
        report.append({"url": url, "path": path.relative_to(ROOT).as_posix(), "md5": checksum, "sha256": digest(path, "sha256"), "bytes": path.stat().st_size})
        print(f"Verified {filename}", flush=True)
    (ROOT / "data/raw/download_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
