"""Verify the frozen assets, catalog alignment, and clean project manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_assets(config: dict) -> list[tuple[str, str]]:
    assets = [
        (config["backbone"]["checkpoint"], config["backbone"]["checkpoint_sha256"]),
        (config["retriever"]["checkpoint"], config["retriever"]["checkpoint_sha256"]),
        (config["text_va"]["checkpoint"], config["text_va"]["checkpoint_sha256"]),
        (config["catalog"]["music"], config["catalog"]["music_sha256"]),
        (config["catalog"]["metadata"], config["catalog"]["metadata_sha256"]),
        (
            config["catalog"]["factor_track_ids"],
            config["catalog"]["factor_track_ids_sha256"],
        ),
        (
            config["affect_direction"]["public_terms"],
            config["affect_direction"]["public_terms_sha256"],
        ),
    ]
    assets.extend((item["path"], item["sha256"]) for item in config["catalog"]["factors"].values())
    relation = config["relation"]
    for seed in relation["seeds"]:
        hashes = relation["checkpoints_sha256"]
        assets.append((relation["checkpoint_template"].format(seed=seed), hashes.get(seed, hashes.get(str(seed)))))
        calibration_hashes = relation["calibration_sha256"]
        calibration_hash = calibration_hashes.get(seed, calibration_hashes.get(str(seed))) if isinstance(calibration_hashes, dict) else calibration_hashes
        assets.append((relation["calibration_template"].format(seed=seed), calibration_hash))
    return assets


def project_manifest() -> list[dict]:
    included = []
    roots = [
        ROOT / "configs",
        ROOT / "scripts",
        ROOT / "src",
        ROOT / "tests",
        ROOT / "results",
        ROOT / "PPT_汇报材料" / "制作脚本",
    ]
    # One detailed method guide; mandatory attribution remains in NOTICE.
    for path in [
        ROOT / "PPT_汇报材料/方法说明/EmoMM-Sequence_完整方法训练与案例说明.md",
        ROOT / "NOTICE",
        ROOT / "pyproject.toml",
        ROOT / "requirements.txt",
    ]:
        if path.is_file():
            included.append(path)
    for base in roots:
        if base.is_dir():
            included.extend(
                path for path in base.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts and path.name != "verification.json"
            )
    return [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(set(included))
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--output", default="results/verification.json")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checked = []
    for relative, expected in expected_assets(config):
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else None
        checked.append(
            {
                "path": relative,
                "bytes": path.stat().st_size if path.is_file() else None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "valid": actual == expected,
            }
        )

    with np.load(ROOT / config["catalog"]["music"], allow_pickle=False) as data:
        catalog_ids = data["track_ids"].astype(str)
        catalog_shape = list(data["embeddings"].shape)
    factor_ids = np.load(ROOT / config["catalog"]["factor_track_ids"], allow_pickle=False).astype(str)
    aligned = bool(np.array_equal(catalog_ids, factor_ids))
    unique = len(set(catalog_ids.tolist())) == len(catalog_ids)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": args.config,
        "config_sha256": sha256(config_path),
        "assets_valid": all(item["valid"] for item in checked),
        "assets": checked,
        "catalog": {
            "shape": catalog_shape,
            "track_ids_unique": unique,
            "factor_rows_aligned_by_track_id": aligned,
        },
        "project_manifest": project_manifest(),
    }
    if not report["assets_valid"] or not aligned or not unique:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "assets_valid": True,
        "catalog_shape": catalog_shape,
        "track_ids_unique": unique,
        "factor_rows_aligned_by_track_id": aligned,
        "output": output.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
