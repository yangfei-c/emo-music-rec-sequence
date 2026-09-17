"""Prepare or run validation-only Ours semantic-preservation experiments."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def build_configs(protocol: dict) -> list[dict]:
    base_path = resolve(protocol["base_training_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base_pairs = list(protocol["semantic_pairs"]["base"])
    view_pairs = list(protocol["semantic_pairs"]["merge_train_views"])
    output_root = Path(protocol["output_root"])
    records = []
    for variant in protocol["variants"]:
        config = copy.deepcopy(base)
        deep_update(config, protocol["fixed_overrides"])
        config["name"] = "Ours"
        config["run_id"] = f"semantic_preservation_20260914_{variant['id']}"
        config["model"]["semantic_output_blend"] = float(
            variant["semantic_output_blend"]
        )
        config["loss"]["semantic_distillation_weight"] = float(
            variant["semantic_distillation_weight"]
        )
        config["supervision"]["semantic_pairs"] = base_pairs + (
            view_pairs if variant["use_merge_views"] else []
        )
        records.append(
            {
                "id": variant["id"],
                "config": config,
                "output": output_root / variant["id"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/experiments/semantic_preservation.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument(
        "--run",
        action="store_true",
        help="Run the prepared validation-only training jobs sequentially.",
    )
    args = parser.parse_args()
    protocol_path = resolve(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    records = build_configs(protocol)
    config_root = resolve(protocol["generated_config_root"])
    config_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol_path.relative_to(ROOT).as_posix(),
        "test_used_for_selection": False,
        "jobs": [],
    }
    for record in records:
        config_path = config_root / f"{record['id']}.yaml"
        if args.prepare and config_path.exists():
            raise FileExistsError(f"prepared config already exists: {config_path}")
        if args.prepare:
            config_path.write_text(
                yaml.safe_dump(
                    record["config"], sort_keys=False, allow_unicode=True
                ),
                encoding="utf-8",
            )
        command = [
            sys.executable,
            str(ROOT / "scripts" / "training" / "train_retriever.py"),
            "--config",
            str(config_path),
            "--validation-only",
            "--output",
            str(resolve(record["output"])),
        ]
        job = {
            "id": record["id"],
            "config": config_path.relative_to(ROOT).as_posix(),
            "output": resolve(record["output"]).relative_to(ROOT).as_posix(),
            "command": command,
        }
        if args.run:
            if not config_path.is_file():
                raise FileNotFoundError(
                    f"run --prepare first; missing {config_path.relative_to(ROOT)}"
                )
            if resolve(record["output"]).exists():
                raise FileExistsError(f"training output already exists: {record['output']}")
            completed = subprocess.run(command, cwd=ROOT, check=False)
            job["returncode"] = completed.returncode
            if completed.returncode != 0:
                manifest["jobs"].append(job)
                (config_root.parent / "execution.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(completed.returncode)
        manifest["jobs"].append(job)

    destination = config_root.parent / (
        "execution.json" if args.run else "plan.json"
    )
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
