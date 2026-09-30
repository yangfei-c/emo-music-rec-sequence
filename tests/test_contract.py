from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/system_route2.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_frozen_route2_runtime_contract() -> None:
    config = _config()
    route = config["route2"]

    assert config["backbone"]["frozen"] is True
    assert config["retriever"]["top_per_route"] == 100
    assert config["retriever"]["factor_weights"] == [0.25, 0.40, 0.35]
    assert config["endpoint"]["start_top"] == 20
    assert config["endpoint"]["target_top"] == 20
    assert route["model"] == "joint_axis_cqr_quantile_regressor"
    assert route["search"] == "length_conditioned_full_path_interval_dp_with_beam"
    assert route["calibration_alpha"] == 0.1
    assert route["default_tracks"] == 5
    assert route["max_tracks"] == 20
    assert route["lambda_retrieval"] == 1.0
    assert route["beam_width"] == 50
    assert "relation" not in config
    assert "search" not in config


def test_route2_asset_set_and_calibration_binding() -> None:
    config = _config()
    route = config["route2"]
    paths = [
        config["backbone"]["checkpoint"],
        config["retriever"]["checkpoint"],
        config["text_va"]["checkpoint"],
        config["catalog"]["music"],
        config["catalog"]["metadata"],
        config["catalog"]["factor_track_ids"],
        config["affect_direction"]["public_terms"],
        route["checkpoint"],
        route["calibration"],
    ]
    paths.extend(item["path"] for item in config["catalog"]["factors"].values())
    assert all((ROOT / path).is_file() for path in paths)

    calibration = json.loads((ROOT / route["calibration"]).read_text(encoding="utf-8"))
    assert calibration["checkpoint_sha256"] == route["checkpoint_sha256"]
    assert calibration["alpha"] == route["calibration_alpha"]
    assert calibration["run_id"] == config["run_id"]
    assert calibration["fitted_partition"] == "calibration"
    assert set(calibration["source_corrections"]) == {"deam", "pmemo"}
