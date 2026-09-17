from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_runtime_contract() -> None:
    config = yaml.safe_load((ROOT / "configs/system.yaml").read_text(encoding="utf-8"))
    assert config["retriever"]["top_per_route"] == 100
    assert config["endpoint"]["start_top"] == 20
    assert config["endpoint"]["target_top"] == 20
    assert config["search"] == {
        "objective": "mean_adjacent_edge",
        "sequence_length": 5,
        "search_width": 50,
        "preference_threshold": 0.80,
        "endpoint_threshold": 0.50,
        "edge_threshold": 0.40,
        "fallback": "preference_then_target",
        "deterministic_tie_break": True,
    }
    assert config["backbone"]["frozen"] is True
    assert config["endpoint"]["matcher"] == "Ours"
    assert config["relation"]["model"] == "calibrated_regression_ensemble"


def test_catalog_asset_set_is_complete() -> None:
    config = yaml.safe_load((ROOT / "configs/system.yaml").read_text(encoding="utf-8"))
    paths = [
        config["backbone"]["checkpoint"],
        config["retriever"]["checkpoint"],
        config["text_va"]["checkpoint"],
        config["catalog"]["music"],
        config["catalog"]["metadata"],
        config["catalog"]["factor_track_ids"],
        config["affect_direction"]["public_terms"],
    ]
    paths.extend(item["path"] for item in config["catalog"]["factors"].values())
    relation = config["relation"]
    for seed in relation["seeds"]:
        paths.append(relation["checkpoint_template"].format(seed=seed))
        paths.append(relation["calibration_template"].format(seed=seed))
    assert all((ROOT / path).is_file() for path in paths)
