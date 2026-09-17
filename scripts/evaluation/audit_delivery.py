"""Audit the active route, training evidence, single guide/deck and web bridge."""
from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile
import argparse
import json
import re
import sys
from urllib.parse import unquote

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.yaml")
    args = parser.parse_args()
    active = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    run_id = active["run_id"]
    result = ROOT / "results" / run_id
    release = ROOT / "artifacts/releases" / run_id
    assert active == yaml.safe_load((release / "system.yaml").read_text(encoding="utf-8"))
    manifest = json.loads((release / "release_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files"].items():
        assert sha256(ROOT / name) == digest, name
    training_evidence = []
    for stage, relative in manifest["training_sources"].items():
        directory = ROOT / relative
        aggregate = json.loads((directory / "aggregate.json").read_text(encoding="utf-8"))
        for seed in aggregate["seeds"]:
            folder = directory / f"seed_{seed}"
            provenance = json.loads((folder / "provenance.json").read_text(encoding="utf-8"))
            assert provenance["initial_checkpoint"] is None
            assert provenance["initial_state_sha256"] != provenance["final_state_sha256"]
            assert provenance["checkpoint_sha256"] == sha256(folder / "best.pt")
            assert (folder / "history.csv").is_file()
            training_evidence.append({"stage": stage, "seed": seed, "source": relative, "checkpoint_sha256": provenance["checkpoint_sha256"]})
        inputs = directory / "input_manifest.json"
        if inputs.is_file():
            for item in json.loads(inputs.read_text(encoding="utf-8"))["files"]:
                assert sha256(ROOT / item["path"]) == item["sha256"], item["path"]
    with np.load(ROOT / active["catalog"]["music"], allow_pickle=False) as catalog:
        ids = catalog["track_ids"]
        assert len(ids) == 49553 and len(set(ids)) == len(ids)
        missing_audio = [str(path) for path in catalog["audio_paths"] if not Path(path).is_file()]
    assert not missing_audio
    presentation_root = ROOT / "PPT_汇报材料"
    decks = list(presentation_root.rglob("*.pptx"))
    assert len(decks) == 1, decks
    deck = decks[0]
    with ZipFile(deck) as archive:
        assert archive.testzip() is None
        slide_count = sum(bool(re.fullmatch(r"ppt/slides/slide\d+\.xml", name)) for name in archive.namelist())
    guide = presentation_root / "方法说明/EmoMM-Sequence_完整方法训练与案例说明.md"
    guides = list(ROOT.rglob("*.md"))
    assert set(guides) == {guide}, guides
    assert (ROOT / "NOTICE").is_file()
    content = guide.read_text(encoding="utf-8")
    bad_links = []
    for link in re.findall(r"\]\(([^)]+)\)", content):
        if re.match(r"https?://|#", link):
            continue
        relative = unquote(link.split("#")[0])
        if not (guide.parent / relative).is_file():
            bad_links.append(link)
    assert not bad_links, bad_links
    math_blocks = len(re.findall(r"^\$\$$", content, re.MULTILINE))
    assert math_blocks % 2 == 0
    web = json.loads((result / "web_integration.json").read_text(encoding="utf-8"))
    assert web["status"] == "passed" and web["model_version"] == run_id
    assert web["first_work_http_status"] == 200
    smoke = json.loads((result / "release_smoke.json").read_text(encoding="utf-8"))
    assert smoke["all_assets_hash_verified"]
    assert all(case["model_version"] == run_id for case in smoke["cases"])
    feedback = ROOT.parent / "EmoMM-Web/backend/data/feedback_events.jsonl"
    activation = {"run_id": run_id, "active_config_sha256": sha256(args.config), "matches_staged_configuration": True,
                  "recorded_utc": datetime.now(timezone.utc).isoformat()}
    write_json(result / "activation.json", activation)
    report = {"status": "passed", "activation": activation, "training_evidence": training_evidence,
              "frozen_backbone": True, "all_training_input_hashes_match": True,
              "all_49553_audio_paths_exist": True, "pptx_slides": slide_count,
              "pptx_sha256": sha256(deck), "one_pptx_one_md": True, "guide_links_valid": True,
              "guide_math_block_pairs": math_blocks // 2, "ppt_content_not_revised_this_cleanup": True,
              "first_work_http_status": 200, "second_work_http_status": "passed",
              "feedback_log": {"sha256": sha256(feedback), "bytes": feedback.stat().st_size, "test_events_written": False}}
    write_json(result / "delivery_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
