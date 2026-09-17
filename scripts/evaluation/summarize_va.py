"""Summarize independent text/music VA runs without selecting on test results."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.training.records import write_json


def stats(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = ROOT / "artifacts/runs" / args.run_id
    text = json.loads((root / "text_va/aggregate.json").read_text(encoding="utf-8"))
    music = json.loads((root / "music_va/aggregate.json").read_text(encoding="utf-8"))
    result = {"run_id": args.run_id, "text_va_selected_seed": text["selected_seed"], "text_target_scale": [-1, 1], "music_target_scale": [0, 1],
              "text_va": {key: stats([run["results"]["test"][key] for run in text["runs"]]) for key in text["runs"][0]["results"]["test"]},
              "music_va": {source: {key: stats([run["results"]["test"]["va"][source][key] for run in music["runs"]])
                                       for key in music["runs"][0]["results"]["test"]["va"][source]} for source in ("deam", "pmemo")},
              "relations": {key: stats([run["results"]["test"]["relations"]["overall"][key] for run in music["runs"]])
                            for key in ("accuracy_v", "accuracy_a", "macro_f1_v", "macro_f1_a", "nll_v", "nll_a", "count")},
              "best_epochs": {"text_va": {run["seed"]: run["best_epoch"] for run in text["runs"]},
                              "music_va": {run["seed"]: run["best_epoch"] for run in music["runs"]}},
              "test_used_for_selection": False,
              "source_files": [f"artifacts/runs/{args.run_id}/{stage}/aggregate.json" for stage in ("text_va", "music_va")]}
    write_json(ROOT / "results" / args.run_id / "va_summary.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
