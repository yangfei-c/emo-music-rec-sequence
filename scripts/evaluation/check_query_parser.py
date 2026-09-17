"""Small disclosed live Qwen3:4B smoke suite; not a statistical benchmark."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from emomm_sequence.query import NaturalLanguageQueryParser, QueryParsingError
from emomm_sequence.training.records import write_json

CASES = [
    "我现在很烦",
    "我现在很烦，想慢慢平静下来，最好听钢琴音乐。",
    "来点吉他吧，最近有些低落，希望听完能开心一点。",
    "今天心情很好，想继续保持，听点爵士。",
    "我很疲惫，想精神一点，不要摇滚，最好是电子音乐。",
    "推荐几首钢琴曲。",
]


def main():
    parser = NaturalLanguageQueryParser(expected_digest="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7")
    rows = []
    for text in CASES:
        try:
            row = {"input": text, "result": parser.parse(text).as_dict(), "status": "parsed"}
        except QueryParsingError as error:
            row = {"input": text, "status": "needs_clarification", "error": str(error)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    write_json(ROOT / "results/current_route_20260915/query_parser_smoke.json", {"test_type": "six_disclosed_smoke_examples_not_accuracy_estimate", "cases": rows})


if __name__ == "__main__":
    main()
