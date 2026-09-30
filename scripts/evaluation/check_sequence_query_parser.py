"""Disclosed live Chinese four-field parser checks, separate from algorithm scores."""
import argparse
import json
from pathlib import Path
import re
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.inference.query import NaturalLanguageQueryParser, QueryParsingError
from emomm_sequence.training.records import write_json, sha256

CASES = [
    {"text": "我有些焦虑，希望放松，来5首钢琴曲。", "current": "anxious|anxiety|tense", "target": "relax|calm", "preference": "piano", "number": 5},
    {"text": "最近有些低落，希望开心一点，推荐7首吉他音乐。", "current": "sad|low|down|depress", "target": "happy|happier|cheer", "preference": "guitar", "number": 7},
    {"text": "我现在很开心，继续保持，来3首爵士乐。", "current": "happy|joy|cheer", "target": "happy|joy|cheer", "preference": "jazz", "number": 3},
    {"text": "我很疲惫，想精神一点，来4首电子音乐，不要摇滚。", "current": "tired|fatigue|exhaust", "target": "energ|alert|refresh|awake", "preference": "electronic", "exclusion": "(?:no|not|without|avoid|excluding)\\s+rock", "number": 4},
    {"text": "9月30日，我有些紧张，希望平静，听钢琴。", "current": "tense|nervous|anxious", "target": "calm|peace|relax", "preference": "piano", "number": None},
    {"text": "来5首钢琴曲。", "needs_clarification": True},
    {"text": "我现在焦虑，想放松，来20首不带人声的电子音乐。", "current": "anxious|anxiety|tense", "target": "relax|calm", "preference": "electronic", "exclusion": "instrumental|no vocals|without vocals|no vocal", "number": 20},
    {"text": "今天很开心，保持就好。", "current": "happy|joy|cheer", "target": "happy|joy|cheer", "preference": "^$", "number": None},
]

# Revision after semantic review of the original eight observations. Preserve
# that first report; these add explicit over-extraction checks and two new inputs.
CASES_V2 = [dict(case) for case in CASES]
for case in CASES_V2:
    if "current" in case and "exclusion" not in case:
        case["forbidden_preference"] = "no vocals|instrumental|vocal-free|without vocals|no rock"
CASES_V2[6]["exclusion"] += "|vocal[- ]free"
CASES_V2.extend([
    {"text": "心情不错，继续保持，推荐4首爵士音乐。", "current": "happy|good|positive|cheer", "target": "happy|good|positive|cheer", "preference": "jazz", "number": 4, "forbidden_preference": "no vocals|instrumental|vocal-free|without vocals"},
    {"text": "很疲惫，想清醒一些，给我6首钢琴曲。", "current": "tired|fatigue|exhaust", "target": "alert|awake|refresh|energ", "preference": "piano", "number": 6, "forbidden_preference": "no vocals|instrumental|vocal-free|without vocals"},
])


class CPUParser(NaturalLanguageQueryParser):
    def _request(self, path, payload=None):
        if payload is not None and path == "/api/chat":
            payload = {**payload, "options": {**payload["options"], "num_gpu": 0, "num_thread": 4}, "keep_alive": "2m"}
        return super()._request(path, payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/sequence_routes_v1/reports/query_parser_cases.json"))
    p.add_argument("--revision", type=int, choices=[1, 2], default=1)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve the existing parser observations; use a fresh output path")
    parser = CPUParser(expected_digest="359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7", timeout_seconds=180)
    digest = parser.model_digest()  # Failing availability/digest checks must not count as expected clarification.
    cases = CASES if args.revision == 1 else CASES_V2
    result = {"status": "running", "revision": args.revision, "model_digest": digest, "hardware": "CPU, four threads; GPU left for the active trainer",
              "cases_frozen_before_observations": cases, "cases": [],
              "parser_source_sha256": sha256(Path(__file__).resolve().parents[2] / "src/emomm_sequence/inference/query.py"),
              "evaluation_scope": "disclosed functional examples; revision 2 follows observed revision-1 failure; lexical checks need semantic review; not a population accuracy estimate"}
    write_json(args.output, result)
    for case in cases:
        before = time.perf_counter()
        row = {"input": case["text"]}
        try:
            parsed = parser.parse(case["text"])
            row.update(status="parsed", result=parsed.as_dict())
            checks = {"expected_parse": not case.get("needs_clarification", False)}
            for name, field in (("current", "current_affect"), ("target", "target_affect"), ("preference", "music_preference"), ("exclusion", "music_preference")):
                if name in case:
                    checks[name] = bool(re.search(case[name], getattr(parsed, field), re.IGNORECASE))
            checks["number"] = parsed.num_tracks == case.get("number")
            if "forbidden_preference" in case:
                checks["no_extra_exclusion"] = not bool(re.search(case["forbidden_preference"], parsed.music_preference, re.IGNORECASE))
            row["checks"] = checks
        except QueryParsingError as error:
            row.update(status="needs_clarification", error=str(error),
                       checks={"expected_missing_current": bool(case.get("needs_clarification")) and "当前心情" in str(error)})
        row.update(seconds=time.perf_counter() - before, passed=all(row["checks"].values()))
        result["cases"].append(row)
        write_json(args.output, result)
        print(json.dumps(row, ensure_ascii=True), flush=True)
    result.update(status="completed", passed=sum(r["passed"] for r in result["cases"]), total=len(cases))
    write_json(args.output, result)


if __name__ == "__main__":
    main()
