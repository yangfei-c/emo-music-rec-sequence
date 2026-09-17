"""Run one recommendation request from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emomm_sequence import EmoMMSequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Return five unique ordered tracks for an affect transition."
    )
    parser.add_argument("--text", help="One natural-language sentence parsed by frozen Qwen3:4B")
    parser.add_argument("--current", help="Current affect text")
    parser.add_argument("--target", help="Target affect text")
    parser.add_argument("--preference", default="", help="Optional music preference text")
    parser.add_argument("--method", choices=("ours", "clamp3"), default="ours")
    parser.add_argument("--scope", choices=("all_library", "unseen_library"), default="unseen_library")
    parser.add_argument("--device", default=None, help="cuda or cpu; falls back to CPU if needed")
    parser.add_argument("--config", default="configs/system.yaml")
    parser.add_argument("--no-verify", action="store_true", help="Skip asset SHA-256 checks")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    if args.text:
        from emomm_sequence.runtime import EmoMMSequenceRuntime
        runtime = EmoMMSequenceRuntime(ROOT, config_path=args.config, device=args.device)
        result = runtime.recommend_from_text(args.text, args.method, args.scope)
    else:
        if not args.current or not args.target:
            parser.error("Provide --text or both --current and --target")
        system = EmoMMSequence(root=ROOT, config=args.config, device=args.device, verify_hashes=not args.no_verify)
        result = system.recommend(args.current, args.target, args.preference, method=args.method, scope=args.scope)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output if args.output.is_absolute() else ROOT / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
