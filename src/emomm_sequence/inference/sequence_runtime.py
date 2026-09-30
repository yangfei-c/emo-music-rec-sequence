"""Public frozen-route adapter using exactly the benchmark objective and planner."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import yaml

from emomm_sequence.data.sequence_tasks import ROOT, load_config, load_bank
from emomm_sequence.losses.sequence_cost import make_problem
from emomm_sequence.inference.sequence_planner import beam_search
from emomm_sequence.inference.trajectory_candidates import select_candidates, refine
from emomm_sequence.training.sequence_selector import load_model
from emomm_sequence.training.records import sha256


def encoded_recommend(bank, vectors, n, budget, method, *, model=None, device="cpu", width=50,
                      has_preference=True):
    """Label-free core; separate encoding/parser latency from the solver."""
    problem = make_problem(bank.x, bank.ids, vectors[0], vectors[1],
                           vectors[2] if has_preference else None, n)
    budget = min(int(budget), len(bank.ids))
    begin = time.perf_counter()
    if method == "b_set":
        if model is None:
            raise ValueError("B route requires a validation-selected trained checkpoint")
        with torch.inference_mode():
            selected = model.select(torch.from_numpy(problem.x).to(device),
                                    torch.from_numpy(problem.query).to(device), n, budget,
                                    problem.start, problem.target)
        candidates = np.asarray(selected.indices)
        result = beam_search(problem, candidates, width=width)
    elif method in ("a_grid", "a_refine"):
        candidates = select_candidates(problem, budget, method)
        if method == "a_refine":
            candidates, result, _ = refine(problem, candidates, width=width)
        else:
            result = beam_search(problem, candidates, width=width)
    else:
        raise ValueError(method)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    if result.status != "ok":
        raise RuntimeError(f"sequence planner failed: {result.status}")
    path = list(result.path)
    if len(set(bank.groups[path])) != n:
        raise AssertionError("repeated music content")
    return {"status": "ok", "method": method, "num_tracks": n, "candidate_budget": budget,
            "candidate_ids": problem.ids[candidates].tolist(), "track_ids": problem.ids[path].tolist(),
            "content_group_ids": bank.groups[path].tolist(), "parts": problem.parts(result.path),
            "planner_seconds": time.perf_counter() - begin, "search_expansions": result.expansions,
            "supervised_va_used": False, "listener_effect_measured": False}


def recommend(args):
    torch.set_num_threads(4)
    config, root = load_config(args.sequence_config)
    device = args.device or "cpu"
    legacy = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    parsed, parsing_seconds = None, 0.0
    if args.text:
        from emomm_sequence.inference.query import NaturalLanguageQueryParser
        begin = time.perf_counter()
        request = NaturalLanguageQueryParser.from_config(legacy["query_parser"]).parse(args.text)
        parsing_seconds = time.perf_counter() - begin
        parsed = request.as_dict()
        current, target, pref = request.current_affect, request.target_affect, request.music_preference
        n = args.num_tracks if args.num_tracks is not None else request.num_tracks or 5
    else:
        current, target, pref = args.current, args.target, args.preference
        n = args.num_tracks if args.num_tracks is not None else 5
    if not current or not target:
        raise ValueError("current and target affect are required")
    if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= 20:
        raise ValueError("num_tracks must be an integer from 1 to 20")
    prompts = [f"Music that sounds {current.strip().rstrip('.')}.",
               f"Music that sounds {target.strip().rstrip('.')}.",
               f"Music featuring {pref.strip().rstrip('.')}."]
    backbone = ROOT / config["assets"]["backbone"]
    backbone_hash = sha256(backbone)
    if backbone_hash != config["assets"]["backbone_sha256"]:
        raise ValueError("frozen backbone hash mismatch")
    key = hashlib.sha256(json.dumps([prompts, backbone_hash]).encode()).hexdigest()
    cache = root / "cache" / "user_queries" / f"{key}.npz"
    begin = time.perf_counter()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            vectors = z["embeddings"]
    else:
        from emomm_sequence.encoding.text import FrozenTextEncoder
        encoder = FrozenTextEncoder(backbone, legacy["backbone"]["text_model"]).to(device).eval()
        tokens = encoder.tokenizer()(prompts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        vectors = encoder.encode(**{k: v.to(device) for k, v in tokens.items()}).cpu().numpy()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, embeddings=vectors)
        del encoder
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    encoding_seconds = time.perf_counter() - begin
    split = "test" if args.scope == "unseen_library" else "all"
    bank = load_bank(root, args.source, split)
    model, selection = None, None
    if args.method == "b_set":
        model, selection = load_model(config, root, args.model_profile, "set", args.seed, device, args.model_run_id)
    result = encoded_recommend(bank, vectors, n, args.budget, args.method, model=model, device=device,
                               width=config["planner"]["beam_width"], has_preference=bool(pref.strip()))
    result.update(request={"current_affect": current, "target_affect": target, "music_preference": pref,
                           "num_tracks": n}, parsed_request=parsed, source=args.source, split=split,
                  parser_seconds=parsing_seconds, text_encoding_seconds=encoding_seconds,
                  text_cache_key=key, backbone_sha256=backbone_hash, checkpoint_selection=selection,
                  model_profile=args.model_profile if model is not None else None)
    return result
