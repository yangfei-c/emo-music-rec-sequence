"""Solver-guided warm-up and hard-set policy training, with resumable records."""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import yaml
import psutil

from emomm_sequence.data.sequence_tasks import ROOT, load_config, load_bank, read_jsonl, make_task_problem
from emomm_sequence.inference.sequence_planner import beam_search
from emomm_sequence.models.sequence_selector import build_selector
from emomm_sequence.models.sequence_reduction import build_dynamic_selector
from emomm_sequence.losses.sequence_policy import policy_loss
from emomm_sequence.training.records import environment, sha256, state_sha256, write_json
from emomm_sequence.data.sequence_va_tasks import VAInputs, VASequenceTask, VA_ENDPOINT_PROTOCOL
from emomm_sequence.inference.sequence_va import va_beam_search


def task_selector(config, variant, objective="frozen", va_feature_dimensions=7, **overrides):
    model_config = copy.deepcopy(config)
    if objective != "frozen":
        model_config["assets"]["embedding_dim"] += va_feature_dimensions
    return (build_dynamic_selector(model_config, **overrides) if variant == "dynamic_reduce"
            else build_selector(model_config, variant=variant, **overrides))


def process_identity():
    process = psutil.Process()
    return {"pid": process.pid, "created_utc": datetime.fromtimestamp(process.create_time(), timezone.utc).isoformat()}


def checkpoint_path(root, profile, variant, seed, run_id="main"):
    return Path(root) / "runs" / profile / variant / f"seed_{seed}" / run_id


def load_model(config, root, profile, variant, seed, device, run_id="main"):
    directory = checkpoint_path(root, profile, variant, seed, run_id)
    if (directory / "invalidated.json").exists():
        raise ValueError(f"checkpoint invalidated; inspect {directory / 'invalidated.json'}")
    checkpoint = directory / "best.pt"
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    if sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError("checkpoint differs from validation selection")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if variant == "dynamic_reduce":
        trained_k = payload["manifest"]["config"]["route_b"].get("dynamic_k", 16)
        requested_k = config["route_b"].get("dynamic_k", 16)
        if requested_k != trained_k:
            raise ValueError(f"dynamic_k differs from the selected checkpoint: trained={trained_k}, requested={requested_k}")
    overrides = payload.get("model_overrides", {})
    model = task_selector(config, variant, payload["manifest"].get("objective", "frozen"),
                          payload["manifest"].get("va_feature_dimensions", 6), **overrides).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, selection


def reference_paths(problem, task, config, root, objective="frozen"):
    namespace = "teachers" if objective == "frozen" else f"teachers_va_{objective}_{VA_ENDPOINT_PROTOCOL}"
    path = root / "cache" / namespace / f"{task['query_id']}.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["track_ids"] != problem.ids.tolist():
            raise ValueError("teacher bank identity mismatch")
        for candidate, cost in zip(data["paths"], data["costs"], strict=True):
            if (len(candidate) != problem.n or len(set(candidate)) != problem.n
                    or candidate[0] != problem.start or candidate[-1] != problem.target):
                raise ValueError("cached teacher violates length, identity or endpoint protocol")
            if abs(problem.cost(candidate) - cost) > 1e-7:
                raise ValueError("cached teacher objective changed")
        return data["paths"]
    generated_at = time.perf_counter()
    proposals = []
    c = config["route_b"]
    for width in c["teacher_beam_widths"]:
        r = (va_beam_search if isinstance(problem, VASequenceTask) else beam_search)(problem, range(problem.m), width=width)
        if r.status == "ok":
            proposals.append(r.path)
    base_seed = int(task["query_id"].rsplit("_", 1)[-1]) + 12345
    for i in range(0 if isinstance(problem, VASequenceTask) else c["teacher_perturbed_searches"]):
        r = beam_search(problem, range(problem.m), width=c["teacher_perturbed_beam_width"],
                        perturb_seed=base_seed + i, perturb_scale=c["teacher_gumbel_scale"])
        if r.status == "ok":
            proposals.append(r.path)
    paths = sorted(set(proposals), key=problem.cost)[:c["max_reference_paths_per_episode"]]
    if not paths:
        raise RuntimeError("all reference solvers failed; do not silently omit the episode")
    write_json(path, {"task_id": task["query_id"], "track_ids": problem.ids.tolist(),
                      "teacher_is_exact": False, "paths": paths, "costs": [problem.cost(p) for p in paths],
                      "generation_seconds": time.perf_counter() - generated_at,
                      "endpoint_protocol": VA_ENDPOINT_PROTOCOL if objective != "frozen" else "frozen_shared_endpoints",
                      "generator_source_sha256": sha256(Path(__file__))})
    return paths


def selection_cost(problem, selection, variant, width):
    if variant == "set":
        r = (va_beam_search if isinstance(problem, VASequenceTask) else beam_search)(problem, selection.indices, width=width)
        return r.objective if r.status == "ok" else 1.0
    return problem.cost(selection.indices)


def train(config, root, args):
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    directory = checkpoint_path(root, args.profile, args.variant, args.seed, args.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    taskroot = root / "tasks" / args.profile
    training = read_jsonl(taskroot / "train.jsonl")
    validation = read_jsonl(taskroot / "validation.jsonl")
    if args.limit_tasks:
        training, validation = training[:args.limit_tasks], validation[:min(args.limit_tasks, len(validation))]
    with np.load(root / "cache" / "text_bank.npz", allow_pickle=False) as z:
        texts = {k: z[k] for k in z.files}
    banks = {(s, p): load_bank(root, s, p) for s in ("deam", "pmemo", "merge", "mtg") for p in ("train", "validation")}
    cfg = config["training"]
    objective = getattr(args, "objective", "frozen")
    no_aux_warm = getattr(args, "no_aux_warm", False)
    va_inputs = VAInputs(root, objective) if objective != "frozen" else None
    model_overrides = {"no_set": args.no_set, "no_length": args.no_length}
    model = task_selector(config, args.variant, objective, **model_overrides).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    initial_hash = state_sha256(model)
    signatures = {name: sha256(taskroot / name) for name in ("train.jsonl", "validation.jsonl", "manifest.json")}
    signatures["text_bank"] = sha256(root / "cache" / "text_bank.npz")
    if args.profile == "full":
        for relative in ("src/emomm_sequence/training/sequence_selector.py", "src/emomm_sequence/data/sequence_tasks.py"):
            signatures[relative] = sha256(ROOT / relative)
    # Track source snapshots as well as configuration for each model run.
    for relative in ("src/emomm_sequence/models/sequence_selector.py", "src/emomm_sequence/losses/sequence_policy.py",
                     "src/emomm_sequence/losses/sequence_cost.py", "src/emomm_sequence/inference/sequence_planner.py"):
        signatures[relative] = sha256(ROOT / relative)
    if args.variant == "dynamic_reduce":
        signatures["src/emomm_sequence/models/sequence_reduction.py"] = sha256(ROOT / "src/emomm_sequence/models/sequence_reduction.py")
    if va_inputs is not None:
        for relative in ("manifest.json", "text.npz", *(f"{s}_{p}.npz" for s, p in banks)):
            signatures[f"va/{relative}"] = sha256(va_inputs.directory / relative)
        for relative in ("src/emomm_sequence/data/sequence_va_tasks.py", "src/emomm_sequence/inference/sequence_va.py"):
            signatures[relative] = sha256(ROOT / relative)
    manifest = {"config": config, "variant": args.variant, "seed": args.seed, "model_overrides": model_overrides,
                "limit_tasks": args.limit_tasks, "stage": args.stage, "max_epochs_override": args.max_epochs}
    if objective != "frozen":
        manifest["objective"] = objective
        manifest["supervised_prediction_features"] = True
        manifest["va_feature_dimensions"] = 7
        manifest["va_endpoint_protocol"] = VA_ENDPOINT_PROTOCOL
    if no_aux_warm:
        manifest["no_aux_warm"] = True
    last_file = directory / "last.pt"
    cursor = {"phase": 0, "epoch": 0, "offset": 0}
    best, bad_epochs, global_step = float("inf"), 0, 0
    elapsed_prior = 0.0
    if last_file.exists():
        if not args.resume:
            raise FileExistsError(f"run exists, use --resume or a new --run-id: {directory}")
        state = torch.load(last_file, map_location=device, weights_only=False)
        if state["manifest"] != manifest or state["input_hashes"] != signatures:
            raise ValueError("resume configuration/input/source hashes changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["rng_cuda"]])
        cursor, best, bad_epochs, global_step = state["cursor"], state["best"], state["bad_epochs"], state["global_step"]
        elapsed_prior = state["elapsed_seconds"]
        initial_hash = state["initial_state_sha256"]
    elif args.resume:
        raise FileNotFoundError("cannot resume absent last.pt")
    write_json(directory / "environment.json", environment())
    write_json(directory / "input_hashes.json", signatures)
    write_json(directory / "manifest.json", manifest)
    snapshot = {str(p.relative_to(ROOT)): sha256(p) for package in ("data", "models", "losses", "inference", "training")
                for p in (ROOT / "src" / "emomm_sequence" / package).glob("*.py")}
    for relative, digest in snapshot.items():
        destination = root / "source_snapshots" / digest / Path(relative).name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((ROOT / relative).read_bytes())
    with (directory / "invocations.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({**process_identity(), "recorded_utc": datetime.now(timezone.utc).isoformat(),
                            "resume": bool(args.resume), "source_hashes": snapshot}) + "\n")
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    identity = process_identity()
    begin = time.monotonic()
    pc = config["profiles"][args.profile]
    run_limit = pc.get("max_run_seconds", pc.get("max_training_seconds_per_seed", 28800))
    effective_batch = cfg["microbatch"] * cfg["gradient_accumulation_steps"]
    phases = ["warm", "rl"] if args.stage == "all" else [args.stage]
    def progress(status, **extra):
        write_json(directory / "progress.json", {**identity, "scope": args.profile, "variant": args.variant,
                   "seed": args.seed, "status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
                   "cursor": cursor, "global_step": global_step, "elapsed_seconds": elapsed_prior + time.monotonic() - begin, **extra})
    def save():
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "cursor": dict(cursor),
                   "best": best, "bad_epochs": bad_epochs, "global_step": global_step, "manifest": manifest,
                   "model_overrides": model_overrides, "input_hashes": signatures, "initial_state_sha256": initial_hash,
                   "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   "elapsed_seconds": elapsed_prior + time.monotonic() - begin}
        temporary = directory / "last.pt.tmp"
        torch.save(payload, temporary)
        temporary.replace(last_file)
        return payload
    def inputs(task):
        bank = banks[(task["source"], task["split"])]
        if va_inputs is not None:
            _, problem, _ = va_inputs.make(bank, task, texts)
        else:
            _, problem = make_task_problem(bank, task, texts)
        return problem, torch.from_numpy(problem.x).to(device), torch.from_numpy(problem.query).to(device)
    def validate():
        model.eval()
        costs = defaultdict(list)
        with torch.no_grad():
            for ti, task in enumerate(validation):
                problem, x, query = inputs(task)
                out = model.select(x, query, problem.n, task["budget"], problem.start, problem.target)
                costs[task["source"]].append(selection_cost(problem, out, args.variant, config["planner"]["beam_width"]))
                if ti % 16 == 0:
                    progress("validating", validation_episode=ti, validation_total=len(validation))
        return float(np.mean([np.mean(v) for v in costs.values()])), {k: float(np.mean(v)) for k, v in costs.items()}
    progress("running", trainable_parameters=sum(p.numel() for p in model.parameters()))
    try:
        for phase_i in range(cursor["phase"], len(phases)):
            phase = phases[phase_i]
            max_epochs = args.max_epochs or cfg[f"{phase}_max_epochs"]
            records = training[:pc.get("warm_tasks", len(training))] if phase == "warm" else training
            first_epoch = cursor["epoch"] if cursor["phase"] == phase_i else 0
            for epoch in range(first_epoch, max_epochs):
                offset = cursor["offset"] if cursor["phase"] == phase_i and cursor["epoch"] == epoch else 0
                order = np.random.default_rng(args.seed + phase_i * 10000 + epoch).permutation(len(records))
                losses, sample_costs = [], []
                model.train()
                for chunk_start in range(offset, len(records), effective_batch):
                    indices = order[chunk_start:chunk_start + effective_batch]
                    optimizer.zero_grad(set_to_none=True)
                    for index in indices:
                        task = records[int(index)]
                        problem, x, query = inputs(task)
                        warm_loss = x.sum() * 0
                        if phase == "warm" or not no_aux_warm:
                            teachers = reference_paths(problem, task, config, root, objective)
                            teacher = teachers[epoch % len(teachers)]
                            warm = model.select(x, query, problem.n, task["budget"], problem.start, problem.target, teacher=teacher)
                            warm_loss = warm.warm_loss
                        if phase == "warm":
                            loss = warm_loss
                        else:
                            sampled = model.select(x, query, problem.n, task["budget"], problem.start, problem.target, sample=True)
                            cost = selection_cost(problem, sampled, args.variant, config["planner"]["beam_width"])
                            with torch.no_grad():
                                greedy = model.select(x, query, problem.n, task["budget"], problem.start, problem.target)
                                baseline = selection_cost(problem, greedy, args.variant, config["planner"]["beam_width"])
                            loss = policy_loss(sampled, cost, baseline, warm_loss,
                                               0.0 if no_aux_warm else config["route_b"]["rl_aux_warm_weight"], config["route_b"]["entropy_weight"])
                            sample_costs.append(cost)
                        if not torch.isfinite(loss):
                            raise FloatingPointError("nonfinite training loss")
                        (loss / len(indices)).backward()
                        losses.append(float(loss.detach()))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"], error_if_nonfinite=True)
                    optimizer.step()
                    global_step += 1
                    cursor = {"phase": phase_i, "epoch": epoch, "offset": min(chunk_start + len(indices), len(records))}
                    save()
                    progress("running", phase_name=phase, epoch=epoch, episodes=cursor["offset"], total_episodes=len(records),
                             loss=float(np.mean(losses[-len(indices):])))
                    if time.monotonic() - begin >= run_limit:
                        progress("incomplete_timeout", phase_name=phase, resumable=True)
                        return 2
                score, sources = validate()
                improved = score < best - 1e-6
                bad_epochs = 0 if improved else bad_epochs + 1
                if improved:
                    best = score
                cursor = {"phase": phase_i, "epoch": epoch + 1, "offset": 0}
                payload = save()
                if improved:
                    torch.save(payload, directory / "best.pt")
                    write_json(directory / "selection.json", {"basis": "source_equal_validation_surrogate_only", "metric": best,
                               "phase": phase, "epoch": epoch, "checkpoint_sha256": sha256(directory / "best.pt"),
                               "input_hashes": signatures, "test_used": False})
                row = {"phase": phase, "epoch": epoch, "validation_cost": score, "source_costs": sources,
                       "loss": float(np.mean(losses)) if losses else None,
                       "loss_episodes_this_invocation": len(losses),
                       "loss_covers_full_epoch": len(losses) == len(records),
                       "sample_cost": float(np.mean(sample_costs)) if sample_costs else None,
                       "elapsed_seconds": elapsed_prior + time.monotonic() - begin, "selected": improved}
                with (directory / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if bad_epochs >= cfg["early_stop_patience"]:
                    break
            cursor = {"phase": phase_i + 1, "epoch": 0, "offset": 0}
            bad_epochs = 0
            save()
        final_hash = state_sha256(model)
        if final_hash == initial_hash:
            raise RuntimeError("training finished without a parameter change")
        summary = {"status": "completed", "scope": args.profile, "variant": args.variant, "seed": args.seed,
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "best_validation_cost": best, "initial_state_sha256": initial_hash, "final_state_sha256": final_hash,
                   "parameters_changed": True, "global_step": global_step,
                   "elapsed_seconds": elapsed_prior + time.monotonic() - begin}
        write_json(directory / "summary.json", summary)
        progress("completed", expected_outputs=[str(directory / x) for x in ("best.pt", "selection.json", "summary.json")])
        return 0
    except Exception as exc:
        progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/training/sequence_selector.yaml")
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--variant", choices=["set", "direct_path", "dynamic_reduce"], default="set")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stage", choices=["all", "warm", "rl"], default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--run-id", default="main")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit-tasks", type=int, default=0)
    p.add_argument("--max-epochs", type=int, default=0)
    p.add_argument("--no-set", action="store_true")
    p.add_argument("--no-length", action="store_true")
    p.add_argument("--objective", choices=["frozen", "point", "raw_interval", "cqr_interval"], default="frozen")
    p.add_argument("--no-aux-warm", action="store_true", help="For a genuine RL-only ablation, combine with --stage rl")
    args = p.parse_args()
    entry = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    config, root = load_config(entry.get("protocol", args.config))
    raise SystemExit(train(config, root, args))
