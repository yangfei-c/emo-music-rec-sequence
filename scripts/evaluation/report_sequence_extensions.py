"""Report supervised VA variants and sparse observed mood tags separately."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys
import subprocess
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from emomm_sequence.data.sequence_tasks import read_jsonl
from emomm_sequence.training.records import write_json
from report_sequence_routes import family_bootstrap, mean_present
from emomm_sequence.data.sequence_va_tasks import VA_ENDPOINT_PROTOCOL

# A declared descriptor proxy, not continuous VA truth or a complete mood taxonomy.
MOOD = {(0.25, 0.25): ("sad", "melancholic"), (0.25, 0.75): (),
        (0.75, 0.25): ("calm", "relaxing", "meditative"), (0.75, 0.75): ("happy", "energetic", "upbeat")}


def paired_va_comparisons(records):
    """Average fitted seeds per request, then bootstrap the shared request families."""
    indexed = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row.get("va_endpoint_protocol") != VA_ENDPOINT_PROTOCOL:
            raise ValueError("VA main report contains a retired endpoint protocol")
        indexed[(row["split"], row["mode"], row["method"])][row["query_id"]].append(row)
    contrasts = []
    independent = ("true_va_mae", "true_endpoint_mae", "true_step_deviation", "observed_preference_tag_match")
    coverage = ("selected_song_joint_coverage", "whole_playlist_covered", "selected_interval_width")
    for split in sorted({r["split"] for r in records}):
        for mode in ("point", "raw_interval", "cqr_interval"):
            for reference in ("a_grid", "a_refine"):
                metrics = ("objective", *independent, *(coverage if mode != "point" else ()))
                contrasts.append(((split, mode, "b_set"), (split, mode, reference), metrics))
        for method in ("a_grid", "a_refine", "b_set"):
            # The two interval objectives have different uncertainty sets: never subtract their J.
            contrasts.append(((split, "cqr_interval", method), (split, "raw_interval", method), (*independent, *coverage)))
    output = []
    for first, second, metrics in contrasts:
        a, b = indexed.get(first, {}), indexed.get(second, {})
        if not a or not b:
            continue
        clusters = defaultdict(list)
        counts = defaultdict(lambda: {"paired_requests": 0, "requests_with_failure": 0})
        for query in sorted(a.keys() & b.keys()):
            rows_a, rows_b = a[query], b[query]
            rows = rows_a + rows_b
            if len({(r["start_id"], r["target_id"], r["library_size"]) for r in rows}) != 1:
                raise ValueError("VA comparison changed endpoints or library")
            example = rows[0]
            suite = example["suite"]
            counts[suite]["paired_requests"] += 1
            counts[suite]["requests_with_failure"] += int(any(r["status"] != "ok" for r in rows))
            for metric in metrics:
                if metric == "objective":
                    va, vb = [float(np.mean([r[metric] if r["status"] == "ok" else 1.0 for r in group]))
                              for group in (rows_a, rows_b)]
                else:
                    if any(r["status"] != "ok" for r in rows):
                        continue
                    va, vb = mean_present(rows_a, metric), mean_present(rows_b, metric)
                if va is not None and vb is not None:
                    clusters[(suite, metric, example["source"], example["query_family_id"])].append(va - vb)
        for suite, metric in sorted({(k[0], k[1]) for k in clusters}):
            sources = defaultdict(dict)
            for (su, me, source, family), values in clusters.items():
                if (su, me) == (suite, metric):
                    sources[source][family] = float(np.mean(values))
            output.append({"split": first[0], "suite": suite, "first": {"mode": first[1], "method": first[2]},
                           "second": {"mode": second[1], "method": second[2]}, "contrast": "first_minus_second",
                           "metric": metric, **counts[suite], **family_bootstrap(sources),
                           "unpaired_requests": len(a.keys() ^ b.keys()),
                           "interval_scope": "request uncertainty conditional on observed fitted seeds and eligible libraries",
                           "failure_rule": "objective=1 for failures; independent metrics require all paired runs successful"})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--profile", default="pilot", choices=["pilot", "full"])
    p.add_argument("--latency-run-id", default="isolated_latency_v1")
    p.add_argument("--va-latency-run-id", default="va_isolated_latency_v1")
    args = p.parse_args()
    if any(not value.replace("_", "").isalnum() for value in (args.latency_run_id, args.va_latency_run_id)):
        p.error("latency run IDs must contain only letters, digits and underscores")
    out = args.root / "reports" / args.profile
    out.mkdir(parents=True, exist_ok=True)
    groups, states, va_records = defaultdict(list), [], []
    for path in (args.root / "evaluations" / args.profile / "va_retrained").glob("*/*/*/seed_*/main/predictions.jsonl"):
        summary = path.parent / "summary.json"
        if not summary.exists():
            continue
        states.append(str(path.parent))
        for row in read_jsonl(path):
            va_records.append(row)
            key = tuple(row[k] for k in ("split", "mode", "method", "source", "num_tracks", "budget"))
            groups[key].append(row)
    metrics = ["true_va_mae", "true_endpoint_mae", "true_step_deviation", "observed_preference_tag_match",
               "selected_song_joint_coverage", "whole_playlist_covered", "selected_interval_width", "inference_seconds"]
    table = []
    for key, rows in sorted(groups.items()):
        row = dict(zip(("split", "mode", "method", "source", "num_tracks", "budget"), key))
        row.update(rows=len(rows), failed=sum(r["status"] != "ok" for r in rows))
        for metric in metrics:
            values = [r[metric] for r in rows if r.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
        table.append(row)
    if table:
        with (out / "va_results.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(out / "va_summary.json", {"completed_runs": states, "tables": table,
                                         "paired_comparisons": paired_va_comparisons(va_records),
                                         "va_endpoint_protocol": VA_ENDPOINT_PROTOCOL,
                                         "playlist_coverage_guarantee": False,
                                         "cross_mode_objective_comparison_valid": False})
    mood = []
    for split in ("validation", "test"):
        task_file = args.root / "tasks" / args.profile / f"evaluate_{split}.jsonl"
        if not task_file.exists():
            continue
        tasks = {r["query_id"]: r for r in read_jsonl(task_file)}
        with np.load(args.root / "labels" / f"mtg_{split}.npz", allow_pickle=False) as z:
            labels = z["affect_targets"]
            names = z["affect_labels"].astype(str).tolist()
            lookup = {str(v): i for i, v in enumerate(z["track_ids"])}
        for path in (args.root / "evaluations" / args.profile / split).glob("*/seed_*/main/predictions.jsonl"):
            if not (path.parent / "summary.json").exists():
                continue
            values, eligible, requested = [], 0, 0
            for row in read_jsonl(path):
                if row["suite"] != "e3" or row["status"] != "ok":
                    continue
                task = tasks[row["query_id"]]
                requested += 2
                for track, coordinate in zip((row["track_ids"][0], row["track_ids"][-1]), (task["start_va"], task["target_va"])):
                    descriptors = MOOD[tuple(coordinate)]
                    if not descriptors:
                        continue
                    columns = [names.index(f"mood/theme---{tag}") for tag in descriptors]
                    eligible += 1
                    values.append(bool(np.any(labels[lookup[track], columns])))
            if values:
                mood.append({"split": split, "method": path.parents[2].name, "seed": path.parents[1].name,
                             "supported_endpoints": eligible, "all_endpoints": requested,
                             "observed_descriptor_tag_match": float(np.mean(values))})
    write_json(out / "mood_tag_diagnostic.json", {"mapping": {str(k): v for k, v in MOOD.items()}, "results": mood,
               "scope": "posthoc exploratory endpoint descriptor proxy, not VA trajectory quality",
               "limitations": ["tense/agitated state unsupported by this tag mapping", "missing positive tags are not confirmed negatives",
                              "an energetic tag alone does not establish positive valence", "not used for training or model selection"]})
    ablations = []
    split = "validation" if args.profile == "pilot" else "test"
    task_file = args.root / "tasks" / args.profile / f"evaluate_{split}.jsonl"
    tasks = {r["query_id"]: r for r in read_jsonl(task_file)}
    for file in (args.root / "evaluations" / args.profile / split).glob("*/seed_*/*/predictions.jsonl"):
        run_id = file.parent.name
        if not (run_id.startswith("ablation_") or run_id in ("changed_planner_exact", "beam_200", "a_no_smooth", "dynamic_k8")):
            continue
        baseline_file = file.parent.parent / "main" / "predictions.jsonl"
        if not (file.parent / "summary.json").exists() or not (baseline_file.parent / "summary.json").exists():
            continue
        baseline_rows = read_jsonl(baseline_file)
        baseline = {r["query_id"]: r for r in baseline_rows}
        observed = read_jsonl(file)
        if len(baseline) != len(baseline_rows) or len({r["query_id"] for r in observed}) != len(observed):
            raise ValueError("duplicate query IDs in an ablation comparison")
        identity_fields = ("query_family_id", "source", "suite", "split", "method", "seed", "num_tracks",
                           "requested_budget", "start_id", "target_id", "library_size")
        counts = defaultdict(lambda: {"paired_requests": 0, "requests_with_failure": 0})
        clusters = defaultdict(list)
        for row in observed:
            base = baseline[row["query_id"]]
            if any(row[field] != base[field] for field in identity_fields):
                raise ValueError("ablation changed the request, endpoints or library contract")
            counts[row["suite"]]["paired_requests"] += 1
            counts[row["suite"]]["requests_with_failure"] += int(row["status"] != "ok" or base["status"] != "ok")
            def common_cost(record):
                if record["status"] != "ok":
                    return 1.0
                if tasks[row["query_id"]]["preference_text_id"] is None:
                    return (2 * record["emotion_distance"] + record["smooth_distance"]) / 3
                return .5 * record["emotion_distance"] + .25 * record["preference_distance"] + .25 * record["smooth_distance"]
            for metric in ("common_frozen_objective", "true_va_mae", "observed_preference_tag_match"):
                value, reference = (common_cost(row), common_cost(base)) if metric == "common_frozen_objective" else (row.get(metric), base.get(metric))
                if value is not None and reference is not None:
                    clusters[(row["suite"], metric, row["source"], row["query_family_id"])].append(value - reference)
        for suite, metric in sorted({(k[0], k[1]) for k in clusters}):
            per_source = defaultdict(dict)
            for (su, me, source, family), values in clusters.items():
                if su == suite and me == metric:
                    per_source[source][family] = float(np.mean(values))
            ablations.append({"run_id": run_id, "method": file.parents[2].name, "seed": file.parents[1].name,
                              "split": split, "suite": suite, "metric": metric,
                              "contrast": "ablation_minus_same_method_main", **counts[suite],
                              **family_bootstrap(per_source)})
    write_json(out / "ablation_summary.json", {"results": ablations,
               "objective_rule": "all generated paths rescored under the ORIGINAL frozen objective, including no-smooth ablation",
               "interval_scope": "request uncertainty conditional on the fitted seed and music library"})
    main_summary = out / "summary.json"
    if main_summary.exists() and json.loads(main_summary.read_text(encoding="utf-8")).get("paired_comparisons"):
        reference_summary = out / "references" / "summary.json"
        split = "validation" if args.profile == "pilot" else "test"
        expected_seeds = [42] if args.profile == "pilot" else [42, 43, 44]
        pending = [f"{method}/seed_{seed}" for method in ("a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce")
                   for seed in (expected_seeds if method in ("b_set", "direct_path", "dynamic_reduce") else [42])
                   if not (args.root / "evaluations" / args.profile / split / method / f"seed_{seed}" / "main" / "summary.json").exists()]
        if not reference_summary.exists() and not pending:
            subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_sequence_reference.py")), "--profile", args.profile,
                            "--root", str(args.root), "--split", "validation" if args.profile == "pilot" else "test"], check=True)
        write_json(out / "reference_status.json", {"pending_primary_runs": pending, "reference_completed": reference_summary.exists()})
        subprocess.run([sys.executable, str(Path(__file__).with_name("diagnose_sequence_labels.py")),
                        "--root", str(args.root), "--profile", args.profile], check=True)
        if not pending:
            subprocess.run([sys.executable, str(Path(__file__).with_name("benchmark_sequence_latency.py")),
                            "--root", str(args.root), "--profile", args.profile, "--defer-if-busy",
                            "--run-id", args.latency_run_id], check=True)
        va_root = args.root / "evaluations" / args.profile / "va_retrained" / split
        if all((va_root / mode / method / "seed_42/main/summary.json").exists()
               for mode in ("point", "raw_interval", "cqr_interval") for method in ("a_grid", "a_refine", "b_set")):
            subprocess.run([sys.executable, str(Path(__file__).with_name("evaluate_sequence_va_latency.py")),
                            "--root", str(args.root), "--profile", args.profile, "--defer-if-busy",
                            "--run-id", args.va_latency_run_id], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("report_sequence_resources.py")),
                        "--root", str(args.root), "--profile", args.profile,
                        "--latency-run-id", args.latency_run_id, "--va-latency-run-id", args.va_latency_run_id], check=True)
        subprocess.run([sys.executable, str(Path(__file__).with_name("plot_sequence_results.py")),
                        "--root", str(args.root), "--profile", args.profile,
                        "--latency-run-id", args.latency_run_id], check=True)
    print(json.dumps({"va_completed_runs": len(states), "va_table_rows": len(table), "mood_rows": len(mood), "ablation_contrasts": len(ablations)}))


if __name__ == "__main__":
    main()
