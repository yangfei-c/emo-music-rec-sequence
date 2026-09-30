"""Plot observed paired effects with source data and rendered PDF checks."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def isolated_cost_points(root, profile, points, run_id="isolated_latency_v1"):
    """Join full-main quality to completed, balanced timing measurements."""
    summary_file = root / "reports" / profile / run_id / "summary.json"
    if not summary_file.exists():
        return None
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    if summary["status"] != "completed" or summary["overlap_samples"]:
        raise ValueError("isolated timing is incomplete or overlaps project computation")
    protocol = summary["fingerprint"]
    attempt = Path(summary["attempt"])
    measurements = [json.loads(line) for line in (attempt / "measurements.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    tasks = [json.loads(line) for line in (attempt / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    digest = hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()
    if digest != protocol["selected_tasks_sha256"] or len(measurements) != summary["measurements"]:
        raise ValueError("timing manifest or measurement count changed after completion")
    expected = Counter((method, task["query_id"], repetition)
                       for method in protocol["methods"] for task in tasks
                       for repetition in range(1, protocol["repeats"] + 1))
    actual = Counter((row["method"], row["query_id"], row["timing_repeat"]) for row in measurements)
    if actual != expected:
        raise ValueError("timing measurements are incomplete, duplicated or unbalanced")
    split = "validation" if profile == "pilot" else "test"
    for method, expected_hash in protocol["main_predictions_sha256"].items():
        file = root / "evaluations" / profile / split / method / f"seed_{protocol['fitted_seed']}" / "main" / "predictions.jsonl"
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("main quality changed after independent timing")
    selected_tasks = [task for task in tasks if task["suite"] == "e2" and task["num_tracks"] == 5]
    selected_points = [point for point in points if point["seed"] == protocol["fitted_seed"]]
    expected_points = {(method, task["budget"]) for method in protocol["methods"] for task in selected_tasks}
    if {(point["method"], point["requested_budget"]) for point in selected_points} != expected_points:
        raise ValueError("main quality and independent timing method/budget groups differ")
    result = []
    for point in selected_points:
        subset = [row for row in measurements if row["suite"] == "e2" and row["num_tracks"] == 5
                  and row["method"] == point["method"] and row["requested_budget"] == point["requested_budget"]]
        if not subset or any(row["status"] != "ok" or not np.isfinite(row["total_seconds"])
                             or row["total_seconds"] <= 0 for row in subset):
            raise ValueError("missing, failed or invalid timing measurements")
        result.append({**point, "median_total_ms": 1000 * float(np.median([row["total_seconds"] for row in subset])),
                       "timing_unique_requests": len({row["query_id"] for row in subset}),
                       "timing_measurements": len(subset), "timing_sources": sorted({row["source"] for row in subset}),
                       "timing_summary": str(summary_file), "timing_device": protocol["device"]})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    p.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    p.add_argument("--latency-run-id", default="isolated_latency_v1")
    p.add_argument("--figure-tools", type=Path, default=Path.home() / ".agents/skills/nature-figure/scripts")
    args = p.parse_args()
    if not args.latency_run_id.replace("_", "").isalnum():
        p.error("--latency-run-id must contain only letters, digits and underscores")
    sys.path.insert(0, str(args.figure_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    report = args.root / "reports" / args.profile
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .8, "savefig.facecolor": "white"})
    generated = []
    names = {"random": "Random", "three_route": "Three-route", "independent": "Independent", "a_grid": "A-grid",
             "a_refine": "A-refine", "b_set": "B-set", "direct_path": "Direct path", "dynamic_reduce": "Dynamic reduction"}
    def save(fig, name, source, caption):
        base = output / name
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(base) + ".alignment.json",
                                           overlay_svg=str(base) + ".alignment.svg", tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5, strict=True)
        for extension in ("pdf", "svg"):
            fig.savefig(str(base) + "." + extension)
        fig.savefig(str(base) + ".png", dpi=600)
        plt.close(fig)
        (output / f"{name}.source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
        (output / f"{name}.caption.md").write_text(caption + "\n", encoding="utf-8")
        with (output / f"{name}.qa.log").open("w", encoding="utf-8") as stream:
            for script, tail in (("audit_pdf_text.py", ["--min-pt", "5"]),
                                 ("audit_figure_collisions.py", ["--json-out", str(base) + ".collision.json", "--overlay-pdf", str(base) + ".collision.pdf"])):
                result = subprocess.run([sys.executable, str(args.figure_tools / script), str(base) + ".pdf", *tail],
                                        stdout=stream, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"figure QA failed for {name}: {script}")
        generated.append(name)
    def save_cost_plot(points, name, caption):
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        fig.subplots_adjust(left=.13, right=.71, bottom=.19, top=.88)
        colors = {"random": "#999999", "three_route": "#767676", "independent": "#444444", "a_grid": "#6E9FB7",
                  "a_refine": "#286B8E", "b_set": "#B17837", "direct_path": "#6A8B7B", "dynamic_reduce": "#80738F"}
        markers = dict(zip(names, ("x", "+", "v", "o", "s", "D", "^", "P")))
        for method in names:
            rows = [r for r in points if r["method"] == method]
            if not rows:
                continue
            # Each point remains a fitted seed/budget observation.
            for seed in sorted({r["seed"] for r in rows}):
                group = sorted([r for r in rows if r["seed"] == seed], key=lambda r: r["requested_budget"])
                ax.plot([r["median_total_ms"] for r in group], [r["source_equal_va_mae"] for r in group],
                        marker=markers[method], color=colors[method], linewidth=1, markersize=4,
                        label=names[method] if seed == min(r["seed"] for r in rows) else None)
        if any(not np.isfinite(r["median_total_ms"]) or r["median_total_ms"] <= 0
               or not np.isfinite(r["source_equal_va_mae"]) for r in points):
            raise ValueError("quality-cost axes require finite values and strictly positive times")
        ax.set_xscale("log")
        ax.set_xlabel("Median total request time (ms)", labelpad=8)
        ax.set_ylabel("Source-equal true VA MAE (lower is better)", labelpad=8)
        ax.set_title("Quality and computation on E2 (N = 5)", pad=12)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
        save(fig, name, points, caption)
    main_summary = json.loads((report / "summary.json").read_text(encoding="utf-8"))
    comparison = [r for r in main_summary["paired_comparisons"] if r["suite"] == "e2" and r.get("metric", "objective") in ("objective", "true_va_mae") and r.get("ci95")]
    if comparison:
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
        fig.subplots_adjust(left=.18, right=.98, bottom=.24, top=.84, wspace=.75)
        for ax, metric, title, letter in zip(axes, ("objective", "true_va_mae"),
                                            ("Shared surrogate cost", "Independent VA error"), ("a", "b")):
            rows = [r for r in comparison if r.get("metric", "objective") == metric]
            for y, row in enumerate(rows):
                mean, lo, hi = row["mean"], *row["ci95"]
                ax.hlines(y, lo, hi, color="#286B8E", linewidth=1.2)
                ax.plot(mean, y, "o", color="#286B8E", markersize=4)
            ax.axvline(0, color="#777777", linewidth=.8, linestyle="--")
            ax.set_yticks(range(len(rows)), ["vs " + names.get(r["contrast"].removeprefix("b_set_minus_"),
                                                            r["contrast"]) for r in rows])
            ax.set_ylim(-.6, max(len(rows) - .4, .6))
            ax.invert_yaxis()
            ax.set_title(title, pad=12)
            ax.set_xlabel("B minus comparison (lower favors B)", labelpad=8)
            ax.text(-.23, 1.15, letter, transform=ax.transAxes, fontsize=10, weight="bold")
        save(fig, "paired_path_quality", comparison,
             "Paired E2 differences on completed runs. Source-equal means and 95% percentile bootstrap intervals over request families (2,000 draws); the same family draw is shared across sources. Seeds are averaged before request resampling, so intervals do not estimate training-seed variability. All paired eligible E2 requests are included. Surrogate cost and independent true-VA error answer different parts of the quality claim. No p-value or corrected significance claim is made; this is a pilot when the profile is pilot.")
        points = []
        split = "validation" if args.profile == "pilot" else "test"
        for file in (args.root / "evaluations" / args.profile / split).glob("*/seed_*/main/predictions.jsonl"):
            if not (file.parent / "summary.json").exists():
                continue
            records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            records = [r for r in records if r["suite"] == "e2" and r["num_tracks"] == 5]
            if not records:
                continue
            if any(r["status"] != "ok" for r in records):
                raise ValueError("quality-cost plot requires an explicit failure treatment before plotting incomplete paths")
            for budget in sorted({r["requested_budget"] for r in records}):
                selected = [r for r in records if r["requested_budget"] == budget]
                source_values = [np.mean([r["true_va_mae"] for r in selected if r["source"] == source]) for source in sorted({r["source"] for r in selected})]
                points.append({"method": selected[0]["method"], "seed": selected[0]["seed"], "requested_budget": budget,
                               "median_total_ms": 1000 * float(np.median([r["total_seconds"] for r in selected])),
                               "source_equal_va_mae": float(np.mean(source_values)), "requests": len(selected)})
        if points:
            save_cost_plot(points, "quality_cost",
                 "Quality-computation tradeoff on all completed E2 requests with N=5, chosen as the canonical five-song case before B outcomes were available. Each point is one fitted seed and requested budget (32 or 64); lines connect budgets within that seed. Independent VA error is averaged equally across sources. Time is the median observed request total including objective setup and excluding cached text encoding. Dynamic reduction uses per-step K=16; the plot compares measured cost rather than asserting identical cumulative candidate counts. Pilot timings were collected during development with other CPU work possible and should be remeasured in an isolated timing pass for publication. Other lengths and failures remain in the full result tables; this is a descriptive diagnostic without population intervals.")
            isolated = isolated_cost_points(args.root, args.profile, points, args.latency_run_id)
            if isolated:
                save_cost_plot(isolated, "quality_cost_isolated",
                    "Quality-computation comparison for E2 with N=5. Vertical coordinates use every completed main E2 request at this length, with true VA error averaged equally across sources. Horizontal coordinates use the median of repeated measurements on a fixed stratified subset selected independently of scores and latency. The source-data file records separate quality-request, timing-request and measurement counts. Only the timing protocol's fitted seed is shown; lines connect requested budgets 32 and 64. One warmup pass is excluded. Project training/evaluation workers were monitored for overlap; unrelated OS activity remains possible. Time includes measured objective setup plus candidate/search time and excludes encoding, checkpoint loading, disk I/O and metric computation. Dynamic reduction uses K=16 per step. This descriptive pilot plot does not estimate training-seed or population uncertainty; repeated timings are not additional scientific samples.")
    ablation_size = [7.2, 3.2]
    ablation_file = report / "ablation_summary.json"
    if ablation_file.exists():
        ablation_names = {"ablation_no_set": "No set interaction", "ablation_no_length": "No length input",
                          "ablation_warm_only": "Warm only", "ablation_rl_only": "RL only"}
        panels = [("e2", "common_frozen_objective", "Surrogate cost (E2)", 1000., "Difference x 1,000\nLower favours variant"),
                  ("e2", "true_va_mae", "True VA error (E2)", 1000., "Difference x 1,000\nLower favours variant"),
                  ("e3", "observed_preference_tag_match", "Tag match (E3)", 100., "Difference (pp)\nHigher favours variant")]
        records = [r for r in json.loads(ablation_file.read_text(encoding="utf-8"))["results"]
                   if r["run_id"] in ablation_names and r["method"] == "b_set"]
        keys = sorted({(r["run_id"], r["seed"]) for r in records},
                      key=lambda item: (list(ablation_names).index(item[0]), item[1]))
        if keys:
            index = {(r["run_id"], r["seed"], r["suite"], r["metric"]): r for r in records}
            if len(index) != len(records):
                raise ValueError("duplicate ablation estimates in the figure input")
            ablation_size = [7.2, max(3.2, 1.0 + .25 * len(keys))]
            fig, axes = plt.subplots(1, 3, figsize=ablation_size, sharey=True)
            fig.subplots_adjust(left=.20, right=.98, bottom=.28, top=.84, wspace=.35)
            multiple_seeds = len({seed for _, seed in keys}) > 1
            labels = [ablation_names[run] + (" / " + seed if multiple_seeds else "") for run, seed in keys]
            axes[0].set_yticks(range(len(keys)), labels)
            plotted = []
            for ax, (suite, metric, title, scale, xlabel), letter in zip(axes, panels, "abc"):
                for y, (run_id, seed) in enumerate(keys):
                    row = index.get((run_id, seed, suite, metric))
                    if row is None or not row.get("ci95"):
                        ax.text(.5, y, "Not estimable", transform=ax.get_yaxis_transform(),
                                ha="center", va="center", color="#767676", fontsize=7)
                        continue
                    values = np.array([row["mean"], *row["ci95"]], dtype=float) * scale
                    if not np.isfinite(values).all() or values[1] > values[2]:
                        raise ValueError("invalid ablation effect or interval")
                    mean, lo, hi = values
                    ax.hlines(y, lo, hi, color="#286B8E", linewidth=1.2)
                    ax.plot(mean, y, "o", color="#286B8E", markersize=4)
                    plotted.append({**row, "plot_scale": scale})
                ax.axvline(0, color="#777777", linewidth=.8, linestyle="--")
                ax.set_ylim(len(keys) - .4, -.6)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.set_title(title, pad=12)
                ax.set_xlabel(xlabel, labelpad=8)
                ax.text(-.15, 1.15, letter, transform=ax.transAxes, fontsize=10, weight="bold")
            save(fig, "ablation_effects", {"source_sha256": hashlib.sha256(ablation_file.read_bytes()).hexdigest(),
                 "row_keys": keys, "plotted": plotted, "panels": panels},
                 "Effects of completed component and training-stage ablations relative to full B. Each row retains its fitted seed; points are source-equal paired means and horizontal lines are 95% request-family percentile bootstrap intervals (2,000 draws). E2 cost and true-VA differences are multiplied by 1,000; E3 observed preference-tag differences are percentage points. Lower values favor the ablation in panels a/b, and higher values favor it in c. Source data record paired-request and failure counts; repetitions over lengths, budgets and phrasings are clustered by request family. Missing estimates are marked, and unfinished variants have no invented points. Intervals condition on the fitted models and libraries, do not estimate training-seed variability, and are not adjusted for multiple comparisons. Models use their own validation-selected checkpoints and have different realized training budgets; this is not an equal-update module-speed comparison. Other suites and metrics remain in the full tables.")
    # VA coverage: each curve is a source, which exposes unsupported source transfer.
    va_file = report / "va_summary.json"
    if va_file.exists():
        va = json.loads(va_file.read_text(encoding="utf-8"))
        rows = [r for r in va["tables"] if r["method"] == "a_refine" and r["mode"] in ("raw_interval", "cqr_interval") and r["source"] in ("deam", "pmemo", "merge")]
        if rows:
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
            fig.subplots_adjust(left=.10, right=.98, bottom=.27, top=.78, wspace=.35)
            colors = {"deam": "#286B8E", "pmemo": "#A87A42", "merge": "#767676"}
            display_names = {"deam": "DEAM", "pmemo": "PMEmo", "merge": "MERGE"}
            points = []
            for ax, metric, title, letter in zip(axes, ("selected_song_joint_coverage", "whole_playlist_covered"),
                                                ("Selected-song coverage", "Whole-playlist coverage"), ("a", "b")):
                for source, color in colors.items():
                    values = []
                    for mode in ("raw_interval", "cqr_interval"):
                        selected = [r for r in rows if r["source"] == source and r["mode"] == mode]
                        value = float(np.average([r[metric] for r in selected], weights=[r["rows"] for r in selected]))
                        values.append(value)
                        points.append({"source": source, "mode": mode, "metric": metric, "value": value,
                                       "requests": sum(r["rows"] for r in selected)})
                    ax.plot([0, 1], values, "o-", color=color, markersize=4, linewidth=1.2, label=display_names[source])
                ax.set_xticks([0, 1], ["Raw interval", "CQR interval"])
                ax.set_xlim(-.15, 1.15)
                ax.set_ylim(0, 1)
                ax.set_ylabel("Empirical fraction")
                ax.set_title(title, pad=12)
                ax.text(-.20, 1.15, letter, transform=ax.transAxes, fontsize=10, weight="bold")
            handles, legend_names = axes[0].get_legend_handles_labels()
            fig.legend(handles, legend_names, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .025))
            save(fig, "selected_coverage", points,
                 "A-refine with frozen supervised VA predictions: empirical coverage on the policy-only validation libraries for DEAM/PMEmo and the validation library for MERGE. Each plotted point uses all eligible completed requests, weighted by request count across lengths and budgets. The left panel averages within-playlist joint V/A song coverage; the right records whether every song in a playlist is covered. MERGE has no source-specific calibration and uses the maximum original correction. Lines show source-specific descriptive aggregates, without population confidence intervals or a 90% playlist guarantee. Point predictions have no interval and are excluded from this interval-specific figure. B-VA results belong to the separate paired comparison when complete.")
    (output / "manifest.json").write_text(json.dumps({"figures": generated, "profile": args.profile,
         "backend": "python_matplotlib", "dimensions_inches": {"paired_path_quality": [7.2, 3.2], "quality_cost": [7.2, 3.5], "quality_cost_isolated": [7.2, 3.5], "selected_coverage": [7.2, 3.1], "ablation_effects": ablation_size},
         "isolated_latency_figure": "generated" if "quality_cost_isolated" in generated else "pending_completed_timing",
         "requested_latency_run_id": args.latency_run_id,
         "rendered_qa": "PDF font and collision tools completed; human-style visual inspection must also be recorded",
         "data_scope": "completed runs only; absent methods are not invented"}, indent=2), encoding="utf-8")
    print(json.dumps({"generated": generated}))


if __name__ == "__main__":
    main()
