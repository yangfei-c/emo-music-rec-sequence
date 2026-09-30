"""Plot complete paired budget evidence and separately measured inference cost."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

METHODS = ("a_grid", "a_refine", "b_set", "direct_path", "dynamic_reduce")
BUDGETS = (32, 64, 128, 256)
LENGTHS = (5, 10)
SOURCES = {"e2": ("deam", "pmemo", "merge"), "e3": ("mtg",)}
NAMES = {"a_grid": "A-grid", "a_refine": "A-refine", "b_set": "B-set",
         "direct_path": "Direct path", "dynamic_reduce": "Dynamic (K=16)"}
COLORS = {"a_grid": "#6E9FB7", "a_refine": "#286B8E", "b_set": "#B17837",
          "direct_path": "#6A8B7B", "dynamic_reduce": "#80738F"}
MARKERS = {32: "o", 64: "s", 128: "^", 256: "D"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_points(summary, timing, measurements, timed_tasks):
    """Join complete source-equal quality to balanced repeated latency records."""
    if summary["status"] != "completed" or summary["failed"]:
        raise ValueError("complete quality and an explicit treatment of all failures are required")
    protocol = timing["fingerprint"]
    if (timing["status"] != "completed" or timing["overlap_samples"] or protocol["fitted_seed"] != 42
            or set(protocol["methods"]) != set(METHODS) or protocol["repeats"] != 5 or protocol["warmup_passes"] != 1
            or protocol["per_stratum"] != 4
            or protocol["main_run_id"] != summary["run_id"] or protocol["source_tasks_sha256"] != summary["task_sha256"]):
        raise ValueError("incompatible, overlapping or incomplete independent timing")
    strata = Counter((t["suite"], t["source"], t["num_tracks"], t["budget"]) for t in timed_tasks)
    expected_strata = {(suite, source, n, budget) for suite, sources in SOURCES.items()
                       for source in sources for n in LENGTHS for budget in BUDGETS}
    if (set(strata) != expected_strata or set(strata.values()) != {4}
            or len({t["query_id"] for t in timed_tasks}) != len(timed_tasks)):
        raise ValueError("fixed timing strata are incomplete or duplicated")
    expected = Counter((method, task["query_id"], repeat) for method in METHODS for task in timed_tasks for repeat in range(1, 6))
    actual = Counter((r["method"], r["query_id"], r["timing_repeat"]) for r in measurements)
    if actual != expected or len(measurements) != timing["measurements"]:
        raise ValueError("duplicate, missing or unbalanced timing records")
    for row in measurements:
        if row["status"] != "ok" or not np.isfinite(row["total_seconds"]) or row["total_seconds"] <= 0:
            raise ValueError("log timing axes require finite positive successful observations")
    table = {(r["method"], r["suite"], r["source"], r["num_tracks"], r["requested_budget"]): r for r in summary["tables"]}
    expected_cells = {(method, suite, source, n, budget) for method in METHODS for suite, sources in SOURCES.items()
                      for source in sources for n in LENGTHS for budget in BUDGETS}
    if set(table) != expected_cells or len(table) != len(summary["tables"]):
        raise ValueError("quality table has missing or duplicate method/source/length/budget cells")
    points = []
    for method in METHODS:
        for n in LENGTHS:
            for budget in BUDGETS:
                for suite, sources in SOURCES.items():
                    metric = "true_va_mae" if suite == "e2" else "observed_preference_tag_match"
                    quality = [table[method, suite, source, n, budget] for source in sources]
                    if any(r["failed"] or r[f"{metric}_observed_requests"] != r["requests"] or
                           r[metric] is None or not np.isfinite(r[metric]) for r in quality):
                        raise ValueError("quality curve would silently omit failed or unobserved requests")
                    subset = [r for r in measurements if (r["method"], r["suite"], r["num_tracks"], r["requested_budget"]) == (method, suite, n, budget)]
                    by_source = Counter(r["source"] for r in subset)
                    if set(by_source) != set(sources) or len(set(by_source.values())) != 1:
                        raise ValueError("timing sources must remain balanced")
                    points.append({"method": method, "suite": suite, "num_tracks": n, "requested_budget": budget,
                                   "metric": metric, "source_equal_quality": float(np.mean([r[metric] for r in quality])),
                                   "median_total_ms": float(np.median([r["total_seconds"] for r in subset])) * 1000,
                                   "quality_requests": sum(r["requests"] for r in quality),
                                   "timing_requests": len({r["query_id"] for r in subset}), "measurements": len(subset),
                                   "timing_request_family_clusters": len({r["query_family_id"] for r in subset}),
                                   "effective_budget_by_source": {r["source"]: r["budget"] for r in quality},
                                   "saturated_sources": [r["source"] for r in quality if r["saturated"]]})
    comparisons = [r for r in summary["paired_comparisons"] if r["suite"] == "e2" and r["metric"] == "objective"]
    expected_pairs = {(f"b_set_minus_{method}", n, budget) for method in METHODS if method != "b_set" for n in LENGTHS for budget in BUDGETS}
    if {(r["contrast"], r["num_tracks"], r["requested_budget"]) for r in comparisons} != expected_pairs or len(comparisons) != len(expected_pairs):
        raise ValueError("paired budget comparisons incomplete or duplicated")
    for row in comparisons:
        if row["ci95"] is None or not np.isfinite([row["mean"], *row["ci95"]]).all() or row["ci95"][0] > row["ci95"][1]:
            raise ValueError("paired interval is missing or invalid")
    return points, comparisons


def load_inputs(report, workspace):
    summary_file = report / "summary.json"
    summary = read(summary_file)
    timing_file = Path(summary["independent_timing"]["summary"])
    if digest(timing_file) != summary["independent_timing"]["sha256"]:
        raise ValueError("timing summary changed after quality report")
    timing = read(timing_file)
    for filename, expected in summary["input_files_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"quality input changed: {filename}")
    for filename, expected in timing["fingerprint"]["code_sha256"].items():
        if digest(workspace / filename) != expected:
            raise ValueError("timing code changed; use the versioned measurement protocol")
    attempt = Path(timing["attempt"])
    if digest(attempt / "measurements.jsonl") != timing["measurements_sha256"]:
        raise ValueError("raw timing records changed after completion")
    measured = rows(attempt / "measurements.jsonl")
    tasks = rows(attempt / "tasks.jsonl")
    if hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest() != timing["fingerprint"]["selected_tasks_sha256"]:
        raise ValueError("selected timing tasks changed")
    # Validate measured task identity and quality against the complete originals.
    originals = {}
    for filename, expected in summary["input_files_sha256"].items():
        path = Path(filename)
        if path.name == "predictions.jsonl":
            if digest(path) != timing["fingerprint"]["main_predictions_sha256"][path.parents[2].name]:
                raise ValueError("timing was compared against a different quality run")
            for row in rows(path):
                originals[row["method"], row["query_id"]] = row
    for row in measured:
        old = originals[row["method"], row["query_id"]]
        fields = ("source", "suite", "num_tracks", "requested_budget", "budget", "library_size", "start_id", "target_id", "seed")
        if any(row[k] != old[k] for k in fields) or abs(row["objective"] - old["objective"]) > 2e-6:
            raise ValueError("timing and quality request contracts differ")
    points, comparisons = prepare_points(summary, timing, measured, tasks)
    provenance = {"quality_sha256": digest(summary_file), "timing_sha256": digest(timing_file),
                  "measurements_sha256": digest(attempt / "measurements.jsonl"),
                  "quality_outputs": summary["outputs"], "timing_measurements": len(measured),
                  "scope": "all planned N=5/10 and B=32/64/128/256 cells; one fitted seed"}
    return summary, points, comparisons, provenance


def render(report, profile, points, comparisons, provenance, figure_tools):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    sys.path.insert(0, str(figure_tools))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": .8, "savefig.facecolor": "white"})
    output = report / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pair_map = {(r["contrast"], r["num_tracks"], r["requested_budget"]): r for r in comparisons}
    refs = [m for m in METHODS if m != "b_set"]

    def padded(values, logarithmic=False):
        lo, hi = min(values), max(values)
        if logarithmic:
            return lo / 1.25, hi * 1.25
        gap = max((hi-lo)*.12, .0001)
        return lo-gap, hi+gap

    pair_limits = padded([0., *(1000 * x for r in comparisons for x in (*r["ci95"], r["mean"]))])
    time_limits = {suite: padded([r["median_total_ms"] for r in points if r["suite"] == suite], True) for suite in SOURCES}
    quality_limits = {suite: padded([r["source_equal_quality"] * (100 if suite == "e3" else 1) for r in points if r["suite"] == suite]) for suite in SOURCES}
    generated = []
    pmemo_saturation = sorted({(r["requested_budget"], r["effective_budget_by_source"]["pmemo"])
                              for r in points if "pmemo" in r["saturated_sources"]})
    saturation_note = ("PMEmo full-bank settings: " + "; ".join(f"B = {b} becomes {effective}" for b, effective in pmemo_saturation)
                       if pmemo_saturation else "No PMEmo full-bank saturation in the displayed budgets.")
    for n in LENGTHS:
        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6))
        fig.subplots_adjust(left=.10, right=.985, bottom=.39, top=.81, wspace=.64)
        fig.suptitle(f"Budget sensitivity (N = {n}; {profile}, seed 42)", fontsize=9, y=.98)
        ax = axes[0]
        for reference, offset in zip(refs, (-.12, -.04, .04, .12)):
            values = [pair_map[f"b_set_minus_{reference}", n, budget] for budget in BUDGETS]
            x = np.arange(len(BUDGETS)) + offset
            center = np.asarray([r["mean"] for r in values]) * 1000
            ax.plot(x, center, color=COLORS[reference], linewidth=.8)
            for position, value, budget in zip(x, values, BUDGETS):
                ax.vlines(position, 1000*value["ci95"][0], 1000*value["ci95"][1], color=COLORS[reference], linewidth=.8)
                ax.plot(position, 1000*value["mean"], marker=MARKERS[budget], markersize=3.5,
                        color=COLORS[reference], linestyle="none")
        ax.axhline(0, color="#777777", linewidth=.7, linestyle="--")
        ax.set_xticks(range(4), [str(b) for b in BUDGETS])
        ax.set_xlim(-.35, 3.35)
        ax.set_ylim(*pair_limits)
        ax.set_xlabel("Requested candidate budget", labelpad=7)
        ax.set_ylabel("B minus reference (x 1,000)", labelpad=7)
        ax.set_title("Paired surrogate cost (E2)", pad=11)
        for ax, suite, title, ylabel in zip(axes[1:], ("e2", "e3"),
                ("Independent VA quality", "Observed preference"), ("True VA MAE", "Tag match (%)")):
            for method in METHODS:
                group = sorted([r for r in points if r["method"] == method and r["suite"] == suite and r["num_tracks"] == n],
                               key=lambda r: r["requested_budget"])
                if len(group) != 4:
                    raise ValueError("missing budget point; do not draw an incomplete curve")
                scale = 100 if suite == "e3" else 1
                ax.plot([r["median_total_ms"] for r in group], [r["source_equal_quality"]*scale for r in group],
                        color=COLORS[method], linewidth=1.4 if method == "b_set" else .8)
                for point in group:
                    ax.plot(point["median_total_ms"], point["source_equal_quality"]*scale,
                            marker=MARKERS[point["requested_budget"]], markersize=4,
                            color=COLORS[method], linestyle="none")
            ax.set_xscale("log")
            # Minor log tick labels collide on narrow panels; keep decade labels.
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(*time_limits[suite])
            ax.set_ylim(*quality_limits[suite])
            ax.set_xlabel("Median request time (ms)", labelpad=7)
            ax.set_ylabel(ylabel, labelpad=7)
            ax.set_title(title, pad=11)
        for ax, letter in zip(axes, "abc"):
            ax.text(-.22, 1.15, letter, transform=ax.transAxes, fontsize=10, weight="bold")
        method_handles = [Line2D([0], [0], color=COLORS[m], linewidth=1.4 if m == "b_set" else .8, label=NAMES[m]) for m in METHODS]
        budget_handles = [Line2D([0], [0], color="#555555", marker=MARKERS[b], linestyle="none", markersize=4, label=f"B = {b}") for b in BUDGETS]
        fig.legend(handles=method_handles, loc="lower center", bbox_to_anchor=(.5, .17), ncol=5, frameon=False,
                   handlelength=1.3, columnspacing=1.1)
        fig.legend(handles=budget_handles, loc="lower center", bbox_to_anchor=(.5, .085), ncol=4, frameon=False,
                   handlelength=1.0, columnspacing=1.5)
        fig.text(.5, .035, saturation_note, ha="center", fontsize=7)
        base = output / f"budget_quality_cost_n{n}"
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig, json_out=str(base)+".alignment.json",
                                           overlay_svg=str(base)+".alignment.svg", tolerance_pt=1.5,
                                           gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
        for suffix in ("pdf", "svg"):
            fig.savefig(str(base)+"."+suffix)
        fig.savefig(str(base)+".png", dpi=600)
        plt.close(fig)
        data = {"points": [r for r in points if r["num_tracks"] == n],
                "paired_comparisons": [r for r in comparisons if r["num_tracks"] == n],
                "provenance": provenance, "axis_ranges_shared_across_lengths": True}
        base.with_suffix(".source.json").write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
        caption = (f"Budget-dependent optimization and independent quality at N={n}. "
                   "a, Source-equal E2 differences between B-set and each reference at the same requested budget; negative values favor B. "
                   "Points show paired means, vertical lines show 95% request-family percentile bootstrap intervals (2,000 draws). "
                   "b, Source-equal true VA MAE across three E2 sources against median cached-feature request time. "
                   "c, Observed preference-tag match on MTG against the same timing definition. "
                   "Colors denote methods, shapes denote requested budgets, and lines connect budgets without fitting a monotonic trend. "
                   "Panels b/c are descriptive mean curves; uncertainty inference uses the complete paired tables, not visual curve separation. "
                   "Quality uses all planned requests; timing uses four fixed families per source/length/budget, five measured passes after one warmup. "
                   "Source data give quality-request, timing-request and repeat counts separately. Repeated timings are not additional scientific samples. "
                   "Inference includes objective setup and candidate/search work, excluding encoding, model loading, disk access and metric computation. "
                   f"{saturation_note} Dynamic reduction uses per-step K=16, not global set budget B. "
                   "One fitted seed and fixed libraries are used; intervals do not estimate training-seed variability and have no multiplicity adjustment.")
        base.with_suffix(".caption.md").write_text(caption+"\n", encoding="utf-8")
        with base.with_suffix(".qa.log").open("w", encoding="utf-8") as stream:
            for script, tail in (("audit_pdf_text.py", ["--min-pt", "5"]),
                                 ("audit_figure_collisions.py", ["--json-out", str(base)+".collision.json", "--overlay-pdf", str(base)+".collision.pdf"])):
                code = subprocess.run([sys.executable, str(figure_tools/script), str(base)+".pdf", *tail],
                                      stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    raise RuntimeError(f"figure QA requires repair: {base.name}/{script}")
        generated.append(base.name)
    record = {"status": "exported_requires_visual_review", "figures": generated, "backend": "python_matplotlib",
              "provenance": provenance, "source_sha256": digest(Path(__file__)), "visual_review_completed": False,
              "layout": {"inches": [7.2, 3.6], "dpi": 600, "alignment_tolerance_pt": 1.5}}
    (output/"manifest.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/sequence_routes_v1"))
    parser.add_argument("--profile", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--run-id", default="budget_curve_v1")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--figure-tools", type=Path, default=Path.home()/".agents/skills/nature-figure/scripts")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        parser.error("invalid run ID")
    report = args.root / "reports" / args.profile / args.run_id
    report.mkdir(parents=True, exist_ok=True)
    summary_file = report / "summary.json"
    if not summary_file.exists() or read(summary_file)["status"] != "completed":
        status = {"status": "pending_quality_or_timing", "figure_exported": False}
        (report/"figure_preflight.json").write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(status))
        return 2
    _, points, comparisons, provenance = load_inputs(report, Path(__file__).resolve().parents[2])
    if args.preflight:
        print(json.dumps({"status": "inputs_verified_not_rendered", "points": len(points), "paired_estimates": len(comparisons)}))
        return 0
    audited = subprocess.run([sys.executable, str(args.figure_tools/"validate_figure.py"), str(Path(__file__)), "--json"],
                             capture_output=True, text=True, check=True)
    (report/"figure_source_audit.json").write_text(audited.stdout, encoding="utf-8")
    render(report, args.profile, points, comparisons, provenance, args.figure_tools)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
