#!/usr/bin/env python3
"""Plot a run_matrix.py results file as PNGs.

Usage:
    python3 benchmarks/plot_results.py benchmarks/results/<file>.jsonl
        [--out DIR]

Needs matplotlib (the system python3 has it; the project venv does not).
Writes, into DIR (default: benchmarks/results/plots/<file stem>/), every
chart the file has data for:
  size.png      GFLOPS against N, when >= 3 square shapes ran (skipped when
                every shape is <= 128, where fixed cost dominates)
  scaling.png   GFLOPS against thread count, when >= 3 thread counts ran
  epilogue.png  dense / matmul time per backend, when both ops ran
  shapes.png    GFLOPS per shape as grouped bars, for the remaining files
Each GFLOPS chart has a *_time.png twin with time per call on a log axis.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator,  # noqa: E402
                               NullFormatter)

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Fixed per backend, so a backend keeps its color whichever subset ran.
# Slots 1-5 of the validated reference palette, in its order; three are
# under 3:1 on the surface, so lines also get a marker shape and an end label.
STYLE = {
    "edsl": ("#2a78d6", "o"),
    "mkl": ("#eb6834", "s"),
    "jax": ("#1baf7a", "^"),
    "openblas": ("#eda100", "D"),
    "blis": ("#e87ba4", "v"),
}
SURFACE, INK, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(path: Path) -> tuple:
    """(meta, gflops, seconds, failed count); the maps are keyed by
    (op, (m, n, k), backend, threads), median over rounds."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    meta, points = lines[0]["meta"], lines[1:]
    gf, secs = {}, {}
    for r in points:
        if "error" in r:
            continue
        key = (r["op"], (r["m"], r["n"], r["k"]), r["backend"], r["threads"])
        gf.setdefault(key, []).append(r["gflops"])
        secs.setdefault(key, []).append(r["median_s"])
    failed = sum("error" in r for r in points)
    return (meta, {k: statistics.median(v) for k, v in gf.items()},
            {k: statistics.median(v) for k, v in secs.items()}, failed)


def setup_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Recessive grid and spines, text in ink colors."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_xlabel(xlabel, color=INK_MUTED, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_MUTED, fontsize=9)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED, labelsize=8)


def line_panel(ax: plt.Axes, xs: list, series: dict) -> None:
    """One line per backend; label_line_ends adds the direct labels once
    the axis limits are final."""
    for backend, ys in series.items():
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            continue
        color, marker = STYLE[backend]
        ax.plot(*zip(*pts), color=color, marker=marker, linewidth=2,
                markersize=5, markeredgecolor=SURFACE, label=backend)


def label_line_ends(ax: plt.Axes, min_gap: float = 0.07) -> None:
    """Label each line at its last point, nudging labels apart vertically
    (in axes fraction) where lines end close together."""
    to_axes = ax.transData + ax.transAxes.inverted()
    ends = []
    for line in ax.get_lines():
        if line.get_label().startswith("_"):
            continue
        x, y = line.get_xdata()[-1], line.get_ydata()[-1]
        ends.append([to_axes.transform((x, y))[1], x, line.get_label()])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + min_gap)
    # If the stack ran off the top, pull it back down from there.
    if ends and ends[-1][0] > 1.0:
        ends[-1][0] = 1.0
        for i in range(len(ends) - 2, -1, -1):
            ends[i][0] = min(ends[i][0], ends[i + 1][0] - min_gap)
    for frac, x, label in ends:
        ax.annotate(label, (x, frac), xycoords=("data", "axes fraction"),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK_MUTED, annotation_clip=False)


def bar_panel(ax: plt.Axes, groups: list, series: dict) -> None:
    """Grouped bars, one group per x label, one bar per backend."""
    backends = [b for b, vals in series.items() if any(vals)]
    width = 0.8 / max(len(backends), 1)
    for i, b in enumerate(backends):
        xs = [g + (i - (len(backends) - 1) / 2) * width
              for g in range(len(groups))]
        ax.bar(xs, [v or 0 for v in series[b]], width=width * 0.92,
               color=STYLE[b][0], label=b, linewidth=0)
    ax.set_xticks(range(len(groups)), groups)


def figure(panels: int, title: str, meta: dict) -> tuple:
    """A row-wrapped grid of panels with the run described under the title."""
    cols = min(panels, 4)
    rows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows + 0.7),
                             squeeze=False, facecolor=SURFACE)
    edsl = meta.get("edsl")
    build = f" · edsl .so built {edsl['so_built']}" if edsl else ""
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left",
                 y=1 - 0.1 / fig.get_figheight(), va="top")
    fig.text(0.01, 1 - 0.42 / fig.get_figheight(),
             f"preset {meta.get('preset') or 'custom'} · {meta['rounds']} "
             f"round(s) · min_time {meta['min_time']} s{build}",
             color=INK_MUTED, fontsize=8)
    flat = [ax for row in axes for ax in row]
    for ax in flat[panels:]:
        ax.set_visible(False)
    return fig, flat[:panels]


def finish(fig: plt.Figure, axes: list, path: Path) -> None:
    """Shared legend in one row above the panels, then save."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="upper right", ncol=len(labels),
               frameon=False, fontsize=8, labelcolor=INK_MUTED,
               bbox_to_anchor=(1, 1 - 0.05 / fig.get_figheight()))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.45 / fig.get_figheight()))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def threads_label(t: int) -> str:
    return f"{t} thread{'s' if t > 1 else ''}"


def fmt_time(seconds: float) -> str:
    """1us, 250us, 3ms, 1.2s: the unit that keeps the number readable."""
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "us")):
        if seconds >= scale:
            return f"{seconds / scale:g}{unit}"
    return f"{seconds * 1e9:g}ns"


def apply_metric(ax: plt.Axes, metric: str) -> None:
    """GFLOPS from zero; time on a log axis labelled in us/ms/s."""
    if metric == "gflops":
        ax.set_ylim(bottom=0)
    elif metric == "time":
        # 1-2-5 steps, all labelled: a panel can span less than a decade,
        # where the default leaves only minor ticks in 10^x notation.
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_time(v)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        raise ValueError(f"unknown metric {metric!r}")


METRIC_LABEL = {"gflops": "GFLOPS", "time": "time per call (log)"}
METRIC_TITLE = {"gflops": "GFLOPS", "time": "Time per call"}
METRIC_SUFFIX = {"gflops": "", "time": "_time"}


def plot_size(meta, data, metric, shapes, out: Path) -> None:
    """`metric` against N, one panel per (op, threads)."""
    ns = [s[0] for s in shapes]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs size", meta)
    for ax, (op, t) in zip(axes, combos):
        series = {b: [data.get((op, s, b, t)) for s in shapes]
                  for b in meta["backends"]}
        # Evenly spaced categories rather than a log axis, so 1000, 1023
        # and 1024 each get their own slot.
        slots = list(range(len(ns)))
        line_panel(ax, slots, series)
        ax.set_xticks(slots, [str(n) for n in ns])
        ax.set_xlim(-0.4, len(ns) - 0.4)
        apply_metric(ax, metric)
        setup_axes(ax, f"{op}, {threads_label(t)}", "N  (N x N x N)",
                   METRIC_LABEL[metric])
        ax.tick_params(axis="x", labelrotation=45)
        label_line_ends(ax)
    finish(fig, axes, out / f"size{METRIC_SUFFIX[metric]}.png")


def plot_scaling(meta, data, metric, shapes, out: Path) -> None:
    """`metric` against thread count, one panel per (op, shape)."""
    threads = meta["threads"]
    combos = [(op, s) for op in meta["ops"] for s in shapes]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs threads",
                       meta)
    for ax, (op, s) in zip(axes, combos):
        series = {b: [data.get((op, s, b, t)) for t in threads]
                  for b in meta["backends"]}
        line_panel(ax, threads, series)
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        apply_metric(ax, metric)
        setup_axes(ax, f"{op} {'x'.join(map(str, s))}", "threads",
                   METRIC_LABEL[metric])
        label_line_ends(ax)
    finish(fig, axes, out / f"scaling{METRIC_SUFFIX[metric]}.png")


def plot_epilogue(meta, secs, shapes, out: Path) -> None:
    """dense / matmul time per backend, one panel per thread count."""
    labels = ["x".join(map(str, s)) for s in shapes]
    fig, axes = figure(len(meta["threads"]), "Dense layer time / matmul "
                       "time  (dashed line = epilogue is free)", meta)
    for ax, t in zip(axes, meta["threads"]):
        series = {}
        for b in meta["backends"]:
            vals = []
            for s in shapes:
                d = secs.get(("dense", s, b, t))
                m = secs.get(("matmul", s, b, t))
                vals.append(d / m if d and m else None)
            series[b] = vals
        bar_panel(ax, labels, series)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(bottom=0)
        setup_axes(ax, threads_label(t), "shape", "dense / matmul time")
    finish(fig, axes, out / "epilogue.png")


def plot_shapes(meta, data, metric, shapes, out: Path) -> None:
    """`metric` per shape as grouped bars, one panel per (op, threads)."""
    labels = ["x".join(map(str, s)) for s in shapes]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} per shape", meta)
    for ax, (op, t) in zip(axes, combos):
        series = {b: [data.get((op, s, b, t)) for s in shapes]
                  for b in meta["backends"]}
        bar_panel(ax, labels, series)
        apply_metric(ax, metric)
        setup_axes(ax, f"{op}, {threads_label(t)}", "shape",
                   METRIC_LABEL[metric])
        ax.tick_params(axis="x", labelrotation=20)
    finish(fig, axes, out / f"shapes{METRIC_SUFFIX[metric]}.png")


def main() -> int:
    """Draw every chart the results file has data for."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    meta, gf, secs, failed = load(args.results)
    if failed:
        print(f"note: {failed} failed point(s) left out", file=sys.stderr)
    out = args.out or RESULTS_DIR / "plots" / args.results.stem
    out.mkdir(parents=True, exist_ok=True)

    shapes = [tuple(int(p) for p in s.lower().split("x"))
              for s in meta["shapes"]]
    metrics = {"gflops": gf, "time": secs}
    drew = False
    if all(m == n == k for m, n, k in shapes) and len(shapes) >= 3:
        # GFLOPS says little where fixed per-call cost dominates.
        small = max(s[0] for s in shapes) <= 128
        for metric, data in metrics.items():
            if not (small and metric == "gflops"):
                plot_size(meta, data, metric, sorted(shapes), out)
        drew = True
    if len(meta["threads"]) >= 3:
        for metric, data in metrics.items():
            plot_scaling(meta, data, metric, shapes, out)
        drew = True
    if {"matmul", "dense"} <= set(meta["ops"]):
        plot_epilogue(meta, secs, shapes, out)
        drew = True
    if not drew:
        for metric, data in metrics.items():
            plot_shapes(meta, data, metric, shapes, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
