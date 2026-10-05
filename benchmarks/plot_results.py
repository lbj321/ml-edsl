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
  speedup.png   GFLOPS / mkl against N, paired per round, for the same
                files when mkl ran
  scaling.png   GFLOPS against thread count, when >= 3 thread counts ran
  epilogue.png  dense / matmul time per backend, when both ops ran
  shapes.png    GFLOPS per shape as grouped bars, for the remaining files
Each GFLOPS chart has a *_time.png twin with time per call on a log axis.
Lines are the median over rounds, with a band from the slowest to the
fastest round; bars show the median only.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator,  # noqa: E402
                               NullFormatter)

from aggregate import (BASELINE, Results, Spread, load,  # noqa: E402
                       paired_ratios, peak_gflops, spread)

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


def point_spread(res: Results, key: tuple, metric: str) -> Optional[Spread]:
    """Spread over rounds of one point's metric, or None if it never ran."""
    point = res.points.get(key)
    if point is None:
        return None
    if metric == "gflops":
        return spread(point.gflops)
    if metric == "time":
        return spread(point.secs)
    raise ValueError(f"unknown metric {metric!r}")


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
    """One median line per backend over its between-round band;
    label_line_ends adds the direct labels once the axis limits are final."""
    for backend, spreads in series.items():
        pts = [(x, s) for x, s in zip(xs, spreads) if s is not None]
        if not pts:
            continue
        px, ps = zip(*pts)
        color, marker = STYLE[backend]
        ax.fill_between(px, [s.lo for s in ps], [s.hi for s in ps],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(px, [s.median for s in ps], color=color, marker=marker,
                linewidth=2, markersize=5, markeredgecolor=SURFACE,
                label=backend)


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
    """Grouped bars at the median, one group per x label, one bar per
    backend."""
    backends = [b for b, spreads in series.items()
                if any(s is not None for s in spreads)]
    width = 0.8 / max(len(backends), 1)
    for i, b in enumerate(backends):
        xs = [g + (i - (len(backends) - 1) / 2) * width
              for g in range(len(groups))]
        ax.bar(xs, [s.median if s else 0 for s in series[b]],
               width=width * 0.92, color=STYLE[b][0], label=b, linewidth=0)
    ax.set_xticks(range(len(groups)), groups)


def figure(panels: int, title: str, meta: dict,
           bands: bool = False) -> tuple:
    """A row-wrapped grid of panels with the run described under the title;
    `bands` notes what the line plots' bands mean."""
    cols = min(panels, 4)
    rows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows + 0.7),
                             squeeze=False, facecolor=SURFACE)
    edsl = meta.get("edsl")
    build = f" · edsl .so built {edsl['so_built']}" if edsl else ""
    # Not a confidence interval; say so where the reader sees the plot.
    band = (" · bands: min-max over rounds"
            if bands and meta["rounds"] > 1 else "")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left",
                 y=1 - 0.1 / fig.get_figheight(), va="top")
    fig.text(0.01, 1 - 0.42 / fig.get_figheight(),
             f"preset {meta.get('preset') or 'custom'} · {meta['rounds']} "
             f"round(s) · min_time {meta['min_time']} s{band}{build}",
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


def peak_level(ax: plt.Axes, meta: dict, threads: int) -> None:
    """Dashed line at the peak for one thread count; files from before the
    peak was recorded get none."""
    peak = peak_gflops(meta, threads)
    if peak is None:
        return
    # "_" keeps it out of the legend and label_line_ends.
    ax.axhline(peak, color=INK_MUTED, linewidth=1, linestyle="--",
               label="_peak")
    ax.annotate(f"peak {peak:.0f}", (0.01, peak),
                xycoords=("axes fraction", "data"), xytext=(0, 2),
                textcoords="offset points", va="bottom", fontsize=7,
                color=INK_MUTED)


def peak_curve(ax: plt.Axes, meta: dict, threads: list) -> None:
    """Dashed peak against thread count: ideal scaling, with the clock
    falling as more cores are busy."""
    if meta.get("peak") is None:
        return
    peaks = [peak_gflops(meta, t) for t in threads]
    ax.plot(threads, peaks, color=INK_MUTED, linewidth=1, linestyle="--",
            label="_peak")
    ax.annotate("peak", (threads[-1], peaks[-1]), xytext=(-2, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=7, color=INK_MUTED)


METRIC_LABEL = {"gflops": "GFLOPS", "time": "time per call (log)"}
METRIC_TITLE = {"gflops": "GFLOPS", "time": "Time per call"}
METRIC_SUFFIX = {"gflops": "", "time": "_time"}


def plot_size(res: Results, metric: str, shapes: list, out: Path) -> None:
    """`metric` against N, one panel per (op, threads)."""
    meta = res.meta
    ns = [s[0] for s in shapes]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs size", meta,
                       bands=True)
    for ax, (op, t) in zip(axes, combos):
        series = {b: [point_spread(res, (op, s, b, t), metric)
                      for s in shapes]
                  for b in meta["backends"]}
        slots = n_slots(ax, ns)
        line_panel(ax, slots, series)
        if metric == "gflops":
            peak_level(ax, meta, t)
        apply_metric(ax, metric)
        setup_axes(ax, f"{op}, {threads_label(t)}", "N  (N x N x N)",
                   METRIC_LABEL[metric])
        ax.tick_params(axis="x", labelrotation=45)
        label_line_ends(ax)
    finish(fig, axes, out / f"size{METRIC_SUFFIX[metric]}.png")


def n_slots(ax: plt.Axes, ns: list) -> list:
    """Evenly spaced categories for N rather than a log axis, so 1000, 1023
    and 1024 each get their own slot; returns the slot positions."""
    slots = list(range(len(ns)))
    ax.set_xticks(slots, [str(n) for n in ns])
    ax.set_xlim(-0.4, len(ns) - 0.4)
    return slots


def ratio_axis(ax: plt.Axes) -> None:
    """Log y-axis labelled 0.5x, 1x, ...; tick steps chosen from the data's
    span, since ratios range from a few percent to three decades."""
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    decades = math.log10(hi / lo)
    if decades < 0.5:
        subs = range(1, 10)
    elif decades < 1.5:
        subs = (1, 2, 3, 5, 7)
    else:
        subs = (1, 2, 5)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=subs))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}x"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_speedup(res: Results, shapes: list, out: Path) -> None:
    """GFLOPS / BASELINE against N, paired per round, one panel per
    (op, threads); the baseline itself is the dashed line at 1."""
    meta = res.meta
    ns = [s[0] for s in shapes]
    others = [b for b in meta["backends"] if b != BASELINE]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"Speedup vs {BASELINE}  (above 1 = "
                       f"faster than {BASELINE})", meta, bands=True)
    for ax, (op, t) in zip(axes, combos):
        series = {}
        for b in others:
            ratios = [paired_ratios(res, (op, s, b, t)) for s in shapes]
            series[b] = [spread(r) if r else None for r in ratios]
        slots = n_slots(ax, ns)
        line_panel(ax, slots, series)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--",
                   label="_baseline")
        ax.annotate(BASELINE, (0.01, 1.0), xycoords=("axes fraction", "data"),
                    xytext=(0, 2), textcoords="offset points", va="bottom",
                    fontsize=7, color=INK_MUTED)
        ratio_axis(ax)
        setup_axes(ax, f"{op}, {threads_label(t)}", "N  (N x N x N)",
                   f"GFLOPS / {BASELINE} (log)")
        ax.tick_params(axis="x", labelrotation=45)
        label_line_ends(ax)
    finish(fig, axes, out / "speedup.png")


def plot_scaling(res: Results, metric: str, shapes: list, out: Path) -> None:
    """`metric` against thread count, one panel per (op, shape)."""
    meta = res.meta
    threads = meta["threads"]
    combos = [(op, s) for op in meta["ops"] for s in shapes]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs threads",
                       meta, bands=True)
    for ax, (op, s) in zip(axes, combos):
        series = {b: [point_spread(res, (op, s, b, t), metric)
                      for t in threads]
                  for b in meta["backends"]}
        line_panel(ax, threads, series)
        if metric == "gflops":
            peak_curve(ax, meta, threads)
        ax.set_xscale("log", base=2)
        ax.set_xticks(threads, [str(t) for t in threads])
        apply_metric(ax, metric)
        setup_axes(ax, f"{op} {'x'.join(map(str, s))}", "threads",
                   METRIC_LABEL[metric])
        label_line_ends(ax)
    finish(fig, axes, out / f"scaling{METRIC_SUFFIX[metric]}.png")


def epilogue_spread(res: Results, shape: tuple, backend: str,
                    threads: int) -> Optional[Spread]:
    """dense / matmul time, paired per round like the ratios to MKL."""
    dense = res.points.get(("dense", shape, backend, threads))
    matmul = res.points.get(("matmul", shape, backend, threads))
    if dense is None or matmul is None:
        return None
    ratios = {rnd: d / matmul.secs[rnd] for rnd, d in dense.secs.items()
              if rnd in matmul.secs}
    return spread(ratios) if ratios else None


def plot_epilogue(res: Results, shapes: list, out: Path) -> None:
    """dense / matmul time per backend, one panel per thread count."""
    meta = res.meta
    labels = ["x".join(map(str, s)) for s in shapes]
    fig, axes = figure(len(meta["threads"]), "Dense layer time / matmul "
                       "time  (dashed line = epilogue is free)", meta)
    for ax, t in zip(axes, meta["threads"]):
        series = {b: [epilogue_spread(res, s, b, t) for s in shapes]
                  for b in meta["backends"]}
        bar_panel(ax, labels, series)
        ax.axhline(1.0, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.set_ylim(bottom=0)
        setup_axes(ax, threads_label(t), "shape", "dense / matmul time")
    finish(fig, axes, out / "epilogue.png")


def plot_shapes(res: Results, metric: str, shapes: list, out: Path) -> None:
    """`metric` per shape as grouped bars, one panel per (op, threads)."""
    meta = res.meta
    labels = ["x".join(map(str, s)) for s in shapes]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} per shape", meta)
    for ax, (op, t) in zip(axes, combos):
        series = {b: [point_spread(res, (op, s, b, t), metric)
                      for s in shapes]
                  for b in meta["backends"]}
        bar_panel(ax, labels, series)
        if metric == "gflops":
            peak_level(ax, meta, t)
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

    res = load(args.results)
    if res.failed:
        print(f"note: {len(res.failed)} failed point(s) left out",
              file=sys.stderr)
    out = args.out or RESULTS_DIR / "plots" / args.results.stem
    out.mkdir(parents=True, exist_ok=True)

    meta, shapes = res.meta, res.shapes
    metrics = ("gflops", "time")
    drew = False
    if all(m == n == k for m, n, k in shapes) and len(shapes) >= 3:
        # GFLOPS says little where fixed per-call cost dominates.
        small = max(s[0] for s in shapes) <= 128
        for metric in metrics:
            if not (small and metric == "gflops"):
                plot_size(res, metric, sorted(shapes), out)
        if BASELINE in meta["backends"] and len(meta["backends"]) > 1:
            plot_speedup(res, sorted(shapes), out)
        drew = True
    if len(meta["threads"]) >= 3:
        for metric in metrics:
            plot_scaling(res, metric, shapes, out)
        drew = True
    if {"matmul", "dense"} <= set(meta["ops"]):
        plot_epilogue(res, shapes, out)
        drew = True
    if not drew:
        for metric in metrics:
            plot_shapes(res, metric, shapes, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
