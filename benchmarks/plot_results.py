#!/usr/bin/env python3
"""Plot a run_matrix.py results file as PNGs.

Usage:
    python3 benchmarks/plot_results.py benchmarks/results/<file>.jsonl
        [--out DIR]

Needs matplotlib (the system python3 has it; the project venv does not).
Writes, into DIR (default: benchmarks/results/plots/<file stem>/), every
chart the file has data for:
  size.png      GFLOPS against M, when >= 3 shapes ran that are all cubes
                or all share N and K (skipped when every dimension is
                <= 128, where fixed cost dominates)
  shapes.png    GFLOPS per shape, for files without a size sweep
  scaling.png   GFLOPS against thread count, when >= 3 thread counts ran
  epilogue.png  dense / matmul time per backend, when both ops ran
  summary.md    GFLOPS and % of peak per backend at 1 and all threads,
                for every file
Each GFLOPS chart has a *_time.png twin with time per call on a log axis.
Backends are shown side by side, each in absolute terms, not as ratios to
one of them. size and shapes draw grouped bars; scaling keeps lines,
since its slope against the peak curve is the point. Each round counts as
the median of its call times. Bars and lines show the 20% trimmed mean
over rounds, whiskers and bands its 95% CI (Yuen); see aggregate.py.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import (FuncFormatter, LogLocator,  # noqa: E402
                               NullFormatter)

from aggregate import (Results, Spread, clock_label,  # noqa: E402
                       epilogue_ratios, gflops_spread, load, peak_gflops,
                       ratio_spread, secs_spread, trim_count)

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
        return gflops_spread(point.gflops)
    if metric == "time":
        return secs_spread(point.secs)
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
    """One mean line per backend over a band spanning its CI;
    label_line_ends adds the direct labels once the axis limits are final."""
    for backend, spreads in series.items():
        pts = [(x, s) for x, s in zip(xs, spreads) if s is not None]
        if not pts:
            continue
        px, ps = zip(*pts)
        color, marker = STYLE[backend]
        ax.fill_between(px, [s.lo for s in ps], [s.hi for s in ps],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(px, [s.mean for s in ps], color=color, marker=marker,
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
    """Grouped bars at the mean, one group per x label, one bar per
    backend, each with a whisker over its CI."""
    backends = [b for b, spreads in series.items()
                if any(s is not None for s in spreads)]
    width = 0.8 / max(len(backends), 1)
    for i, b in enumerate(backends):
        xs = [g + (i - (len(backends) - 1) / 2) * width
              for g in range(len(groups))]
        pts = [(x, s) for x, s in zip(xs, series[b]) if s is not None]
        px, ps = zip(*pts)
        ax.bar(px, [s.mean for s in ps], width=width * 0.92,
               color=STYLE[b][0], label=b, linewidth=0)
        ax.errorbar(px, [s.mean for s in ps],
                    yerr=[[s.mean - s.lo for s in ps],
                          [s.hi - s.mean for s in ps]],
                    fmt="none", ecolor=INK, elinewidth=0.8, capsize=0,
                    label="_ci")
    ax.set_xticks(range(len(groups)), groups)


MARKS = {"bars": ("bars", "whiskers"), "lines": ("lines", "bands")}


def figure(panels: int, title: str, meta: dict, kind: str) -> tuple:
    """A row-wrapped grid of panels with the run described under the title,
    including what the `kind` of chart ('bars' or 'lines') draws."""
    if kind not in MARKS:
        raise ValueError(f"unknown chart kind {kind!r}")
    # Two columns keep panels legible when the PNG is scaled to page width.
    cols = min(panels, 2)
    rows = (panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows + 0.7),
                             squeeze=False, facecolor=SURFACE)
    parts = [f"preset {meta.get('preset') or 'custom'}",
             f"{meta['rounds']} round(s)"]
    if meta["rounds"] > 1:
        mark, interval = MARKS[kind]
        g = trim_count(meta["rounds"])
        parts.append(f"{mark}: mean over rounds of each round's median "
                     f"call time, {g} lowest and {g} highest rounds trimmed")
        parts.append(f"{interval}: 95% CI (Yuen)")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left",
                 y=1 - 0.1 / fig.get_figheight(), va="top")
    wrapped_text(fig, 0.01, 1 - 0.3 / fig.get_figheight(), parts,
                 color=INK_MUTED, fontsize=8, va="top")
    flat = [ax for row in axes for ax in row]
    for ax in flat[panels:]:
        ax.set_visible(False)
    return fig, flat[:panels]


def wrapped_text(fig: plt.Figure, x: float, y: float, parts: list,
                 **kwargs) -> None:
    """`parts` joined by ' · ', starting a new line wherever the next part
    would run past the figure's right edge (one-panel figures are narrow)."""
    text = fig.text(x, y, "", **kwargs)
    renderer = fig.canvas.get_renderer()
    room = fig.bbox.width * (1 - 2 * x)
    lines = []
    for part in parts:
        if lines:
            text.set_text(f"{lines[-1]} · {part}")
            if text.get_window_extent(renderer).width <= room:
                lines[-1] = f"{lines[-1]} · {part}"
                continue
        lines.append(part)
    text.set_text("\n".join(lines))


def finish(fig: plt.Figure, axes: list, path: Path) -> None:
    """Shared legend in one row, top right beside the title, or under the
    header where the title leaves no room; then save."""
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)

    def legend(loc: str, anchor: tuple) -> plt.Artist:
        return fig.legend(handles, labels, loc=loc, ncol=len(labels),
                          frameon=False, fontsize=8, labelcolor=INK_MUTED,
                          bbox_to_anchor=anchor)

    renderer = fig.canvas.get_renderer()
    height = fig.bbox.height
    texts = [t.get_window_extent(renderer) for t in fig.texts]
    header_bottom = min(t.y0 for t in texts)
    leg = legend("upper right", (1, 1 - 0.05 / fig.get_figheight()))
    if any(leg.get_window_extent(renderer).overlaps(t) for t in texts):
        leg.remove()
        leg = legend("upper left", (0, header_bottom / height))
        header_bottom = leg.get_window_extent(renderer).y0
    # Panels start below the header, however many rows it took.
    fig.tight_layout(rect=(0, 0, 1, (header_bottom - 0.04 * fig.dpi)
                           / height))
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print(path)


def threads_label(t: int) -> str:
    """'1 thread', else 'max 4 threads': the thread setting is a cap, and a
    library may use fewer on a small problem."""
    return "1 thread" if t == 1 else f"max {t} threads"


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
    """Dashed line at the peak for one thread count, which is also the top
    of the axis, since no measurement can pass it; files from before the
    peak was recorded get neither."""
    peak = peak_gflops(meta, threads)
    if peak is None:
        return
    # "_" keeps it out of the legend and label_line_ends.
    ax.axhline(peak, color=INK_MUTED, linewidth=1, linestyle="--",
               label="_peak")
    ax.set_ylim(top=peak)
    ax.annotate(f"peak {peak:.0f}", (0.01, peak),
                xycoords=("axes fraction", "data"), xytext=(0, -2),
                textcoords="offset points", va="top", fontsize=7,
                color=INK_MUTED)


def peak_curve(ax: plt.Axes, meta: dict, threads: list) -> None:
    """Dashed peak against thread count: ideal scaling, with the clock
    falling as more cores are busy. Its highest point is the top of the
    axis."""
    if meta.get("peak") is None:
        return
    peaks = [peak_gflops(meta, t) for t in threads]
    ax.plot(threads, peaks, color=INK_MUTED, linewidth=1, linestyle="--",
            label="_peak")
    ax.set_ylim(top=max(peaks))
    ax.annotate("peak", (threads[-1], peaks[-1]), xytext=(-4, -2),
                textcoords="offset points", ha="right", va="top",
                fontsize=7, color=INK_MUTED)


METRIC_LABEL = {"gflops": "GFLOPS", "time": "time per call (log)"}
METRIC_TITLE = {"gflops": "GFLOPS", "time": "Time per call"}
METRIC_SUFFIX = {"gflops": "", "time": "_time"}


def size_axis(shapes: list) -> Optional[str]:
    """X-axis label when the shapes sweep a single size, M, with the rest
    following it (cubes) or fixed (a dense layer over batch); else None."""
    if all(m == n == k for m, n, k in shapes):
        return "N  (M = N = K)"
    if len({(n, k) for _, n, k in shapes}) == 1:
        _, n, k = shapes[0]
        return f"batch M  ({k} → {n})"
    return None


def plot_size(res: Results, metric: str, shapes: list, xlabel: str,
              out: Path) -> None:
    """`metric` against M as grouped bars, one panel per (op, threads).
    Sizes are categories, not a log axis, so 1000, 1023 and 1024 each get
    their own group."""
    meta = res.meta
    groups = [str(s[0]) for s in shapes]
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs size", meta,
                       "bars")
    for ax, (op, t) in zip(axes, combos):
        series = {b: [point_spread(res, (op, s, b, t), metric)
                      for s in shapes]
                  for b in meta["backends"]}
        bar_panel(ax, groups, series)
        if metric == "gflops":
            peak_level(ax, meta, t)
        apply_metric(ax, metric)
        setup_axes(ax, f"{op}, {threads_label(t)}", xlabel,
                   METRIC_LABEL[metric])
    finish(fig, axes, out / f"size{METRIC_SUFFIX[metric]}.png")


def plot_scaling(res: Results, metric: str, shapes: list, out: Path) -> None:
    """`metric` against thread count, one panel per (op, shape)."""
    meta = res.meta
    threads = meta["threads"]
    combos = [(op, s) for op in meta["ops"] for s in shapes]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} vs threads",
                       meta, "lines")
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
        setup_axes(ax, f"{op} {shape_label(op, s)}", "max threads",
                   METRIC_LABEL[metric])
        label_line_ends(ax)
    finish(fig, axes, out / f"scaling{METRIC_SUFFIX[metric]}.png")


def epilogue_spread(res: Results, shape: tuple, backend: str,
                    threads: int) -> Optional[Spread]:
    """dense / matmul time of one backend, paired per round."""
    ratios = epilogue_ratios(res, shape, backend, threads)
    return ratio_spread(ratios) if ratios else None


def plot_epilogue(res: Results, shapes: list, out: Path) -> None:
    """dense / matmul time per backend, one panel per thread count."""
    meta = res.meta
    labels = [shape_label("dense", s) for s in shapes]
    fig, axes = figure(len(meta["threads"]), "Dense layer time / matmul "
                       "time  (dashed line = epilogue is free)", meta, "bars")
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
    combos = [(op, t) for op in meta["ops"] for t in meta["threads"]]
    fig, axes = figure(len(combos), f"{METRIC_TITLE[metric]} per shape", meta,
                       "bars")
    for ax, (op, t) in zip(axes, combos):
        series = {b: [point_spread(res, (op, s, b, t), metric)
                      for s in shapes]
                  for b in meta["backends"]}
        bar_panel(ax, [shape_label(op, s) for s in shapes], series)
        if metric == "gflops":
            peak_level(ax, meta, t)
        apply_metric(ax, metric)
        setup_axes(ax, f"{op}, {threads_label(t)}", "shape",
                   METRIC_LABEL[metric])
        ax.tick_params(axis="x", labelrotation=20)
    finish(fig, axes, out / f"shapes{METRIC_SUFFIX[metric]}.png")


def summary_cell(res: Results, key: tuple) -> str:
    """Mean GFLOPS and its share of the peak, or '-' if it never ran."""
    point = res.points.get(key)
    if point is None:
        return "-"
    gf = gflops_spread(point.gflops).mean
    peak = peak_gflops(res.meta, key[3])
    return f"{gf:.0f} ({gf / peak:.0%})" if peak else f"{gf:.0f}"


def shape_label(op: str, shape: tuple) -> str:
    """'1024³' for a cube, 'batch 16, 1024→4096' for a dense layer (batch,
    in -> out features), else 'MxNxK'."""
    m, n, k = shape
    if m == n == k:
        return f"{m}³"
    if op == "dense":
        return f"batch {m}, {k}→{n}"
    return f"{m}x{n}x{k}"


def write_summary(res: Results, path: Path) -> None:
    """Markdown table per op: (shape, 1 thread and all threads) x backends,
    mean GFLOPS with % of peak; the best in each row bold."""
    meta = res.meta
    backends = meta["backends"]
    threads = sorted({meta["threads"][0], meta["threads"][-1]})
    lines = []
    for op in meta["ops"]:
        lines += [f"### {op}", "",
                  "| shape | max threads | " + " | ".join(backends) + " |",
                  "|---|---:|" + "---:|" * len(backends)]
        for s in res.shapes:
            for t in threads:
                ran = [(gflops_spread(res.points[(op, s, b, t)].gflops).mean, b)
                       for b in backends if (op, s, b, t) in res.points]
                best = max(ran)[1] if ran else None
                cells = []
                for b in backends:
                    cell = summary_cell(res, (op, s, b, t))
                    cells.append(f"**{cell}**" if b == best else cell)
                lines.append(f"| {shape_label(op, s)} | {t} | "
                             + " | ".join(cells) + " |")
        lines.append("")
    lines.append(summary_caption(meta))
    path.write_text("\n".join(lines) + "\n")
    print(path)


def summary_caption(meta: dict) -> str:
    """Hardware and run settings the table's numbers depend on."""
    g = trim_count(meta["rounds"])
    parts = [f"Mean GFLOPS over {meta['rounds']} round(s), the {g} lowest "
             f"and {g} highest trimmed (each round: median call time), "
             f"preset {meta.get('preset') or 'custom'}, "
             f"started {meta['started']}."]
    peak = meta.get("peak")
    if peak:
        parts.append(f"% of peak on {peak['cpu']}: "
                     f"{peak['flop_per_cycle_per_core']} flop/cycle/core at "
                     f"{clock_label(peak)}.")
    return "*" + " ".join(parts) + "*"


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
    xlabel = size_axis(shapes)
    if xlabel and len(shapes) >= 3:
        # GFLOPS says little where fixed per-call cost dominates.
        small = max(max(s) for s in shapes) <= 128
        for metric in metrics:
            if not (small and metric == "gflops"):
                plot_size(res, metric, sorted(shapes), xlabel, out)
    else:
        # Every file gets a per-shape chart; too few shapes for a size sweep
        # (e.g. the large preset's two) would otherwise leave none.
        for metric in metrics:
            plot_shapes(res, metric, shapes, out)
    if len(meta["threads"]) >= 3:
        for metric in metrics:
            plot_scaling(res, metric, shapes, out)
    if {"matmul", "dense"} <= set(meta["ops"]):
        plot_epilogue(res, shapes, out)
    write_summary(res, out / "summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
