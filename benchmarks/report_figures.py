#!/usr/bin/env python3
"""Draw the report's figures and table from one report session.

Usage:
    python3 benchmarks/report_figures.py benchmarks/results/<session dir>
        [--out DIR] [--paper]

Needs matplotlib (the system python3 has it; the project venv does not).
Reads the session's overhead, scaling, large and ml files and writes, into
DIR (default: benchmarks/results/plots/<session dir>/report/, or .../paper/
with --paper):
  fig1_size          matmul GFLOPS against size at 1 and 8 threads
                     (scaling + large): where EDSL stands
  fig2_threads       matmul GFLOPS against threads at 1024^3 and 256^3:
                     whether it scales with cores
  fig3_epilogue      dense / matmul time per backend at the square sizes,
                     1 and 8 threads: what adding bias + ReLU costs over
                     plain matmul, however each backend builds it (a
                     compiler may emit a different matmul for dense)
  fig4_ml            dense layer GFLOPS against batch at 1 and 8 threads:
                     whether it handles real ML layers
  fig5_overhead      time per call at 8-128^3, 1 thread: fixed cost
  table.md           the numbers behind them, plus dense / matmul time
Figures are PNG with title and subtitle by default. --paper writes vector
PDF without them, for a document that captions each figure itself, plus
captions.txt holding the title and subtitle text to caption them with.
Everything else stays in plot_results.py's per-preset output.
"""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

from aggregate import Results, epilogue_ratios, load, ratio_spread
from plot_results import (RESULTS_DIR, epilogue_panel, figure, finish,
                          gflops_row, run_description, scaling_panel,
                          shape_label, size_axis, size_panel, summary_caption)

PRESETS = ("overhead", "scaling", "large", "ml")
# Fields that must agree for two files' numbers to share one figure.
SHARED_META = ("rounds", "backends", "peak", "edsl")


class Output(NamedTuple):
    """Where figures go and in which form; `captions` collects the text a
    paper figure leaves out."""
    dir: Path
    paper: bool
    captions: list


def start(out: Output, stem: str, panels: int, title: str, meta: dict,
          kind: str) -> tuple:
    """A figure for `out`: with its header, or without it and the header's
    text saved as a caption."""
    if out.paper:
        out.captions.append(f"{stem}: {title}. "
                            + "; ".join(run_description(meta, kind)) + ".")
    return figure(panels, title, meta, kind, header=not out.paper)


def save(out: Output, fig, axes: list, stem: str) -> None:
    """Save as PDF for a paper, PNG otherwise."""
    finish(fig, axes, out.dir / f"{stem}.{'pdf' if out.paper else 'png'}")


def load_session(session: Path) -> dict:
    """preset -> Results for every report preset, refusing a session that
    lacks one or mixes builds or settings across them."""
    runs = {}
    for preset in PRESETS:
        path = session / f"{preset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: not a full report "
                                    "session")
        runs[preset] = load(path)
    first = runs[PRESETS[0]].meta
    for preset, res in runs.items():
        for field in SHARED_META:
            if res.meta.get(field) != first.get(field):
                raise ValueError(f"{preset} differs from {PRESETS[0]} in "
                                 f"{field!r}; figures would mix runs")
    return runs


def merge(runs: list, label: str) -> Results:
    """One Results over several files of the same session, e.g. the square
    sizes of scaling and large; `label` names them in subtitles."""
    meta = dict(runs[0].meta, preset=label)
    meta["threads"] = sorted({t for r in runs for t in r.meta["threads"]})
    meta["ops"] = [op for op in runs[0].meta["ops"]
                   if all(op in r.meta["ops"] for r in runs)]
    points, failed = {}, []
    for r in runs:
        points.update(r.points)
        failed += r.failed
    shapes = sorted({s for r in runs for s in r.shapes})
    return Results(meta, shapes, points, failed)


def shapes_run(res: Results, op: str, threads: int) -> list:
    """The shapes some backend ran for `op` at `threads`, in order."""
    return [s for s in res.shapes
            if any((op, s, b, threads) in res.points
                   for b in res.meta["backends"])]


def fig_size(square: Results, out: Output) -> None:
    """Figure 1: matmul GFLOPS against size, 1 thread and all threads."""
    stem = "fig1_size"
    threads = (1, square.meta["threads"][-1])
    fig, axes = start(out, stem, len(threads), "Matmul GFLOPS vs size",
                      square.meta, "bars")
    for ax, t in zip(axes, threads):
        shapes = shapes_run(square, "matmul", t)
        size_panel(ax, square, "matmul", t, shapes, "gflops",
                   size_axis(shapes))
    save(out, fig, axes, stem)


def fig_threads(scaling: Results, out: Output) -> None:
    """Figure 2: matmul GFLOPS against threads at a typical size and the
    smallest, where threading stalls."""
    stem = "fig2_threads"
    shapes = [(1024, 1024, 1024), (256, 256, 256)]
    fig, axes = start(out, stem, len(shapes), "Matmul GFLOPS vs threads",
                      scaling.meta, "lines")
    for ax, s in zip(axes, shapes):
        scaling_panel(ax, scaling, "matmul", s, "gflops")
    save(out, fig, axes, stem)


def fig_epilogue(square: Results, out: Output) -> None:
    """Figure 3: dense / matmul time per backend, 1 and all threads."""
    stem = "fig3_epilogue"
    threads = (1, square.meta["threads"][-1])
    fig, axes = start(out, stem, len(threads), "Dense layer time / matmul "
                      "time  (dashed line = same time as matmul)",
                      square.meta, "bars")
    for ax, t in zip(axes, threads):
        epilogue_panel(ax, square, t, shapes_run(square, "dense", t))
    save(out, fig, axes, stem)


def fig_ml(ml: Results, out: Output) -> None:
    """Figure 4: dense layer GFLOPS against batch, 1 and all threads."""
    stem = "fig4_ml"
    threads = (1, ml.meta["threads"][-1])
    shapes = sorted(ml.shapes)
    fig, axes = start(out, stem, len(threads), "Dense layer GFLOPS vs batch",
                      ml.meta, "bars")
    for ax, t in zip(axes, threads):
        size_panel(ax, ml, "dense", t, shapes, "gflops", size_axis(shapes))
    save(out, fig, axes, stem)


def fig_overhead(overhead: Results, out: Output) -> None:
    """Figure 5: time per call at small sizes, 1 thread; at 8 threads
    thread start-up, not the call, dominates for some libraries."""
    stem = "fig5_overhead"
    shapes = sorted(overhead.shapes)
    fig, axes = start(out, stem, 1, "Time per call, small sizes",
                      overhead.meta, "bars")
    size_panel(axes[0], overhead, "matmul", 1, shapes, "time",
               size_axis(shapes))
    save(out, fig, axes, stem)


def gflops_table(res: Results, op: str, title: str) -> list:
    """Markdown lines: GFLOPS with % of peak per shape at 1 and all
    threads."""
    backends = res.meta["backends"]
    lines = [f"### {title}", "",
             "| shape | max threads | " + " | ".join(backends) + " |",
             "|---|---:|" + "---:|" * len(backends)]
    for t in (1, res.meta["threads"][-1]):
        for s in shapes_run(res, op, t):
            lines.append(gflops_row(res, op, s, t))
    return lines + [""]


def epilogue_table(square: Results) -> list:
    """Markdown lines: dense / matmul time per backend, paired per round;
    1.00 means dense takes as long as plain matmul."""
    backends = square.meta["backends"]
    lines = ["### Dense / matmul time (1.00 = same time as matmul)", "",
             "| shape | max threads | " + " | ".join(backends) + " |",
             "|---|---:|" + "---:|" * len(backends)]
    for t in (1, square.meta["threads"][-1]):
        for s in shapes_run(square, "dense", t):
            cells = []
            for b in backends:
                ratios = epilogue_ratios(square, s, b, t)
                cells.append(f"{ratio_spread(ratios).mean:.2f}"
                             if ratios else "-")
            lines.append(f"| {shape_label('dense', s)} | {t} | "
                         + " | ".join(cells) + " |")
    return lines + [""]


def write_table(square: Results, ml: Results, path: Path) -> None:
    """table.md: square matmul, dense layers over batch, epilogue cost."""
    lines = (gflops_table(square, "matmul", "Matmul, square")
             + gflops_table(ml, "dense", "Dense layer over batch")
             + epilogue_table(square)
             + [summary_caption(square.meta)])
    path.write_text("\n".join(lines) + "\n")
    print(path)


def main() -> int:
    """Draw every report figure and the table for one session."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--paper", action="store_true",
                        help="PDF without title or subtitle, plus "
                             "captions.txt")
    args = parser.parse_args()

    runs = load_session(args.session)
    for preset, res in runs.items():
        if res.failed:
            print(f"note: {preset}: {len(res.failed)} failed point(s) left "
                  "out", file=sys.stderr)
    folder = "paper" if args.paper else "report"
    out = Output(args.out or RESULTS_DIR / "plots" / args.session.name
                 / folder, args.paper, [])
    out.dir.mkdir(parents=True, exist_ok=True)

    square = merge([runs["scaling"], runs["large"]], "scaling + large")
    fig_size(square, out)
    fig_threads(runs["scaling"], out)
    fig_epilogue(square, out)
    fig_ml(runs["ml"], out)
    fig_overhead(runs["overhead"], out)
    write_table(square, runs["ml"], out.dir / "table.md")
    if out.paper:
        captions = out.dir / "captions.txt"
        captions.write_text("\n\n".join(out.captions) + "\n")
        print(captions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
