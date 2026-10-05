"""Load a run_matrix.py results file and summarise it over rounds.

Rounds are separate processes, so the spread between their medians is the
run-to-run noise; the spread of samples within one process is far smaller
and not used here. Ratios between backends are taken within each round,
where both ran minutes apart, then summarised over rounds. This cancels
drift across the session that dividing two medians would keep.

Stdlib only: run_matrix.py and plot_results.py both import it.
"""

import json
import statistics
from pathlib import Path
from typing import NamedTuple

BASELINE = "mkl"


class Point(NamedTuple):
    """One grid point's values, each keyed by round index."""
    gflops: dict
    secs: dict


class Spread(NamedTuple):
    """Median over rounds and the between-round min and max."""
    median: float
    lo: float
    hi: float


class Results(NamedTuple):
    """A results file grouped by (op, (m, n, k), backend, threads)."""
    meta: dict
    shapes: list
    points: dict
    failed: list


def parse_shape(shape: str) -> tuple:
    """'64x4096x1024' -> (64, 4096, 1024)."""
    parts = shape.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape {shape!r} is not MxNxK")
    return tuple(int(p) for p in parts)


def load(path: Path) -> Results:
    """Group a results file's records by grid point; failed records are
    kept apart, unaggregated."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    meta, records = lines[0]["meta"], lines[1:]
    points, failed = {}, []
    for r in records:
        if "error" in r:
            failed.append(r)
            continue
        key = (r["op"], (r["m"], r["n"], r["k"]), r["backend"], r["threads"])
        point = points.setdefault(key, Point({}, {}))
        point.gflops[r["round"]] = r["gflops"]
        point.secs[r["round"]] = r["median_s"]
    shapes = [parse_shape(s) for s in meta["shapes"]]
    return Results(meta, shapes, points, failed)


def spread(per_round: dict) -> Spread:
    """Spread of a round -> value map (GFLOPS, seconds or ratios)."""
    vals = list(per_round.values())
    return Spread(statistics.median(vals), min(vals), max(vals))


def paired_ratios(res: Results, key: tuple,
                  baseline: str = BASELINE) -> dict:
    """round -> GFLOPS of `key` / GFLOPS of the same point on `baseline`,
    for the rounds where both ran."""
    op, shape, _, threads = key
    point = res.points.get(key)
    base = res.points.get((op, shape, baseline, threads))
    if point is None or base is None:
        return {}
    return {rnd: gf / base.gflops[rnd] for rnd, gf in point.gflops.items()
            if rnd in base.gflops}


def geomean_ratios(res: Results, backend: str, threads: int,
                   baseline: str = BASELINE) -> dict:
    """round -> geometric mean over all ops and shapes of paired_ratios."""
    by_round = {}
    for op in res.meta["ops"]:
        for shape in res.shapes:
            ratios = paired_ratios(res, (op, shape, backend, threads),
                                   baseline)
            for rnd, ratio in ratios.items():
                by_round.setdefault(rnd, []).append(ratio)
    return {rnd: statistics.geometric_mean(v)
            for rnd, v in sorted(by_round.items())}
