"""Load a run_matrix.py results file and summarise it over rounds.

Each round is a separate process and contributes one value: the median of
its call times, which drops rare stalls (scheduler, interrupts) that a
process mean would turn into a random bias. Rounds are the independent
unit, each weighted equally whatever its sample count, and are summarised
by their TRIM trimmed mean (10 rounds: the 2 lowest and 2 highest dropped,
the middle 6 averaged) with Yuen's 95% confidence interval, built from the
winsorized variance. The mean is taken where the quantity adds up: seconds
directly, GFLOPS through their reciprocal (flops / mean time) and ratios
in log space (geometric).

Trimmed, because a whole round can hit a machine disturbance (20261123
report: 21 of 3100 rounds at 0.05-0.7x, e.g. every backend in one ml round
at 8T, two rounds at one point), and one such round moves a plain 10-round
mean by up to half. Trimming 2 per side absorbs up to two per point with
no threshold to choose; a 1-per-side trim left that two-round point 6% low.

The dense / matmul ratio is taken within each round, where both ops ran
minutes apart, which cancels drift across the session.

Stdlib only: run_matrix.py and plot_results.py both import it.
"""

import json
import math
import statistics
from pathlib import Path
from typing import Callable, NamedTuple, Optional

TRIM = 0.2

# Two-sided 95% t critical values by degrees of freedom (kept rounds - 1).
# The stdlib has no t distribution; more than 31 rounds is not expected.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
       7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
       13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
       19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
       25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


class Point(NamedTuple):
    """One grid point's values, each keyed by round index."""
    gflops: dict
    secs: dict


class Spread(NamedTuple):
    """Trimmed mean over rounds, its 95% confidence interval (lo, hi) and
    the between-round min and max of all rounds, trimmed or not. With one
    round all five are equal."""
    mean: float
    lo: float
    hi: float
    min: float
    max: float

    @property
    def half_ci(self) -> float:
        """Half the CI's width relative to the mean, for '±' columns."""
        return (self.hi - self.lo) / 2 / self.mean


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


def trim_count(n: int) -> int:
    """Rounds trimmed from each end of n: floor(TRIM * n), so 2 of 10 and
    none below 5 rounds."""
    return math.floor(TRIM * n)


def _trimmed_mean_ci(xs: list) -> tuple:
    """(trimmed mean, CI half-width) of xs by Yuen's method: the standard
    error comes from the winsorized sample (each trimmed value replaced by
    the nearest kept one) over the h kept values, with h - 1 df."""
    xs = sorted(xs)
    n, g = len(xs), trim_count(len(xs))
    h = n - 2 * g
    mean = statistics.fmean(xs[g:n - g])
    if h < 2:
        return mean, 0.0
    if h - 1 not in T95:
        raise ValueError(f"no t value for {h} kept rounds")
    wins = [xs[g]] * g + xs[g:n - g] + [xs[n - g - 1]] * g
    w_mean = statistics.fmean(wins)
    ssd = sum((w - w_mean) ** 2 for w in wins)
    return mean, T95[h - 1] * math.sqrt(ssd / (h * (h - 1)))


def _spread(per_round: dict, to_linear: Callable, from_linear: Callable,
            positive: bool) -> Spread:
    """Trimmed mean and Yuen interval of to_linear(values), mapped back by
    from_linear, a monotonic inverse; min and max stay in value units.
    `positive` clamps the interval at 0 in linear space, which a wide
    interval over few rounds can otherwise cross."""
    vals = list(per_round.values())
    if not vals:
        raise ValueError("no rounds to summarise")
    mean, half = _trimmed_mean_ci([to_linear(v) for v in vals])
    low = max(mean - half, 0.0) if positive else mean - half
    ends = sorted((from_linear(low), from_linear(mean + half)))
    return Spread(from_linear(mean), ends[0], ends[1], min(vals), max(vals))


def _reciprocal(v: float) -> float:
    """1 / v, with 0 (a clamped interval end) mapping to infinity."""
    return 1 / v if v > 0 else math.inf


def secs_spread(per_round: dict) -> Spread:
    """Seconds per call over rounds: arithmetic trimmed mean."""
    return _spread(per_round, lambda v: v, lambda v: v, positive=True)


def gflops_spread(per_round: dict) -> Spread:
    """GFLOPS over rounds via time per flop, so the mean is flops / mean
    time (harmonic), not a mean of rates; the same rounds are trimmed as
    for seconds, since the map is monotonic."""
    return _spread(per_round, _reciprocal, _reciprocal, positive=True)


def ratio_spread(per_round: dict) -> Spread:
    """Ratios over rounds in log space: geometric trimmed mean, so 2x and
    0.5x average to 1x; the interval is asymmetric around it."""
    return _spread(per_round, math.log, math.exp, positive=False)


def peak_gflops(meta: dict, threads: int) -> Optional[float]:
    """Peak GFLOPS at `threads` busy cores from the file's recorded peak,
    or None for files written before it was recorded."""
    peak = meta.get("peak")
    if peak is None:
        return None
    ghz = peak_clocks(peak)
    if not 1 <= threads <= len(ghz):
        raise ValueError(f"no clock for {threads} threads on "
                         f"{peak['cpu']} ({len(ghz)} cores)")
    return peak["flop_per_cycle_per_core"] * threads * ghz[threads - 1]


def peak_clocks(peak: dict) -> list:
    """GHz per active core count; older files call it turbo_ghz."""
    if "ghz" in peak:
        return peak["ghz"]
    if "turbo_ghz" in peak:
        return peak["turbo_ghz"]
    raise ValueError(f"peak for {peak.get('cpu')} records no clock")


def clock_label(peak: dict) -> str:
    """How the peak's clock was set, for captions and reports."""
    if "clock" in peak:
        return peak["clock"]
    ghz = peak_clocks(peak)
    return (f"assumed stock turbo ({ghz[0]} GHz on 1 core to {ghz[-1]} GHz "
            f"on {len(ghz)})")


def epilogue_ratios(res: Results, shape: tuple, backend: str,
                    threads: int) -> dict:
    """round -> dense time / matmul time of the same point, for the rounds
    where both ran; 1.0 means the epilogue is free."""
    dense = res.points.get(("dense", shape, backend, threads))
    matmul = res.points.get(("matmul", shape, backend, threads))
    if dense is None or matmul is None:
        return {}
    return {rnd: d / matmul.secs[rnd] for rnd, d in dense.secs.items()
            if rnd in matmul.secs}
