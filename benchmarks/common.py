"""Shared helpers for EDSL vs NumPy benchmarks."""

import json
import math
import statistics
import time
import timeit
from pathlib import Path
from typing import Callable, NamedTuple, Optional

SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WARMUP = 5
BUDGET_CALLS = 5
# Keeps a min_time budget from turning a microsecond call into millions of
# samples. 10k pins one process's median to ~0.03%, far inside the 1-3%
# that medians vary between processes, so more samples only cost time.
MAX_REPEATS = 10_000


class Timing(NamedTuple):
    """One measurement: the reported time plus the spread it came from."""

    median: float
    p10: float
    p90: float
    # Diagnostic only: a mean well above the median flags a heavy tail or a
    # second mode that p10/p90 can miss.
    mean: float
    stdev: float

    @property
    def spread(self) -> float:
        """p90/p10. 1.0 is perfectly stable; above ~1.3 the median is not a
        meaningful summary and the run should be repeated.

        Deliberately percentiles rather than max/min: the extremes grow with
        the sample count, so max/min would flag a 10,000-sample row on one
        stray outlier while calling a noisy 25-sample row clean — the same
        sample-count bias that makes the minimum a poor headline figure."""
        return self.p90 / self.p10 if self.p10 > 0 else float("inf")


def measure(fn: Callable[[], object], n: int) -> list:
    """Time `n` calls to `fn`, returning the durations in the order they ran.

    timeit disables GC while timing; use this for every phase (warmup,
    budget estimate, samples) so they differ only in when they ran."""
    return timeit.repeat(fn, number=1, repeat=n)


def time_call(fn: Callable[[], object], n: int) -> Timing:
    """Time `fn` over `n` samples and summarize them."""
    return summarize(measure(fn, n))


def summarize(samples: list) -> Timing:
    """Median and interpolated p10/p90 of `samples`.

    Median, not minimum. The minimum is the right estimator when noise is
    purely additive — scheduling hiccups, interrupts, page faults — which is
    the case for the small sizes here, where a call is a few microseconds and
    dominated by fixed overhead.

    It is the wrong estimator for the large sizes. A sustained AVX2 FMA load
    drops the core off its turbo clock, so the first samples are genuinely
    faster than steady state and the minimum reports a burst rate the kernel
    cannot hold. Measured on this project: one unchanged matmul binary
    reported 157, 292 and 196 GFLOPS purely by varying the sample count.

    The minimum is also biased by the sample count — min-of-10000 lands much closer
    to the true floor than min-of-5 — so with `repeats_for` scaling the count
    down for large N, minima are not comparable across rows of one table.
    Medians are.
    """
    samples = sorted(samples)
    median = statistics.median(samples)
    mean = statistics.fmean(samples)

    # A single sample has no spread to report; say so rather than letting
    # statistics.quantiles raise.
    if len(samples) < 2:
        return Timing(median, median, median, mean, 0.0)

    # Interpolated deciles, not nearest-rank: a nearest-rank p10 collapses onto
    # samples[0] once n <= 10, which would quietly turn the spread indicator
    # back into the max/min ratio it exists to avoid.
    deciles = statistics.quantiles(samples, n=10, method="inclusive")
    return Timing(median, deciles[0], deciles[8], mean,
                  statistics.stdev(samples))


def repeats_for(N: int) -> int:
    """Samples per measurement, scaled to keep total runtime reasonable.

    The large-N counts are deliberately not tiny: at 5 samples the occasional
    30x outlier seen on this machine (page faults, OpenMP pool wake-up) can
    invert which implementation looks faster. 25 samples costs ~2.5s at
    2048^3 and makes the median stable.
    """
    if N <= 8:
        return 10_000
    if N <= 32:
        return 1_000
    if N <= 256:
        return 200
    if N <= 512:
        return 150
    return 25


def repeats_for_budget(budget_samples: list, N: int, min_time: float) -> int:
    """Samples covering about `min_time` seconds of calls, given a few
    timed calls taken after warmup: at most MAX_REPEATS (so µs calls cover
    less) and no fewer than repeats_for(N).

    A fixed count at large N is over in ~0.1 s multithreaded, before the
    clock has dropped to what it sustains, so short runs report burst rates
    that differ per library. The estimate is the median of the given calls,
    not a mean, so one slow call cannot shrink the budget.
    """
    t_call = statistics.median(budget_samples)
    budget = math.ceil(min_time / t_call) if t_call > 0 else MAX_REPEATS
    return max(repeats_for(N), min(budget, MAX_REPEATS))


class Measurement(NamedTuple):
    """One worker's timing run, with the raw durations of every phase."""

    timing: Timing
    cpu_per_wall: float
    repeats: int
    warmup: list
    budget: list
    samples: list


def run_timed(fn: Callable[[], object], N: int, min_time: float,
              repeats: Optional[int] = None) -> Measurement:
    """Warm up, size the sample count (unless `repeats` is given), then time.

    cpu_per_wall is process CPU seconds per wall second over the samples
    only: roughly how many threads were busy, including any spin-waiting.
    """
    warmup = measure(fn, WARMUP)
    budget = []
    if repeats is None:
        budget = measure(fn, BUDGET_CALLS)
        repeats = repeats_for_budget(budget, N, min_time)

    wall0, cpu0 = time.perf_counter(), time.process_time()
    samples = measure(fn, repeats)
    cpu_per_wall = ((time.process_time() - cpu0)
                    / (time.perf_counter() - wall0))
    return Measurement(summarize(samples), cpu_per_wall, repeats,
                       warmup, budget, samples)


def write_samples(path: Path, meas: Measurement) -> None:
    """Write every phase's raw durations, in timed order, as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"warmup": meas.warmup,
                                "budget": meas.budget,
                                "samples": meas.samples}))


def print_section(title: str, rows: list):
    """Render one table. `rows` holds (label, edsl, numpy, compile_seconds),
    where edsl and numpy are Timing objects."""
    print(f"\n=== {title} ===")
    # The spread column is 9 wide (6 digits + "x" + a 2-char flag), but only
    # its first 7 hold the number, so the header is padded to 7 and the gap
    # before Compile absorbs the flag.
    print(
        f"{'Size':>9}  {'Call (µs)':>12}  {'NumPy (µs)':>12}  {'Ratio':>8}"
        f"  {'Spread':>7}    {'Compile (ms)':>14}"
    )
    print("-" * 74)
    for size_label, edsl, numpy_t, compile_t in rows:
        ratio = edsl.median / numpy_t.median
        # Flag rows where the spread makes the median untrustworthy.
        mark = " !" if edsl.spread > 1.3 else ""
        print(
            f"{size_label:>9}  {edsl.median * 1e6:>12.2f}  "
            f"{numpy_t.median * 1e6:>12.2f}  {ratio:>7.2f}x  "
            f"{edsl.spread:>6.1f}x{mark:<2}  {compile_t * 1e3:>14.1f}"
        )
    if any(edsl.spread > 1.3 for _, edsl, _, _ in rows):
        print("  ! spread > 1.3x — median unreliable, re-run on an idle machine")
