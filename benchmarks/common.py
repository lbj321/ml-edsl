"""Shared helpers for EDSL vs NumPy benchmarks."""

import statistics
import timeit
from typing import Callable, NamedTuple

SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WARMUP = 5


class Timing(NamedTuple):
    """One measurement: the reported time plus the spread it came from."""

    median: float
    p10: float
    p90: float

    @property
    def spread(self) -> float:
        """p90/p10. 1.0 is perfectly stable; above ~1.3 the median is not a
        meaningful summary and the run should be repeated.

        Deliberately percentiles rather than max/min: the extremes grow with
        the sample count, so max/min would flag a 10,000-sample row on one
        stray outlier while calling a noisy 25-sample row clean — the same
        sample-count bias that makes the minimum a poor headline figure."""
        return self.p90 / self.p10 if self.p10 > 0 else float("inf")


def time_call(fn: Callable[[], object], n: int) -> Timing:
    """Time `fn` over `n` samples and report the median.

    Median, not minimum. The minimum is the right estimator when noise is
    purely additive — scheduling hiccups, interrupts, page faults — which is
    the case for the small sizes here, where a call is a few microseconds and
    dominated by fixed overhead.

    It is the wrong estimator for the large sizes. A sustained AVX2 FMA load
    drops the core off its turbo clock, so the first samples are genuinely
    faster than steady state and the minimum reports a burst rate the kernel
    cannot hold. Measured on this project: one unchanged matmul binary
    reported 157, 292 and 196 GFLOPS purely by varying the sample count.

    The minimum is also biased by `n` itself — min-of-10000 lands much closer
    to the true floor than min-of-5 — so with `repeats_for` scaling the count
    down for large N, minima are not comparable across rows of one table.
    Medians are.
    """
    samples = sorted(timeit.repeat(fn, number=1, repeat=n))
    median = statistics.median(samples)

    # A single sample has no spread to report; say so rather than letting
    # statistics.quantiles raise.
    if len(samples) < 2:
        return Timing(median, median, median)

    # Interpolated deciles, not nearest-rank: a nearest-rank p10 collapses onto
    # samples[0] once n <= 10, which would quietly turn the spread indicator
    # back into the max/min ratio it exists to avoid.
    deciles = statistics.quantiles(samples, n=10, method="inclusive")
    return Timing(median, deciles[0], deciles[8])


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
