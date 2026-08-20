"""Shared helpers for EDSL vs NumPy benchmarks."""

import timeit

SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
WARMUP = 5


def best_of(fn, n: int) -> float:
    """Minimum single-call time over `n` samples. More robust than a mean
    (timeit.timeit(...)/n): an averaged measurement gets dragged up by any
    scheduling hiccup among the n calls, while the minimum is only ever
    pulled toward the true best-case cost."""
    return min(timeit.repeat(fn, number=1, repeat=n))


def repeats_for(N: int) -> int:
    """Scale repeat count down for large N to keep benchmark runtime reasonable."""
    if N <= 8:
        return 10_000
    if N <= 32:
        return 1_000
    if N <= 256:
        return 200
    if N <= 512:
        return 150
    return 5


def print_section(title: str, rows: list):
    print(f"\n=== {title} ===")
    print(f"{'Size':>6}  {'Call (µs)':>12}  {'NumPy (µs)':>12}  {'Ratio':>8}  {'Compile (ms)':>14}")
    print("-" * 62)
    for size_label, edsl_t, numpy_t, compile_t in rows:
        ratio = edsl_t / numpy_t
        print(f"{size_label:>6}  {edsl_t * 1e6:>12.2f}  {numpy_t * 1e6:>12.2f}  {ratio:>7.2f}x  {compile_t * 1e3:>14.1f}")
