#!/usr/bin/env python3
"""OpenBLAS reference for the Stage 6 scaling table: numpy float32 A @ B.

Usage: bench_openblas.py M N K THREADS [repeats]

Thread count goes through OPENBLAS_NUM_THREADS, which OpenBLAS reads at load
time, so it is set here before numpy is imported. Run under the same
`taskset -c 0-7` as bench_matmul for a like-for-like comparison. Prints the
same "GFLOPS=" line format as bench_matmul (best of `repeats`, wall clock).
"""

import os
import sys
import time


def main() -> int:
    """Time best-of-N float32 matmul and print GFLOPS."""
    if len(sys.argv) < 5:
        print(__doc__, file=sys.stderr)
        return 1
    m, n, k, threads = (int(x) for x in sys.argv[1:5])
    repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    import numpy as np  # after OPENBLAS_NUM_THREADS is set

    rng = np.random.default_rng(0)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)
    for _ in range(3):
        a @ b
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    print(f"openblas M={m} N={n} K={k} threads={threads} best_of={repeats} "
          f"best_time={best:.6f}s GFLOPS={2 * m * n * k / best / 1e9:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
