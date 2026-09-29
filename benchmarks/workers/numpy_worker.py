#!/usr/bin/env python3
"""Time one float32 NumPy matmul and print the result as a JSON line.

Usage:
    <env>/bin/python benchmarks/workers/numpy_worker.py M N K --threads T

Runs in any of the benchmarks/envs/np-* conda envs. Times C = A @ B with
A: MxK and B: KxN, reporting which BLAS library NumPy actually loaded so the
orchestrator can reject a run on the wrong one. Only the JSON line goes to
stdout; anything else goes to stderr.
"""

import argparse
import json
import os
import sys
from pathlib import Path

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
)


def parse_args() -> argparse.Namespace:
    """Parse the matmul shape and thread count."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=None,
                        help="samples to time; overrides --min-time")
    parser.add_argument("--min-time", type=float, default=3.0,
                        help="seconds of timed calls to cover (default: 3)")
    return parser.parse_args()


def blas_info(threadpool_info: list) -> dict:
    """Return the single BLAS entry from threadpoolctl.threadpool_info().

    The MKL env also lists its OpenMP runtime, so the first entry is not
    necessarily the BLAS.
    """
    blas = [d for d in threadpool_info if d["user_api"] == "blas"]
    if len(blas) != 1:
        raise RuntimeError(f"expected exactly one BLAS library, got {blas}")
    return blas[0]


def main() -> int:
    """Time the matmul and print one JSON result line."""
    args = parse_args()

    # BLAS libraries read these when they load, so they must be set before
    # numpy is imported.
    for var in THREAD_ENV_VARS:
        os.environ[var] = str(args.threads)

    import numpy as np
    import threadpoolctl

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import WARMUP, repeats_for_budget, time_call_with_cpu

    blas = blas_info(threadpoolctl.threadpool_info())
    if blas["num_threads"] != args.threads:
        raise RuntimeError(f"requested {args.threads} threads, "
                           f"{blas['internal_api']} reports "
                           f"{blas['num_threads']}")

    m, n, k = args.m, args.n, args.k
    rng = np.random.default_rng(0)
    a = rng.random((m, k), dtype=np.float32)
    b = rng.random((k, n), dtype=np.float32)

    # Allocate the output per call, as the EDSL call and bench_matmul.py do.
    def call() -> None:
        np.matmul(a, b)

    for _ in range(WARMUP):
        call()
    repeats = args.repeats or repeats_for_budget(
        call, round((m * n * k) ** (1 / 3)), args.min_time)
    timing, cpu_per_wall = time_call_with_cpu(call, repeats)

    result = {
        "framework": "numpy",
        "blas": blas["internal_api"],
        "blas_version": blas["version"],
        # Only OpenBLAS and BLIS report a kernel architecture; MKL does not.
        "blas_arch": blas.get("architecture"),
        "blas_threading": blas["threading_layer"],
        "numpy_version": np.__version__,
        "m": m, "n": n, "k": k,
        "threads": args.threads,
        "repeats": repeats,
        "cpu_per_wall": cpu_per_wall,
        "median_s": timing.median,
        "p10_s": timing.p10,
        "p90_s": timing.p90,
        "spread": timing.spread,
        "gflops": 2 * m * n * k / timing.median / 1e9,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
