#!/usr/bin/env python3
"""Time one float32 NumPy matmul or dense layer and print a JSON line.

Usage:
    <env>/bin/python benchmarks/workers/numpy_worker.py M N K --threads T
        [--op matmul|dense]

Runs in any of the benchmarks/envs/np-* conda envs. With X: MxK, W: KxN and
b: N, times X @ W (matmul) or relu(X @ W + b) (dense), reporting which BLAS
library NumPy actually loaded so the orchestrator can reject a run on the
wrong one. GFLOPS counts 2*M*N*K for both ops. Only the JSON line goes to
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
    parser.add_argument("--op", choices=("matmul", "dense"), default="matmul")
    parser.add_argument("--repeats", type=int, default=None,
                        help="samples to time; overrides --min-time")
    parser.add_argument("--min-time", type=float, default=3.0,
                        help="seconds of timed calls to cover (default: 3)")
    parser.add_argument("--dump-samples", type=Path, default=None,
                        help="write every raw duration to this JSON file")
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
    from common import run_timed, write_samples

    blas = blas_info(threadpoolctl.threadpool_info())
    if blas["num_threads"] != args.threads:
        raise RuntimeError(f"requested {args.threads} threads, "
                           f"{blas['internal_api']} reports "
                           f"{blas['num_threads']}")

    m, n, k = args.m, args.n, args.k
    rng = np.random.default_rng(0)
    x = rng.random((m, k), dtype=np.float32)
    w = rng.random((k, n), dtype=np.float32)
    bias = rng.random(n, dtype=np.float32)

    # Allocate the output per call, as the EDSL call and bench_matmul.py do.
    # The dense epilogue updates it in place: still unfused passes over C,
    # but without bench_dense_layer.py's two extra temporaries.
    if args.op == "matmul":
        def call() -> None:
            np.matmul(x, w)
    elif args.op == "dense":
        def call() -> None:
            c = np.matmul(x, w)
            c += bias
            np.maximum(c, 0.0, out=c)
    else:
        raise ValueError(f"unknown op {args.op!r}")

    meas = run_timed(call, round((m * n * k) ** (1 / 3)), args.min_time,
                     args.repeats)
    timing = meas.timing
    if args.dump_samples:
        write_samples(args.dump_samples, meas)

    result = {
        "framework": "numpy",
        "op": args.op,
        "blas": blas["internal_api"],
        "blas_version": blas["version"],
        # Only OpenBLAS and BLIS report a kernel architecture; MKL does not.
        "blas_arch": blas.get("architecture"),
        "blas_threading": blas["threading_layer"],
        "numpy_version": np.__version__,
        "m": m, "n": n, "k": k,
        "threads": args.threads,
        "repeats": meas.repeats,
        "cpu_per_wall": meas.cpu_per_wall,
        "median_s": timing.median,
        "p10_s": timing.p10,
        "p90_s": timing.p90,
        "mean_s": timing.mean,
        "stdev_s": timing.stdev,
        "spread": timing.spread,
        "gflops": 2 * m * n * k / timing.median / 1e9,
        "samples_file": str(args.dump_samples) if args.dump_samples else None,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
