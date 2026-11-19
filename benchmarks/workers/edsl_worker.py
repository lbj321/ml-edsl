#!/usr/bin/env python3
"""Time one float32 EDSL matmul or dense layer and print a JSON line.

Usage:
    venv/bin/python benchmarks/workers/edsl_worker.py M N K --threads T
        [--op matmul|dense]

Runs in the project venv against this checkout's mlir_edsl. With X: MxK,
W: KxN and b: N, times X @ W (matmul) or relu(X @ W + b) (dense), compiled
at O3. The result is checked against NumPy once before timing. GFLOPS counts
2*M*N*K for both ops. Only the JSON line goes to stdout.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse the shape, op and thread count."""
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


def main() -> int:
    """Compile, verify and time the op, then print one JSON result line."""
    args = parse_args()

    # The JIT's OpenMP runtime reads this when it loads on the first call.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
    from mlir_edsl import Tensor, f32, matmul, ml_function, relu
    from mlir_edsl.backend import get_backend
    from common import run_timed, write_samples

    get_backend().set_optimization_level(3)

    m, n, k = args.m, args.n, args.k
    rng = np.random.default_rng(0)
    x = rng.random((m, k), dtype=np.float32)
    w = rng.random((k, n), dtype=np.float32)
    bias = rng.random(n, dtype=np.float32)

    if args.op == "matmul":
        @ml_function
        def fn(X: Tensor[f32, m, k],
               W: Tensor[f32, k, n]) -> Tensor[f32, m, n]:
            return matmul(X, W)
        inputs = (x, w)
        expected = x @ w
    elif args.op == "dense":
        @ml_function
        def fn(X: Tensor[f32, m, k], W: Tensor[f32, k, n],
               b: Tensor[f32, n]) -> Tensor[f32, m, n]:
            return relu(matmul(X, W) + b)
        inputs = (x, w, bias)
        expected = np.maximum(x @ w + bias, 0.0)
    else:
        raise ValueError(f"unknown op {args.op!r}")

    t0 = time.perf_counter()
    got = fn(*inputs)
    compile_s = time.perf_counter() - t0
    np.testing.assert_allclose(got, expected, rtol=1e-3, atol=1e-3)

    def call() -> None:
        fn(*inputs)

    meas = run_timed(call, round((m * n * k) ** (1 / 3)), args.min_time,
                     args.repeats)
    timing = meas.timing
    if args.dump_samples:
        write_samples(args.dump_samples, meas)

    result = {
        "framework": "edsl",
        "op": args.op,
        "m": m, "n": n, "k": k,
        "threads": args.threads,
        "repeats": meas.repeats,
        "compile_s": compile_s,
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
