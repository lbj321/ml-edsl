#!/usr/bin/env python3
"""Time one float32 jitted JAX matmul or dense layer and print a JSON line.

Usage:
    <env>/bin/python benchmarks/workers/jax_worker.py M N K --threads T
        [--op matmul|dense]

Runs in the benchmarks/envs/jax conda env. With X: MxK, W: KxN and b: N,
times X @ W (matmul) or relu(X @ W + b) (dense) on the CPU device. GFLOPS
counts 2*M*N*K for both ops. Only the JSON line goes to stdout.

threadpoolctl cannot see XLA's thread pool, so the thread count is set by
restricting the process to T CPUs before jax is imported (XLA sizes its pool
from the affinity mask), and checked afterwards by the CPU time the timed
calls used per second of wall time.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


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


def restrict_to_threads(threads: int) -> list:
    """Pin the process to the first `threads` CPUs it is allowed on."""
    allowed = sorted(os.sched_getaffinity(0))
    if threads > len(allowed):
        raise RuntimeError(f"requested {threads} threads, only CPUs "
                           f"{allowed} are available")
    cpus = allowed[:threads]
    os.sched_setaffinity(0, cpus)
    return cpus


def main() -> int:
    """Time the matmul and print one JSON result line."""
    args = parse_args()

    cpus = restrict_to_threads(args.threads)
    os.environ["JAX_PLATFORMS"] = "cpu"
    if args.threads == 1:
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false "
            "intra_op_parallelism_threads=1")

    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from common import run_timed, write_samples

    m, n, k = args.m, args.n, args.k
    rng = np.random.default_rng(0)
    x = jax.device_put(rng.random((m, k), dtype=np.float32))
    w = jax.device_put(rng.random((k, n), dtype=np.float32))
    bias = jax.device_put(rng.random(n, dtype=np.float32))

    if args.op == "matmul":
        fn, inputs = jnp.matmul, (x, w)
    elif args.op == "dense":
        def fn(x, w, bias):
            return jnp.maximum(x @ w + bias, 0.0)
        inputs = (x, w, bias)
    else:
        raise ValueError(f"unknown op {args.op!r}")

    jitted = jax.jit(fn)
    t0 = time.perf_counter()
    jitted(*inputs).block_until_ready()
    compile_s = time.perf_counter() - t0

    def call() -> None:
        jitted(*inputs).block_until_ready()

    meas = run_timed(call, round((m * n * k) ** (1 / 3)), args.min_time,
                     args.repeats)
    timing = meas.timing
    if args.dump_samples:
        write_samples(args.dump_samples, meas)

    result = {
        "framework": "jax",
        "op": args.op,
        "jax_version": jax.__version__,
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "cpus": cpus,
        "m": m, "n": n, "k": k,
        "threads": args.threads,
        "repeats": meas.repeats,
        "compile_s": compile_s,
        "cpu_per_wall": meas.cpu_per_wall,
        "median_s": timing.median,
        "p10_s": timing.p10,
        "p90_s": timing.p90,
        "spread": timing.spread,
        "gflops": 2 * m * n * k / timing.median / 1e9,
        "samples_file": str(args.dump_samples) if args.dump_samples else None,
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
