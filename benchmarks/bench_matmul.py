"""Matmul benchmark: EDSL (O3) vs NumPy across matrix sizes.

Usage:
    python3 benchmarks/bench_matmul.py

Each size is compiled once, warmed up, then timed with enough repeats
to get stable measurements even for small matrices.
"""

import timeit
import numpy as np

from mlir_edsl import ml_function, Tensor, f32, matmul
from mlir_edsl.backend import get_backend

SIZES = [2, 4, 8, 16, 32, 64]
WARMUP = 5


def repeats_for(N: int) -> int:
    """Scale repeat count down for large N to keep benchmark runtime reasonable."""
    if N <= 8:
        return 10_000
    if N <= 32:
        return 1_000
    return 200


def make_edsl_matmul(N: int):
    """Compile a fixed-size NxN matmul function. Returns the callable."""
    backend = get_backend()
    backend.clear_module()
    backend.set_optimization_level(3)

    @ml_function
    def matmul_fn(A: Tensor[f32, N, N], B: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
        return matmul(A, B)

    # Trigger JIT compilation
    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    matmul_fn(A, B)

    return matmul_fn


def benchmark_size(N: int):
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    edsl_fn = make_edsl_matmul(N)

    # Warmup
    for _ in range(WARMUP):
        edsl_fn(A, B)
        np.matmul(A, B)

    n = repeats_for(N)

    edsl_time = timeit.timeit(lambda: edsl_fn(A, B), number=n) / n
    numpy_time = timeit.timeit(lambda: np.matmul(A, B), number=n) / n

    return edsl_time, numpy_time


def main():
    print(f"{'Size':>6}  {'EDSL (µs)':>12}  {'NumPy (µs)':>12}  {'Ratio':>8}")
    print("-" * 46)

    for N in SIZES:
        edsl_t, numpy_t = benchmark_size(N)
        ratio = edsl_t / numpy_t
        print(f"{N:>4}x{N:<2}  {edsl_t * 1e6:>12.2f}  {numpy_t * 1e6:>12.2f}  {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
