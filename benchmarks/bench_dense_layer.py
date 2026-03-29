"""Dense layer benchmark: EDSL (O3) vs NumPy across matrix sizes.

Measures relu(X @ W + b) for square NxN matrices.

Usage:
    python3 benchmarks/bench_dense_layer.py
"""

import timeit
import numpy as np

from mlir_edsl import ml_function, Tensor, f32, relu
from mlir_edsl.backend import get_backend

SIZES = [2, 4, 8, 16, 32, 64, 128]
WARMUP = 5


def repeats_for(N: int) -> int:
    """Scale repeat count down for large N to keep benchmark runtime reasonable."""
    if N <= 8:
        return 10_000
    if N <= 32:
        return 1_000
    if N <= 128:
        return 200
    return 50


def make_edsl_dense(N: int):
    """Compile a fixed-size NxN dense layer function. Returns the callable."""
    backend = get_backend()
    backend.clear_module()
    backend.set_optimization_level(3)

    @ml_function
    def dense_fn(X: Tensor[f32, N, N], W: Tensor[f32, N, N], b: Tensor[f32, N]) -> Tensor[f32, N, N]:
        return relu(X @ W + b)

    # Trigger JIT compilation
    X = np.ones((N, N), dtype=np.float32)
    W = np.ones((N, N), dtype=np.float32)
    b = np.zeros(N, dtype=np.float32)
    dense_fn(X, W, b)

    return dense_fn


def benchmark_size(N: int):
    rng = np.random.default_rng(42)
    X = rng.random((N, N), dtype=np.float32)
    W = rng.random((N, N), dtype=np.float32)
    b = rng.random(N, dtype=np.float32)

    edsl_fn = make_edsl_dense(N)

    # Warmup
    for _ in range(WARMUP):
        edsl_fn(X, W, b)
        np.maximum(X @ W + b, 0.0)

    n = repeats_for(N)

    edsl_time = timeit.timeit(lambda: edsl_fn(X, W, b), number=n) / n
    numpy_time = timeit.timeit(lambda: np.maximum(X @ W + b, 0.0), number=n) / n

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
