"""Dense layer benchmark: EDSL (O3) vs NumPy across matrix sizes.

Measures relu(X @ W + b) for square NxN matrices.

Usage:
    python3 benchmarks/bench_dense_layer.py
"""

import timeit
import numpy as np

from mlir_edsl import ml_function, Tensor, f32, relu
from mlir_edsl.backend import get_backend

SIZES = [2, 4, 8, 16, 32, 64, 128, 256]
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



def main():
    rng = np.random.default_rng(42)

    # Phase 1: all NumPy timings before any EDSL execution.
    # MLIRExecutor::initialize() loads libomp with RTLD_GLOBAL on the first
    # EDSL call, which can interfere with OpenBLAS threading.
    print("Phase 1: NumPy...")
    inputs = {}
    numpy_times = {}

    for N in SIZES:
        X = rng.random((N, N), dtype=np.float32)
        W = rng.random((N, N), dtype=np.float32)
        b = rng.random(N, dtype=np.float32)
        inputs[N] = (X, W, b)
        n = repeats_for(N)

        for _ in range(WARMUP):
            np.maximum(X @ W + b, 0.0)

        numpy_times[N] = timeit.timeit(lambda: np.maximum(X @ W + b, 0.0), number=n) / n
        print(f"  numpy {N:>3}x{N}: {numpy_times[N] * 1e6:.2f} µs")

    # Phase 2: EDSL benchmarks (triggers libomp RTLD_GLOBAL on first call).
    print("\nPhase 2: EDSL...")
    edsl_times = {}

    for N in SIZES:
        X, W, b = inputs[N]
        n = repeats_for(N)

        edsl_fn = make_edsl_dense(N)
        for _ in range(WARMUP):
            edsl_fn(X, W, b)

        edsl_times[N] = timeit.timeit(lambda: edsl_fn(X, W, b), number=n) / n
        print(f"  edsl  {N:>3}x{N}: {edsl_times[N] * 1e6:.2f} µs")

    print(f"\n{'Size':>6}  {'EDSL (µs)':>12}  {'NumPy (µs)':>12}  {'Ratio':>8}")
    print("-" * 46)
    for N in SIZES:
        ratio = edsl_times[N] / numpy_times[N]
        print(f"{N:>4}x{N:<2}  {edsl_times[N] * 1e6:>12.2f}  {numpy_times[N] * 1e6:>12.2f}  {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
