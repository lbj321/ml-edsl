"""Dense layer benchmark: EDSL (O3) vs NumPy across matrix sizes.

Measures relu(X @ W + b) for square NxN matrices.

Usage:
    python3 benchmarks/bench_dense_layer.py
"""

import time
import numpy as np

from mlir_edsl import ml_function, Tensor, f32, relu
from mlir_edsl.backend import get_backend

from common import SIZES, WARMUP, best_of, repeats_for, print_section


def make_edsl_dense(N: int):
    """Compile a fixed-size NxN dense layer function. Returns (callable, compile_time_seconds)."""
    @ml_function
    def dense_fn(X: Tensor[f32, N, N], W: Tensor[f32, N, N], b: Tensor[f32, N]) -> Tensor[f32, N, N]:
        return relu(X @ W + b)

    X = np.ones((N, N), dtype=np.float32)
    W = np.ones((N, N), dtype=np.float32)
    b = np.zeros(N, dtype=np.float32)
    t0 = time.perf_counter()
    dense_fn(X, W, b)
    return dense_fn, time.perf_counter() - t0



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

        numpy_times[N] = best_of(lambda: np.maximum(X @ W + b, 0.0), n)
        print(f"  numpy {N:>3}x{N}: {numpy_times[N] * 1e6:.2f} µs")

    # Phase 2: EDSL benchmarks (triggers libomp RTLD_GLOBAL on first call).
    backend = get_backend()
    backend.set_optimization_level(3)
    print("\nPhase 2: EDSL...")
    edsl_times = {}
    compile_times = {}

    for N in SIZES:
        X, W, b = inputs[N]
        n = repeats_for(N)

        edsl_fn, compile_times[N] = make_edsl_dense(N)
        for _ in range(WARMUP):
            edsl_fn(X, W, b)

        edsl_times[N] = best_of(lambda: edsl_fn(X, W, b), n)
        print(f"  edsl  {N:>3}x{N}: call={edsl_times[N] * 1e6:.2f} µs  compile={compile_times[N] * 1e3:.1f} ms")

    rows = [(f"{N:>4}x{N:<2}", edsl_times[N], numpy_times[N], compile_times[N]) for N in SIZES]
    print_section("Dense Layer: relu(X @ W + b)", rows)


if __name__ == "__main__":
    main()
