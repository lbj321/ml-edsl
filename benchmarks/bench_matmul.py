"""Matmul, bias add, and relu benchmark: EDSL (O3) vs NumPy across matrix sizes.

Usage:
    python3 benchmarks/bench_matmul.py

Each size is compiled once, warmed up, then timed with enough repeats
to get stable measurements even for small matrices.

"""

import time

import numpy as np

from mlir_edsl import ml_function, Tensor, f32, matmul, relu
from mlir_edsl.backend import get_backend

from common import SIZES, WARMUP, best_of, repeats_for, print_section


def make_edsl_matmul(N: int):
    """Compile a fixed-size NxN matmul function. Returns (callable, compile_time_seconds)."""
    @ml_function
    def matmul_fn(A: Tensor[f32, N, N], B: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
        return matmul(A, B)

    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    t0 = time.perf_counter()
    matmul_fn(A, B)
    return matmul_fn, time.perf_counter() - t0


def make_edsl_bias_add(N: int):
    """Compile a fixed-size NxN bias add function. Returns (callable, compile_time_seconds)."""
    @ml_function
    def bias_fn(X: Tensor[f32, N, N], b: Tensor[f32, N]) -> Tensor[f32, N, N]:
        return X + b

    X = np.ones((N, N), dtype=np.float32)
    b = np.ones(N, dtype=np.float32)
    t0 = time.perf_counter()
    bias_fn(X, b)
    return bias_fn, time.perf_counter() - t0


def make_edsl_relu(N: int):
    """Compile a fixed-size NxN relu function. Returns (callable, compile_time_seconds)."""
    @ml_function
    def relu_fn(X: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
        return relu(X)

    X = np.ones((N, N), dtype=np.float32)
    t0 = time.perf_counter()
    relu_fn(X)
    return relu_fn, time.perf_counter() - t0


def main():
    rng = np.random.default_rng(42)

    # Phase 1: collect all NumPy timings before any EDSL execution.
    # MLIRExecutor::initialize() loads libomp with RTLD_GLOBAL on the first
    # EDSL call, which can interfere with OpenBLAS threading. Running NumPy
    # first ensures it benchmarks without that interference.
    inputs = {}
    np_matmul_t = {}
    np_bias_t   = {}
    np_relu_t   = {}

    print("Phase 1: NumPy...")
    for N in SIZES:
        A = rng.random((N, N), dtype=np.float32)
        B = rng.random((N, N), dtype=np.float32)
        b = rng.random(N, dtype=np.float32)
        inputs[N] = (A, B, b)
        n = repeats_for(N)

        for _ in range(WARMUP):
            np.matmul(A, B)
            np.add(A, b)
            np.maximum(A, 0.0)

        np_matmul_t[N] = best_of(lambda: np.matmul(A, B), n)
        np_bias_t[N]   = best_of(lambda: np.add(A, b), n)
        np_relu_t[N]   = best_of(lambda: np.maximum(A, 0.0), n)
        print(f"  numpy {N:>3}x{N}: matmul={np_matmul_t[N]*1e6:.2f} µs  bias={np_bias_t[N]*1e6:.2f} µs  relu={np_relu_t[N]*1e6:.2f} µs")

    # Phase 2: EDSL benchmarks (triggers libomp RTLD_GLOBAL on first call).
    backend = get_backend()
    backend.set_optimization_level(3)
    print("\nPhase 2: EDSL...")
    matmul_rows = []
    bias_rows   = []
    relu_rows   = []

    for N in SIZES:
        A, B, b = inputs[N]
        n = repeats_for(N)
        label = f"{N:>4}x{N:<2}"

        edsl_fn, matmul_compile = make_edsl_matmul(N)
        for _ in range(WARMUP):
            edsl_fn(A, B)
        matmul_t = best_of(lambda: edsl_fn(A, B), n)
        matmul_rows.append((label, matmul_t, np_matmul_t[N], matmul_compile))

        edsl_fn, bias_compile = make_edsl_bias_add(N)
        for _ in range(WARMUP):
            edsl_fn(A, b)
        bias_t = best_of(lambda: edsl_fn(A, b), n)
        bias_rows.append((label, bias_t, np_bias_t[N], bias_compile))

        edsl_fn, relu_compile = make_edsl_relu(N)
        for _ in range(WARMUP):
            edsl_fn(A)
        relu_t = best_of(lambda: edsl_fn(A), n)
        relu_rows.append((label, relu_t, np_relu_t[N], relu_compile))

        print(f"  edsl  {N:>3}x{N}: matmul={matmul_t*1e6:.2f} µs (compile={matmul_compile*1e3:.1f} ms)  bias={bias_t*1e6:.2f} µs  relu={relu_t*1e6:.2f} µs")

    print()
    print_section("Matmul: X @ W", matmul_rows)
    print_section("Bias Add: X + b", bias_rows)
    print_section("ReLU: relu(X)", relu_rows)


if __name__ == "__main__":
    main()
