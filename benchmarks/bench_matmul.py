"""Matmul, bias add, and relu benchmark: EDSL (O3) vs NumPy across matrix sizes.

Usage:
    python3 benchmarks/bench_matmul.py

Each size is compiled once, warmed up, then timed with enough repeats
to get stable measurements even for small matrices.

"""

import timeit

import numpy as np

from mlir_edsl import ml_function, Tensor, f32, matmul, relu
from mlir_edsl.backend import get_backend

SIZES = [2, 4, 8, 16, 32, 64, 128, 256, 512]
WARMUP = 5


def repeats_for(N: int) -> int:
    """Scale repeat count down for large N to keep benchmark runtime reasonable."""
    if N <= 8:
        return 10_000
    if N <= 32:
        return 1_000
    if N <= 128:
        return 200
    if N <= 256:
        return 50
    if N <= 512:
        return 10
    return 5


def make_edsl_matmul(N: int):
    """Compile a fixed-size NxN matmul function. Returns the callable."""
    backend = get_backend()
    backend.clear_module()
    backend.set_optimization_level(3)

    @ml_function
    def matmul_fn(A: Tensor[f32, N, N], B: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
        return matmul(A, B)

    A = np.ones((N, N), dtype=np.float32)
    B = np.ones((N, N), dtype=np.float32)
    matmul_fn(A, B)

    return matmul_fn


def make_edsl_bias_add(N: int):
    """Compile a fixed-size NxN bias add function. Returns the callable."""
    backend = get_backend()
    backend.clear_module()
    backend.set_optimization_level(3)

    @ml_function
    def bias_fn(X: Tensor[f32, N, N], b: Tensor[f32, N]) -> Tensor[f32, N, N]:
        return X + b

    X = np.ones((N, N), dtype=np.float32)
    b = np.ones(N, dtype=np.float32)
    bias_fn(X, b)

    return bias_fn


def make_edsl_relu(N: int):
    """Compile a fixed-size NxN relu function. Returns the callable."""
    backend = get_backend()
    backend.clear_module()
    backend.set_optimization_level(3)

    @ml_function
    def relu_fn(X: Tensor[f32, N, N]) -> Tensor[f32, N, N]:
        return relu(X)

    X = np.ones((N, N), dtype=np.float32)
    relu_fn(X)

    return relu_fn


def print_section(title: str, rows: list):
    print(f"\n=== {title} ===")
    print(f"{'Size':>6}  {'EDSL (µs)':>12}  {'NumPy (µs)':>12}  {'Ratio':>8}")
    print("-" * 46)
    for size_label, edsl_t, numpy_t in rows:
        ratio = edsl_t / numpy_t
        print(f"{size_label:>6}  {edsl_t * 1e6:>12.2f}  {numpy_t * 1e6:>12.2f}  {ratio:>7.2f}x")


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

        np_matmul_t[N] = timeit.timeit(lambda: np.matmul(A, B), number=n) / n
        np_bias_t[N]   = timeit.timeit(lambda: np.add(A, b),    number=n) / n
        np_relu_t[N]   = timeit.timeit(lambda: np.maximum(A, 0.0), number=n) / n

    # Phase 2: EDSL benchmarks (triggers libomp RTLD_GLOBAL on first call).
    matmul_rows = []
    bias_rows   = []
    relu_rows   = []

    for N in SIZES:
        A, B, b = inputs[N]
        n = repeats_for(N)
        label = f"{N:>4}x{N:<2}"

        edsl_fn = make_edsl_matmul(N)
        for _ in range(WARMUP):
            edsl_fn(A, B)
        edsl_t = timeit.timeit(lambda: edsl_fn(A, B), number=n) / n
        matmul_rows.append((label, edsl_t, np_matmul_t[N]))

        edsl_fn = make_edsl_bias_add(N)
        for _ in range(WARMUP):
            edsl_fn(A, b)
        edsl_t = timeit.timeit(lambda: edsl_fn(A, b), number=n) / n
        bias_rows.append((label, edsl_t, np_bias_t[N]))

        edsl_fn = make_edsl_relu(N)
        for _ in range(WARMUP):
            edsl_fn(A)
        edsl_t = timeit.timeit(lambda: edsl_fn(A), number=n) / n
        relu_rows.append((label, edsl_t, np_relu_t[N]))

    print_section("Matmul: X @ W", matmul_rows)
    print_section("Bias Add: X + b", bias_rows)
    print_section("ReLU: relu(X)", relu_rows)


if __name__ == "__main__":
    main()
