"""GPU matmul benchmark — larger sizes to stress-test tiling and block execution.

Skipped automatically when CUDA is unavailable or MLIR_EDSL_CUDA=OFF.
Run with -s to see timing output.

Sizes tested: 256, 512, 1024 — all require LinalgGPUMatmulTilingPass to fire.
CPU comparison deferred until multicore support is added.
"""

import ctypes
import time
import pytest
import numpy as np

from mlir_edsl import ml_function, Tensor, f32


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        ctypes.CDLL("libcuda.so.1")
        return True
    except OSError:
        return False


pytestmark = [
    pytest.mark.skipif(not _cuda_available(), reason="CUDA not available"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WARMUP = 3
_ITERS  = 10


def _bench(fn, *args) -> float:
    """Return mean wall-clock seconds per call over _ITERS iterations."""
    for _ in range(_WARMUP):
        fn(*args)
    t0 = time.perf_counter()
    for _ in range(_ITERS):
        fn(*args)
    return (time.perf_counter() - t0) / _ITERS


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_backend(backend):
    backend.set_target("gpu")
    yield backend
    backend.set_target("cpu")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

class TestGPUMatmulBenchmark:

    def test_matmul_256x256(self, gpu_backend):
        @ml_function(target="gpu")
        def matmul(A: Tensor[f32, 256, 256],
                   B: Tensor[f32, 256, 256]) -> Tensor[f32, 256, 256]:
            return A @ B

        A = np.random.rand(256, 256).astype(np.float32)
        B = np.random.rand(256, 256).astype(np.float32)

        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-3, atol=1e-3)

        ms = _bench(matmul, A, B) * 1000
        print(f"\n  256x256  gpu={ms:.2f}ms")

    def test_matmul_512x512(self, gpu_backend):
        @ml_function(target="gpu")
        def matmul(A: Tensor[f32, 512, 512],
                   B: Tensor[f32, 512, 512]) -> Tensor[f32, 512, 512]:
            return A @ B

        A = np.random.rand(512, 512).astype(np.float32)
        B = np.random.rand(512, 512).astype(np.float32)

        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-3, atol=1e-3)

        ms = _bench(matmul, A, B) * 1000
        print(f"\n  512x512  gpu={ms:.2f}ms")

    def test_matmul_1024x1024(self, gpu_backend):
        @ml_function(target="gpu")
        def matmul(A: Tensor[f32, 1024, 1024],
                   B: Tensor[f32, 1024, 1024]) -> Tensor[f32, 1024, 1024]:
            return A @ B

        A = np.random.rand(1024, 1024).astype(np.float32)
        B = np.random.rand(1024, 1024).astype(np.float32)

        result = matmul(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-3, atol=1e-3)

        ms = _bench(matmul, A, B) * 1000
        print(f"\n  1024x1024  gpu={ms:.2f}ms")
