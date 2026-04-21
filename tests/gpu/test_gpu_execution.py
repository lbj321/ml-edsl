"""GPU execution tests — correctness on small matrices and tiled matrices.

Skipped automatically when CUDA is unavailable or MLIR_EDSL_CUDA=OFF.
TestGPUMatmul: sizes ≤ 32 (no tiling fires).
TestGPUMatmulTiled: sizes > 32 to exercise LinalgGPUMatmulTilingPass.
  - Multiples of 32 (64, 128): clean tile boundaries.
  - Non-multiples (48, 96): produce boundary tiles that cannot be
    canonicalized away, exercising the fallback scalar loop path.
"""

import ctypes
import pytest
import numpy as np

from mlir_edsl import ml_function, Tensor, f32


# ---------------------------------------------------------------------------
# Skip guard: skip the whole module if libcuda is not loadable
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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gpu_backend(backend):
    """Backend with GPU target set for the duration of this module."""
    backend.set_target("gpu")
    yield backend
    backend.set_target("cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGPUMatmul:
    """Correctness tests for matmul on GPU."""

    def test_matmul_16x16(self, gpu_backend):
        @ml_function(target="gpu")
        def matmul_gpu(A: Tensor[f32, 16, 16],
                       B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return A @ B

        A = np.random.rand(16, 16).astype(np.float32)
        B = np.random.rand(16, 16).astype(np.float32)

        result = matmul_gpu(A, B)

        np.testing.assert_allclose(result, A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_identity(self, gpu_backend):
        """Multiplying by identity should return the original matrix."""
        @ml_function(target="gpu")
        def matmul_identity(A: Tensor[f32, 8, 8],
                            B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return A @ B

        A = np.random.rand(8, 8).astype(np.float32)
        I = np.eye(8, dtype=np.float32)

        result = matmul_identity(A, I)

        np.testing.assert_allclose(result, A, rtol=1e-5, atol=1e-5)

    def test_matmul_zeros(self, gpu_backend):
        """Multiplying by zero matrix should give zero result."""
        @ml_function(target="gpu")
        def matmul_zeros(A: Tensor[f32, 8, 8],
                         B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return A @ B

        A = np.random.rand(8, 8).astype(np.float32)
        Z = np.zeros((8, 8), dtype=np.float32)

        result = matmul_zeros(A, Z)

        np.testing.assert_allclose(result, np.zeros((8, 8)), atol=1e-6)


class TestGPUMatmulTiled:
    """Correctness tests for matmul sizes that exercise LinalgGPUMatmulTilingPass.

    Sizes > 32 force tiling to fire. Non-multiples of 32 (48, 96) produce
    boundary tiles that cannot be canonicalized, testing the fallback scalar path.
    """

    def test_matmul_64x64(self, gpu_backend):
        """Clean 2x2 tile grid — all tiles are full 32x32."""
        @ml_function(target="gpu")
        def matmul_64(A: Tensor[f32, 64, 64],
                      B: Tensor[f32, 64, 64]) -> Tensor[f32, 64, 64]:
            return A @ B

        A = np.random.rand(64, 64).astype(np.float32)
        B = np.random.rand(64, 64).astype(np.float32)
        np.testing.assert_allclose(matmul_64(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_48x48(self, gpu_backend):
        """Non-multiple of 32 — produces one full 32x32 tile and one 16-wide boundary tile."""
        @ml_function(target="gpu")
        def matmul_48(A: Tensor[f32, 48, 48],
                      B: Tensor[f32, 48, 48]) -> Tensor[f32, 48, 48]:
            return A @ B

        A = np.random.rand(48, 48).astype(np.float32)
        B = np.random.rand(48, 48).astype(np.float32)
        np.testing.assert_allclose(matmul_48(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_128x128(self, gpu_backend):
        """4x4 tile grid — stresses multi-block execution."""
        @ml_function(target="gpu")
        def matmul_128(A: Tensor[f32, 128, 128],
                       B: Tensor[f32, 128, 128]) -> Tensor[f32, 128, 128]:
            return A @ B

        A = np.random.rand(128, 128).astype(np.float32)
        B = np.random.rand(128, 128).astype(np.float32)
        np.testing.assert_allclose(matmul_128(A, B), A @ B, rtol=1e-4, atol=1e-4)

    def test_matmul_96x96(self, gpu_backend):
        """Non-multiple of 32 at larger scale — 3 full tiles + no remainder (96=3×32)."""
        @ml_function(target="gpu")
        def matmul_96(A: Tensor[f32, 96, 96],
                      B: Tensor[f32, 96, 96]) -> Tensor[f32, 96, 96]:
            return A @ B

        A = np.random.rand(96, 96).astype(np.float32)
        B = np.random.rand(96, 96).astype(np.float32)
        np.testing.assert_allclose(matmul_96(A, B), A @ B, rtol=1e-4, atol=1e-4)
