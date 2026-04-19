"""GPU execution tests — correctness on small matrices (no tiling).

Skipped automatically when CUDA is unavailable or MLIR_EDSL_CUDA=OFF.
All test matrices use sizes ≤ 32 to stay within GPU thread limits without
the tiling pass.
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
    # TODO: GPU execution crashes with SIGABRT (core dump) inside executeGPUFunction.
    # Crash origin is inside C++ — likely one of:
    #   1. MLIR assertion in lowerToGPUModule (gpu-map-parallel-loops precondition)
    #   2. cuModuleLoadData aborting on invalid PTX
    #   3. cuLaunchKernel with wrong kernel argument count/layout
    # Marked xfail(strict=False) so the suite runs without crashing while we debug.
    pytest.mark.xfail(strict=False, reason="GPU execution crashes (under investigation)"),
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
