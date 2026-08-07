"""Regression tests for large tensor operation shapes.

Previously, LinalgGenericTilingPass tiled only the innermost dimension to 8,
leaving outer dims untiled (e.g. memref<1024x8xf32>). The vectorizer then
produced vector<1024x8xf32> (32KB), causing stack overflow in OMP worker
threads. The fix tiles all dimensions to 8 and also covers linalg.fill ops
(not just linalg.generic).
"""

import numpy as np
import pytest

from mlir_edsl import ml_function, Tensor, f32, matmul, relu


class TestLargeShapes:
    """Regression tests for large tensor sizes that crash via untiled linalg.fill."""

    def test_matmul_1024(self, backend):
        """1024x1024 matmul should compile and produce correct results."""
        @ml_function
        def matmul_fn(A: Tensor[f32, 1024, 1024], B: Tensor[f32, 1024, 1024]) -> Tensor[f32, 1024, 1024]:
            return matmul(A, B)

        A = np.ones((1024, 1024), dtype=np.float32)
        result = matmul_fn(A, A)
        np.testing.assert_allclose(result, np.full((1024, 1024), 1024.0), rtol=1e-4)

    def test_bias_1024(self, backend):
        """1024x1024 bias add should not crash during JIT compilation."""
        @ml_function
        def bias_fn(X: Tensor[f32, 1024, 1024], b: Tensor[f32, 1024]) -> Tensor[f32, 1024, 1024]:
            return X + b

        X = np.ones((1024, 1024), dtype=np.float32)
        b = np.ones(1024, dtype=np.float32)
        result = bias_fn(X, b)
        np.testing.assert_allclose(result, np.full((1024, 1024), 2.0), rtol=1e-4)

    def test_relu_1024(self, backend):
        """1024x1024 relu should not crash during JIT compilation."""
        @ml_function
        def relu_fn(X: Tensor[f32, 1024, 1024]) -> Tensor[f32, 1024, 1024]:
            return relu(X)

        X = np.ones((1024, 1024), dtype=np.float32)
        result = relu_fn(X)
        np.testing.assert_allclose(result, X, rtol=1e-4)
