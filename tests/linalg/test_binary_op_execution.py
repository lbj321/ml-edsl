"""Runtime execution tests for LinalgBinaryOp and @ operator

Validates that tensor element-wise ops and bias add compile and execute correctly:
  Python AST → Protobuf → LinalgBuilder → linalg.map / linalg.broadcast IR
  → lowering pipeline → LLVM JIT
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, matmul, relu, leaky_relu


# ==================== SAME-SHAPE ELEMENT-WISE ====================

class TestSameShapeElementWise:
    """Tensor op Tensor with identical shapes (NONE broadcast mode)."""

    def test_1d_add(self, backend):
        """[4] + [4] element-wise addition."""
        @ml_function
        def add_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a + b

        result = add_fn(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result, [11.0, 22.0, 33.0, 44.0], rtol=1e-5)

    def test_1d_mul(self, backend):
        """[4] * [4] element-wise multiply."""
        @ml_function
        def mul_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a * b

        result = mul_fn(
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result, [2.0, 6.0, 12.0, 20.0], rtol=1e-5)

    def test_1d_sub(self, backend):
        """[4] - [4] element-wise subtraction."""
        @ml_function
        def sub_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a - b

        result = sub_fn(
            np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        )
        np.testing.assert_allclose(result, [4.0, 4.0, 4.0, 4.0], rtol=1e-5)

    def test_2d_add(self, backend):
        """[2,3] + [2,3] element-wise addition."""
        @ml_function
        def add_2d(a: Tensor[f32, 2, 3], b: Tensor[f32, 2, 3]) -> Tensor[f32, 2, 3]:
            return a + b

        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        B = np.ones((2, 3), dtype=np.float32)
        result = add_2d(A, B)
        np.testing.assert_allclose(result, A + B, rtol=1e-5)


# ==================== SCALAR BROADCAST ====================

class TestScalarBroadcast:
    """Tensor op scalar and scalar op Tensor (SCALAR_* broadcast modes)."""

    def test_tensor_mul_scalar(self, backend):
        """Tensor[4] * scalar scales all elements."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a * 3.0

        result = scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(result, [3.0, 6.0, 9.0, 12.0], rtol=1e-5)

    def test_scalar_mul_tensor(self, backend):
        """scalar * Tensor[4] — reverse operand order."""
        @ml_function
        def scale_r(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return 2.0 * a

        result = scale_r(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(result, [2.0, 4.0, 6.0, 8.0], rtol=1e-5)

    def test_tensor_add_scalar(self, backend):
        """Tensor[4] + scalar adds a constant offset."""
        @ml_function
        def offset(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a + 1.0

        result = offset(np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32))
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0], rtol=1e-5)

    def test_2d_tensor_mul_scalar(self, backend):
        """Tensor[2,3] * scalar."""
        @ml_function
        def scale_2d(a: Tensor[f32, 2, 3]) -> Tensor[f32, 2, 3]:
            return a * 2.0

        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = scale_2d(A)
        np.testing.assert_allclose(result, A * 2.0, rtol=1e-5)


# ==================== BIAS ADD (TENSOR_BIAS_*) ====================

class TestBiasAdd:
    """Tensor[M,N] + Tensor[N]: 1D bias broadcast over rows."""

    def test_bias_add_2x4(self, backend):
        """[2,4] + [4]: each row gets the bias added."""
        @ml_function
        def bias_add(x: Tensor[f32, 2, 4], b: Tensor[f32, 4]) -> Tensor[f32, 2, 4]:
            return x + b

        X = np.array([[1.0, 2.0, 3.0, 4.0],
                      [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        bias = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = bias_add(X, bias)
        np.testing.assert_allclose(result, X + bias, rtol=1e-5)

    def test_bias_add_left_operand(self, backend):
        """[4] + [2,4]: bias on the left side."""
        @ml_function
        def bias_add_left(b: Tensor[f32, 4], x: Tensor[f32, 2, 4]) -> Tensor[f32, 2, 4]:
            return b + x

        bias = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        X = np.zeros((2, 4), dtype=np.float32)
        result = bias_add_left(bias, X)
        np.testing.assert_allclose(result, bias[np.newaxis, :] + X, rtol=1e-5)

    def test_bias_add_zeros_is_identity(self, backend):
        """Adding zero bias returns the original values."""
        @ml_function
        def add_zero_bias(x: Tensor[f32, 3, 3], b: Tensor[f32, 3]) -> Tensor[f32, 3, 3]:
            return x + b

        X = np.arange(9, dtype=np.float32).reshape(3, 3)
        bias = np.zeros(3, dtype=np.float32)
        result = add_zero_bias(X, bias)
        np.testing.assert_allclose(result, X, rtol=1e-5)

    def test_bias_add_large(self, backend):
        """[8,16] + [16]: larger matrix for coverage."""
        @ml_function
        def bias_add_large(x: Tensor[f32, 8, 16], b: Tensor[f32, 16]) -> Tensor[f32, 8, 16]:
            return x + b

        X = np.random.rand(8, 16).astype(np.float32)
        bias = np.random.rand(16).astype(np.float32)
        result = bias_add_large(X, bias)
        np.testing.assert_allclose(result, X + bias, rtol=1e-4)


# ==================== @ OPERATOR (MATMUL) ====================

class TestMatmulOperator:
    """@ operator dispatches to linalg.matmul."""

    def test_matmul_operator_2x2(self, backend):
        """A @ B produces the same result as matmul(A, B)."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return A @ B

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = mm(A, B)
        np.testing.assert_allclose(result, A @ B, rtol=1e-3)

    def test_matmul_operator_rectangular(self, backend):
        """A[2,3] @ B[3,4] → C[2,4]."""
        @ml_function
        def mm_rect(A: Tensor[f32, 2, 3], B: Tensor[f32, 3, 4]) -> Tensor[f32, 2, 4]:
            return A @ B

        A = np.ones((2, 3), dtype=np.float32)
        B = np.ones((3, 4), dtype=np.float32)
        result = mm_rect(A, B)
        np.testing.assert_allclose(result, np.full((2, 4), 3.0), rtol=1e-5)


# ==================== DENSE LAYER PATTERN ====================

class TestDenseLayerPattern:
    """Compose matmul + bias add + relu to validate dense layer patterns."""

    def test_dense_layer_no_activation(self, backend):
        """output = X @ W + b matches numpy."""
        @ml_function
        def dense(X: Tensor[f32, 2, 4],
                  W: Tensor[f32, 4, 3],
                  b: Tensor[f32, 3]) -> Tensor[f32, 2, 3]:
            return X @ W + b

        X = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        W = np.ones((4, 3), dtype=np.float32)
        b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = dense(X, W, b)
        np.testing.assert_allclose(result, X @ W + b, rtol=1e-4)

    def test_dense_layer_with_relu(self, backend):
        """relu(X @ W + b) matches numpy."""
        @ml_function
        def dense_relu(X: Tensor[f32, 2, 4],
                       W: Tensor[f32, 4, 3],
                       b: Tensor[f32, 3]) -> Tensor[f32, 2, 3]:
            return relu(X @ W + b)

        X = np.random.randn(2, 4).astype(np.float32)
        W = np.random.randn(4, 3).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)
        result = dense_relu(X, W, b)
        expected = np.maximum(X @ W + b, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_two_layer_net(self, backend):
        """Two stacked dense layers with relu: relu(relu(X @ W1 + b1) @ W2 + b2)."""
        @ml_function
        def two_layer(X:  Tensor[f32, 2, 4],
                      W1: Tensor[f32, 4, 8],
                      b1: Tensor[f32, 8],
                      W2: Tensor[f32, 8, 3],
                      b2: Tensor[f32, 3]) -> Tensor[f32, 2, 3]:
            h = relu(X @ W1 + b1)
            return relu(h @ W2 + b2)

        X  = np.random.randn(2, 4).astype(np.float32)
        W1 = np.random.randn(4, 8).astype(np.float32)
        b1 = np.random.randn(8).astype(np.float32)
        W2 = np.random.randn(8, 3).astype(np.float32)
        b2 = np.random.randn(3).astype(np.float32)
        result = two_layer(X, W1, b1, W2, b2)
        expected = np.maximum(np.maximum(X @ W1 + b1, 0.0) @ W2 + b2, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_three_layer_net(self, backend):
        """Three stacked dense layers — verifies arbitrary depth stacking."""
        @ml_function
        def three_layer(X:  Tensor[f32, 4, 8],
                        W1: Tensor[f32, 8, 16],
                        b1: Tensor[f32, 16],
                        W2: Tensor[f32, 16, 8],
                        b2: Tensor[f32, 8],
                        W3: Tensor[f32, 8, 4],
                        b3: Tensor[f32, 4]) -> Tensor[f32, 4, 4]:
            h1 = relu(X  @ W1 + b1)
            h2 = relu(h1 @ W2 + b2)
            return relu(h2 @ W3 + b3)

        X  = np.random.randn(4, 8).astype(np.float32)
        W1 = np.random.randn(8, 16).astype(np.float32)
        b1 = np.random.randn(16).astype(np.float32)
        W2 = np.random.randn(16, 8).astype(np.float32)
        b2 = np.random.randn(8).astype(np.float32)
        W3 = np.random.randn(8, 4).astype(np.float32)
        b3 = np.random.randn(4).astype(np.float32)
        result = three_layer(X, W1, b1, W2, b2, W3, b3)
        h1 = np.maximum(X @ W1 + b1, 0.0)
        h2 = np.maximum(h1 @ W2 + b2, 0.0)
        expected = np.maximum(h2 @ W3 + b3, 0.0)
        np.testing.assert_allclose(result, expected, rtol=1e-4)
