"""Test linalg.matmul execution end-to-end (Phase 8.2)

Validates that matmul(A, B) compiles through the full pipeline:
  Python AST → Protobuf → C++ LinalgBuilder → linalg.matmul IR
  → convert-linalg-to-loops → scf.for → LLVM IR → JIT execution

2D arrays are passed as np.ndarray with shape (M, N) and returned as np.ndarray.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, matmul


# ==================== BASIC MATMUL ====================

class TestMatmulExecution:
    """Test linalg.matmul execution with various inputs"""

    def test_matmul_identity(self, backend):
        """A @ I == A for identity matrix"""
        @ml_function
        def matmul_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = matmul_fn(A, I)

        np.testing.assert_allclose(result, A, rtol=1e-4)

    def test_matmul_known_result_2x2(self, backend):
        """[[1,2],[3,4]] @ [[5,6],[7,8]] == [[19,22],[43,50]]"""
        @ml_function
        def matmul_2x2(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = matmul_2x2(A, B)

        np.testing.assert_allclose(result, [[19.0, 22.0], [43.0, 50.0]], rtol=1e-3)

    def test_matmul_8x8(self, backend):
        """8x8 matmul: ones @ ones == 8*ones"""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        A = np.ones((8, 8), dtype=np.float32)
        B = np.ones((8, 8), dtype=np.float32)
        result = mm_fn(A, B)

        np.testing.assert_allclose(result, np.full((8, 8), 8.0), atol=1e-5)

    def test_matmul_zeros(self, backend):
        """A @ 0 == 0"""
        @ml_function
        def matmul_zeros(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.zeros((2, 2), dtype=np.float32)
        result = matmul_zeros(A, B)

        np.testing.assert_allclose(result, np.zeros((2, 2)), atol=1e-5)

    def test_matmul_integer(self, backend):
        """Integer matmul: I @ [[5,6],[7,8]] == [[5,6],[7,8]]"""
        @ml_function
        def matmul_int(A: Tensor[i32, 2, 2], B: Tensor[i32, 2, 2]) -> Tensor[i32, 2, 2]:
            return matmul(A, B)

        I = np.array([[1, 0], [0, 1]], dtype=np.int32)
        B = np.array([[5, 6], [7, 8]], dtype=np.int32)
        result = matmul_int(I, B)

        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])


# ==================== CHAINED MATMUL ====================

class TestMatmulChained:
    """Test matmul used as a sub-expression (no outParam — tensor.empty path)"""

    def test_matmul_chained(self, backend):
        """(A @ B) @ C exercises the tensor.empty fallback for the inner matmul"""
        @ml_function
        def chained(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2], C: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(matmul(A, B), C)

        A = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # identity
        B = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        C = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)  # identity

        result = chained(A, B, C)
        np.testing.assert_allclose(result, B, rtol=1e-4)


# ==================== TYPE VALIDATION ====================

class TestMatmulTypeValidation:
    """Test that type mismatches are caught at Python level"""

    def test_matmul_requires_2d_arrays(self):
        """matmul requires 2D tensors"""
        from mlir_edsl.ast.nodes.functions import Parameter
        from mlir_edsl.types import TensorType, f32 as f32_type

        a_1d = Parameter("a", TensorType(2, f32_type))
        b_2d = Parameter("b", TensorType((2, 2), f32_type))

        with pytest.raises(TypeError, match="2D tensor"):
            matmul(a_1d, b_2d)

    def test_matmul_requires_matching_inner_dims(self):
        """matmul requires matching inner dimensions"""
        from mlir_edsl.ast.nodes.functions import Parameter
        from mlir_edsl.types import TensorType, f32 as f32_type

        # lhs is 2x3, rhs is 2x2 — K_lhs=3 != K_rhs=2
        a = Parameter("a", TensorType((2, 3), f32_type))
        b = Parameter("b", TensorType((2, 2), f32_type))

        with pytest.raises(TypeError, match="inner dimensions must match"):
            matmul(a, b)

    def test_matmul_requires_matching_element_types(self):
        """matmul requires matching element types"""
        from mlir_edsl.ast.nodes.functions import Parameter
        from mlir_edsl.types import TensorType, i32 as i32_type, f32 as f32_type

        a = Parameter("a", TensorType((2, 2), f32_type))
        b = Parameter("b", TensorType((2, 2), i32_type))

        with pytest.raises(TypeError, match="element types must match"):
            matmul(a, b)
