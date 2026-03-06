"""Test linalg.matmul execution end-to-end (Phase 8.2)

Validates that matmul(A, B) compiles through the full pipeline:
  Python AST → Protobuf → C++ LinalgBuilder → linalg.matmul IR
  → convert-linalg-to-loops → scf.for → LLVM IR → JIT execution

2D arrays are passed as np.ndarray with shape (M, N) and returned as np.ndarray.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Array, f32, i32, matmul


# ==================== BASIC MATMUL ====================

class TestMatmulExecution:
    """Test linalg.matmul execution with various inputs"""

    def test_matmul_identity(self, backend):
        """A @ I == A for identity matrix"""
        @ml_function
        def matmul_fn(A: Array[f32, 2, 2], B: Array[f32, 2, 2]) -> Array[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        I = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = matmul_fn(A, I)

        np.testing.assert_allclose(result, A, rtol=1e-4)

    def test_matmul_known_result_2x2(self, backend):
        """[[1,2],[3,4]] @ [[5,6],[7,8]] == [[19,22],[43,50]]"""
        @ml_function
        def matmul_2x2(A: Array[f32, 2, 2], B: Array[f32, 2, 2]) -> Array[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        result = matmul_2x2(A, B)

        np.testing.assert_allclose(result, [[19.0, 22.0], [43.0, 50.0]], rtol=1e-3)

    def test_matmul_zeros(self, backend):
        """A @ 0 == 0"""
        @ml_function
        def matmul_zeros(A: Array[f32, 2, 2], B: Array[f32, 2, 2]) -> Array[f32, 2, 2]:
            return matmul(A, B)

        A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        B = np.zeros((2, 2), dtype=np.float32)
        result = matmul_zeros(A, B)

        np.testing.assert_allclose(result, np.zeros((2, 2)), atol=1e-5)

    def test_matmul_integer(self, backend):
        """Integer matmul: I @ [[5,6],[7,8]] == [[5,6],[7,8]]"""
        @ml_function
        def matmul_int(A: Array[i32, 2, 2], B: Array[i32, 2, 2]) -> Array[i32, 2, 2]:
            return matmul(A, B)

        I = np.array([[1, 0], [0, 1]], dtype=np.int32)
        B = np.array([[5, 6], [7, 8]], dtype=np.int32)
        result = matmul_int(I, B)

        np.testing.assert_array_equal(result, [[5, 6], [7, 8]])


# ==================== TYPE VALIDATION ====================

class TestMatmulTypeValidation:
    """Test that type mismatches are caught at Python level"""

    def test_matmul_requires_2d_arrays(self):
        """matmul requires 2D arrays"""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, f32 as f32_type

        a_1d = ArrayLiteral([1.0, 2.0], ArrayType(2, f32_type))
        b_2d = ArrayLiteral([[1.0, 2.0], [3.0, 4.0]], ArrayType((2, 2), f32_type))

        with pytest.raises(TypeError, match="2D array"):
            matmul(a_1d, b_2d)

    def test_matmul_requires_matching_inner_dims(self):
        """matmul requires matching inner dimensions"""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, f32 as f32_type

        # lhs is 2x3, rhs is 2x2 — K_lhs=3 != K_rhs=2
        a = ArrayLiteral([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                         ArrayType((2, 3), f32_type))
        b = ArrayLiteral([[1.0, 2.0], [3.0, 4.0]],
                         ArrayType((2, 2), f32_type))

        with pytest.raises(TypeError, match="inner dimensions must match"):
            matmul(a, b)

    def test_matmul_requires_matching_element_types(self):
        """matmul requires matching element types"""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, i32 as i32_type, f32 as f32_type

        a = ArrayLiteral([[1.0, 2.0], [3.0, 4.0]], ArrayType((2, 2), f32_type))
        b = ArrayLiteral([[1, 2], [3, 4]], ArrayType((2, 2), i32_type))

        with pytest.raises(TypeError, match="element types must match"):
            matmul(a, b)
