"""Test linalg.dot execution end-to-end (Phase 8.2)

Validates that dot(a, b) compiles through the full pipeline:
  Python AST → Protobuf → C++ LinalgBuilder → linalg.dot IR
  → convert-linalg-to-loops → scf.for → LLVM IR → JIT execution
"""

import pytest
from mlir_edsl import ml_function, Array, f32, i32, dot


# ==================== BASIC DOT PRODUCT ====================

class TestDotExecution:
    """Test linalg.dot execution with various inputs"""

    def test_dot_all_ones(self, backend):
        """dot([1,1,1,1], [1,1,1,1]) == 4.0"""
        @ml_function
        def dot_ones(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return dot(a, b)

        import ctypes
        result = dot_ones([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0])
        assert abs(result - 4.0) < 1e-5

    def test_dot_known_result(self, backend):
        """dot([1,2,3,4], [1,1,1,1]) == 10.0"""
        @ml_function
        def dot_known(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return dot(a, b)

        result = dot_known([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
        assert abs(result - 10.0) < 1e-5

    def test_dot_sum_of_squares(self, backend):
        """dot([1,2,3], [1,2,3]) == 14.0"""
        @ml_function
        def dot_squares(a: Array[f32, 3], b: Array[f32, 3]) -> f32:
            return dot(a, b)

        result = dot_squares([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(result - 14.0) < 1e-5

    def test_dot_integer(self, backend):
        """dot([1,2,3,4], [4,3,2,1]) == 20 (integer)"""
        @ml_function
        def dot_int(a: Array[i32, 4], b: Array[i32, 4]) -> i32:
            return dot(a, b)

        result = dot_int([1, 2, 3, 4], [4, 3, 2, 1])
        assert result == 20  # 1*4 + 2*3 + 3*2 + 4*1 = 4+6+6+4 = 20

    def test_dot_with_zeros(self, backend):
        """dot of any vector with zero vector is 0"""
        @ml_function
        def dot_zero(a: Array[f32, 3], b: Array[f32, 3]) -> f32:
            return dot(a, b)

        result = dot_zero([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        assert abs(result - 0.0) < 1e-5

    def test_dot_single_element(self, backend):
        """dot of single-element arrays"""
        @ml_function
        def dot_single(a: Array[f32, 1], b: Array[f32, 1]) -> f32:
            return dot(a, b)

        result = dot_single([3.0], [7.0])
        assert abs(result - 21.0) < 1e-5


# ==================== TYPE VALIDATION ====================

class TestDotTypeValidation:
    """Test that type mismatches are caught at Python level"""

    def test_dot_requires_1d_array(self):
        """dot requires 1D arrays, not scalars"""
        from mlir_edsl import Value
        from mlir_edsl.ast.nodes.scalars import Constant
        from mlir_edsl.types import i32 as i32_type
        scalar = Constant(5, i32_type)

        with pytest.raises(TypeError, match="1D array"):
            dot(scalar, scalar)

    def test_dot_requires_matching_element_types(self):
        """dot requires matching element types"""
        from mlir_edsl.ast.nodes.arrays import ArrayLiteral
        from mlir_edsl.types import ArrayType, i32 as i32_type, f32 as f32_type

        a = ArrayLiteral([1.0, 2.0], ArrayType(2, f32_type))
        b = ArrayLiteral([1, 2], ArrayType(2, i32_type))

        with pytest.raises(TypeError, match="element types must match"):
            dot(a, b)
