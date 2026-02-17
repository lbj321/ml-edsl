"""Test tensor.empty execution end-to-end

This test validates that tensor.empty compiles through the full pipeline:
  Python AST → Protobuf → C++ MLIRBuilder → tensor dialect IR
  → bufferization → memref → LLVM IR → JIT execution

Tests cover:
- tensor.empty + tensor.insert + tensor.extract with i32 and f32
- Filling all elements of an empty tensor
- 2D empty tensor creation and insertion
"""

import pytest
from mlir_edsl import ml_function, Tensor, i32, f32


# ==================== BASIC TENSOR EMPTY ====================

class TestTensorEmptyExecution:
    """Test tensor.empty execution end-to-end"""

    def test_tensor_empty_insert_extract(self, backend):
        """Test creating empty tensor, inserting, then extracting"""
        @ml_function
        def empty_insert_extract() -> i32:
            t = Tensor.empty(i32, 4)
            t = t.at[0].set(42)
            return t[0]

        result = empty_insert_extract()
        assert result == 42

    def test_tensor_empty_fill_all(self, backend):
        """Test filling all elements of an empty tensor"""
        @ml_function
        def empty_fill() -> i32:
            t = Tensor.empty(i32, 3)
            t = t.at[0].set(10)
            t = t.at[1].set(20)
            t = t.at[2].set(30)
            return t[0] + t[1] + t[2]

        result = empty_fill()
        assert result == 60

    def test_tensor_empty_f32(self, backend):
        """Test empty tensor with f32"""
        @ml_function
        def empty_f32() -> f32:
            t = Tensor.empty(f32, 2)
            t = t.at[0].set(1.5)
            t = t.at[1].set(2.5)
            return t[0] + t[1]

        result = empty_f32()
        assert abs(result - 4.0) < 0.001

    def test_tensor_empty_2d(self, backend):
        """Test 2D empty tensor"""
        @ml_function
        def empty_2d() -> i32:
            t = Tensor.empty(i32, 2, 3)
            t = t.at[1, 2].set(99)
            return t[1, 2]

        result = empty_2d()
        assert result == 99


# ==================== DYNAMIC TENSOR EMPTY ====================

class TestDynamicTensorEmptyExecution:
    """Test dynamic tensor.empty execution end-to-end"""

    def test_dynamic_empty_insert_extract(self, backend):
        """Test dynamic empty tensor with insert and extract"""
        @ml_function
        def dyn_empty(n: int) -> i32:
            t = Tensor.empty(i32, n)
            t = t.at[0].set(42)
            return t[0]

        result = dyn_empty(4)
        assert result == 42

    def test_dynamic_empty_fill_multiple(self, backend):
        """Test filling multiple elements in dynamic tensor"""
        @ml_function
        def dyn_fill(n: int) -> i32:
            t = Tensor.empty(i32, n)
            t = t.at[0].set(10)
            t = t.at[1].set(20)
            t = t.at[2].set(30)
            return t[0] + t[1] + t[2]

        result = dyn_fill(3)
        assert result == 60

    def test_dynamic_empty_f32(self, backend):
        """Test dynamic empty tensor with f32"""
        @ml_function
        def dyn_f32(n: int) -> f32:
            t = Tensor.empty(f32, n)
            t = t.at[0].set(1.5)
            t = t.at[1].set(2.5)
            return t[0] + t[1]

        result = dyn_f32(2)
        assert abs(result - 4.0) < 0.001

    def test_dynamic_empty_different_sizes(self, backend):
        """Test calling with different runtime sizes"""
        @ml_function
        def dyn_size(n: int) -> i32:
            t = Tensor.empty(i32, n)
            t = t.at[0].set(99)
            return t[0]

        assert dyn_size(1) == 99
        assert dyn_size(10) == 99
        assert dyn_size(100) == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
