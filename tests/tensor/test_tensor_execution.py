"""Test tensor execution end-to-end (Phase 7 - Tensor Dialect Steps 1-3)

This test validates that tensor operations compile through the full pipeline:
  Python AST → Protobuf → C++ MLIRBuilder → tensor dialect IR
  → bufferization → memref → LLVM IR → JIT execution

Tests cover:
- tensor.from_elements + tensor.extract with i32 and f32
- Multi-element tensors with various access patterns
- Computed indices
- 2D tensor creation and extraction
- Tensor element reuse in arithmetic
"""

import pytest
from mlir_edsl import ml_function, Tensor, i32, f32


# ==================== BASIC TENSOR EXTRACT ====================

class TestTensorExtractExecution:
    """Test basic tensor.extract execution"""

    def test_tensor_extract_f32(self, backend):
        """Test extracting f32 element from tensor"""
        @ml_function
        def tensor_extract_f32() -> f32:
            t = Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])
            return t[2]

        result = tensor_extract_f32()
        assert abs(result - 3.0) < 0.001

    def test_tensor_extract_i32(self, backend):
        """Test extracting i32 element from tensor"""
        @ml_function
        def tensor_extract_i32() -> i32:
            t = Tensor[i32, 4]([10, 20, 30, 40])
            return t[2]

        result = tensor_extract_i32()
        assert result == 30

    def test_tensor_extract_first_element(self, backend):
        """Test extracting first element (index 0)"""
        @ml_function
        def first_element() -> i32:
            t = Tensor[i32, 3]([100, 200, 300])
            return t[0]

        result = first_element()
        assert result == 100

    def test_tensor_extract_last_element(self, backend):
        """Test extracting last element"""
        @ml_function
        def last_element() -> i32:
            t = Tensor[i32, 5]([10, 20, 30, 40, 50])
            return t[4]

        result = last_element()
        assert result == 50


# ==================== TENSOR WITH ARITHMETIC ====================

class TestTensorArithmeticExecution:
    """Test tensor extracts used in arithmetic expressions"""

    def test_tensor_extract_add(self, backend):
        """Test adding two extracted tensor elements"""
        @ml_function
        def tensor_add() -> i32:
            t = Tensor[i32, 4]([10, 20, 30, 40])
            return t[0] + t[3]

        result = tensor_add()
        assert result == 50  # 10 + 40

    def test_tensor_extract_mul(self, backend):
        """Test multiplying two extracted tensor elements"""
        @ml_function
        def tensor_mul() -> i32:
            t = Tensor[i32, 3]([2, 3, 5])
            return t[0] * t[2]

        result = tensor_mul()
        assert result == 10  # 2 * 5

    def test_tensor_extract_float_arithmetic(self, backend):
        """Test float arithmetic with tensor extracts"""
        @ml_function
        def tensor_float_arith() -> f32:
            t = Tensor[f32, 3]([1.5, 2.5, 3.5])
            return t[0] + t[1]

        result = tensor_float_arith()
        assert abs(result - 4.0) < 0.001  # 1.5 + 2.5


# ==================== COMPUTED INDEX ====================

class TestTensorComputedIndexExecution:
    """Test tensor access with computed indices"""

    def test_tensor_computed_index(self, backend):
        """Test tensor access with index computed from arithmetic"""
        @ml_function
        def computed_index() -> i32:
            t = Tensor[i32, 5]([10, 20, 30, 40, 50])
            idx = 1 + 2
            return t[idx]

        result = computed_index()
        assert result == 40  # t[3]


# ==================== MULTIPLE TENSORS ====================

class TestMultipleTensorsExecution:
    """Test operations with multiple tensors"""

    def test_two_tensors(self, backend):
        """Test extracting from two different tensors"""
        @ml_function
        def two_tensors() -> i32:
            a = Tensor[i32, 3]([10, 20, 30])
            b = Tensor[i32, 3]([1, 2, 3])
            return a[1] + b[2]

        result = two_tensors()
        assert result == 23  # 20 + 3

    def test_two_float_tensors(self, backend):
        """Test extracting from two f32 tensors"""
        @ml_function
        def two_float_tensors() -> f32:
            a = Tensor[f32, 2]([1.0, 2.0])
            b = Tensor[f32, 2]([10.0, 20.0])
            return a[0] + b[1]

        result = two_float_tensors()
        assert abs(result - 21.0) < 0.001  # 1.0 + 20.0


# ==================== 2D TENSOR ====================

class TestTensor2DExecution:
    """Test 2D tensor creation and extraction"""

    def test_2d_tensor_extract(self, backend):
        """Test extracting element from 2D tensor"""
        @ml_function
        def tensor_2d() -> i32:
            t = Tensor[i32, 2, 3]([[1, 2, 3],
                                    [4, 5, 6]])
            return t[1, 2]

        result = tensor_2d()
        assert result == 6

    def test_2d_tensor_extract_origin(self, backend):
        """Test extracting element at [0,0] from 2D tensor"""
        @ml_function
        def tensor_2d_origin() -> i32:
            t = Tensor[i32, 2, 3]([[10, 20, 30],
                                    [40, 50, 60]])
            return t[0, 0]

        result = tensor_2d_origin()
        assert result == 10


# ==================== SINGLE ELEMENT TENSOR ====================

class TestTensorEdgeCasesExecution:
    """Test edge cases for tensor execution"""

    def test_single_element_tensor(self, backend):
        """Test tensor with single element"""
        @ml_function
        def single_elem() -> i32:
            t = Tensor[i32, 1]([42])
            return t[0]

        result = single_elem()
        assert result == 42

    def test_tensor_same_element_twice(self, backend):
        """Test extracting same element twice in arithmetic"""
        @ml_function
        def same_twice() -> i32:
            t = Tensor[i32, 3]([5, 10, 15])
            return t[1] + t[1]

        result = same_twice()
        assert result == 20  # 10 + 10



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
