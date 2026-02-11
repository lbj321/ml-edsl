"""FileCheck-based IR tests for tensor operations

Validates MLIR IR structure for tensor.from_elements, tensor.extract,
tensor.insert, and multi-dimensional tensors.
"""

import pytest
from mlir_edsl import ml_function, Tensor, i32, f32


# ==================== TENSOR BASICS ====================

class TestTensorBasicIR:
    """Test IR patterns for basic tensor operations"""

    def test_tensor_literal_uses_from_elements(self, check_ir):
        """Test tensor literal emits tensor.from_elements"""
        @ml_function
        def tensor_access() -> i32:
            t = Tensor[4, i32]([10, 20, 30, 40])
            return t[2]

        tensor_access()

        check_ir("""
        // CHECK: func.func @tensor_access() -> i32
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<4xi32>
        // CHECK: tensor.extract
        // CHECK-SAME: tensor<4xi32>
        // CHECK: return
        """)

    def test_float_tensor_uses_f32(self, check_ir):
        """Test float tensor emits tensor<3xf32> type"""
        @ml_function
        def tensor_float() -> f32:
            t = Tensor[3, f32]([1.0, 2.0, 3.0])
            return t[1]

        tensor_float()

        check_ir("""
        // CHECK: func.func @tensor_float() -> f32
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<3xf32>
        // CHECK: tensor.extract
        // CHECK-SAME: tensor<3xf32>
        // CHECK: return
        """)

    def test_tensor_no_memref_ops(self, check_ir):
        """Test tensors use tensor dialect, not memref"""
        @ml_function
        def tensor_only() -> i32:
            t = Tensor[3, i32]([1, 2, 3])
            return t[0]

        tensor_only()

        check_ir("""
        // CHECK: func.func @tensor_only
        // CHECK: tensor.from_elements
        // CHECK: tensor.extract
        // CHECK-NOT: memref.alloca
        // CHECK-NOT: memref.load
        // CHECK: return
        """)


# ==================== TENSOR INSERT ====================

class TestTensorInsertIR:
    """Test IR patterns for tensor.insert"""

    def test_tensor_insert_emits_insert(self, check_ir):
        """Test .at[].set() emits tensor.insert"""
        @ml_function
        def tensor_insert() -> i32:
            t = Tensor[4, i32]([10, 20, 30, 40])
            t = t.at[1].set(99)
            return t[1]

        tensor_insert()

        check_ir("""
        // CHECK: func.func @tensor_insert() -> i32
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<4xi32>
        // CHECK: arith.constant 99 : i32
        // CHECK: tensor.insert
        // CHECK-SAME: tensor<4xi32>
        // CHECK: tensor.extract
        // CHECK: return
        """)


# ==================== MULTI-DIMENSIONAL ====================

class TestTensorMultiDimIR:
    """Test IR patterns for 2D tensors"""

    def test_2d_tensor_uses_2d_type(self, check_ir):
        """Test 2D tensor emits tensor<2x3xi32> type"""
        @ml_function
        def tensor_2d() -> i32:
            t = Tensor[2, 3, i32]([[1, 2, 3], [4, 5, 6]])
            return t[1, 2]

        tensor_2d()

        check_ir("""
        // CHECK: func.func @tensor_2d() -> i32
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<2x3xi32>
        // CHECK: tensor.extract
        // CHECK-SAME: tensor<2x3xi32>
        // CHECK: return
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
