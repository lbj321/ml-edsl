"""FileCheck-based IR tests for tensor operations

Tests structural IR properties that runtime tests cannot catch.
See tests/CLAUDE.md for guidelines on what belongs here.
"""

import pytest
from mlir_edsl import ml_function, Tensor, i32


class TestDialectBoundaryIR:
    """Test that tensors use tensor dialect exclusively"""

    def test_tensor_no_memref_ops(self, check_ir):
        """Test tensors use tensor dialect, not memref"""
        @ml_function
        def tensor_only() -> i32:
            t = Tensor[i32, 3]([1, 2, 3])
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


class TestShapeRepresentationIR:
    """Test that multi-dimensional tensors preserve shape in IR (not flattened)"""

    def test_2d_tensor_shape(self, check_ir):
        """Test 2D tensor emits tensor<2x3xi32>, not tensor<6xi32>"""
        @ml_function
        def tensor_2d_shape() -> i32:
            t = Tensor[i32, 2, 3]([[1, 2, 3], [4, 5, 6]])
            return t[0, 0]

        tensor_2d_shape()

        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<2x3xi32>
        // CHECK-NOT: tensor<6xi32>
        """)

    def test_3d_tensor_shape(self, check_ir):
        """Test 3D tensor emits tensor<2x2x2xi32>, not tensor<8xi32>"""
        @ml_function
        def tensor_3d_shape() -> i32:
            t = Tensor[i32, 2, 2, 2]([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
            return t[0, 0, 0]

        tensor_3d_shape()

        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<2x2x2xi32>
        // CHECK-NOT: tensor<8xi32>
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
