"""FileCheck-based IR tests for tensor operations

Tests structural IR properties that runtime tests cannot catch.
See tests/CLAUDE.md for guidelines on what belongs here.
"""

import pytest
from mlir_edsl import ml_function, Tensor, i32, f32, cast


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


class TestBufferizationIR:
    """Verify the frontend emits correct tensor IR before any lowering passes.

    linalg-fuse-elementwise-ops folds trivial tensor accesses (e.g.
    tensor.extract(tensor.from_elements[const_idx]) → constant) before
    one-shot-bufferize runs. We therefore check the pre-lowering IR here.
    End-to-end correctness of bufferization is covered by execution tests.
    """

    def test_tensor_emits_from_elements(self, check_ir):
        """Frontend emits tensor.from_elements for literal tensor construction."""
        @ml_function
        def tensor_buf() -> i32:
            t = Tensor[i32, 4]([1, 2, 3, 4])
            return t[0]

        tensor_buf()

        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<4xi32>
        // CHECK: tensor.extract
        """)

    def test_tensor_literal_uses_tensor_dialect(self, check_ir):
        """Frontend uses tensor dialect (not memref) for tensor literals."""
        @ml_function
        def tensor_heap() -> i32:
            t = Tensor[i32, 4]([1, 2, 3, 4])
            return t[0]

        tensor_heap()

        check_ir("""
        // CHECK: tensor<4xi32>
        // CHECK-NOT: memref.alloc
        // CHECK-NOT: memref.alloca
        """)

    def test_2d_tensor_shape_not_flattened(self, check_ir):
        """Frontend emits tensor<2x3xi32>, not tensor<6xi32> (shape preserved)."""
        @ml_function
        def tensor_2d_buf() -> i32:
            t = Tensor[i32, 2, 3]([[1, 2, 3], [4, 5, 6]])
            return t[0, 0]

        tensor_2d_buf()

        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK-SAME: tensor<2x3xi32>
        // CHECK-NOT: tensor<6xi32>
        """)

    def test_tensor_extract_emitted(self, check_ir):
        """Frontend emits tensor.extract for element access."""
        @ml_function
        def tensor_dealloc() -> i32:
            t = Tensor[i32, 4]([1, 2, 3, 4])
            return t[0]

        tensor_dealloc()

        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK: tensor.extract
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


class TestTensorEmptyIR:
    """Test that tensor.empty emits correct IR"""

    def test_tensor_empty_op_in_ir(self, check_ir):
        """Test tensor.empty appears in IR"""
        @ml_function
        def empty_tensor() -> i32:
            t = Tensor.empty(i32, 4)
            t = t.at[0].set(1)
            return t[0]

        empty_tensor()

        check_ir("""
        // CHECK: func.func @empty_tensor
        // CHECK: tensor.empty
        // CHECK: tensor.insert
        // CHECK: tensor.extract
        """)

    def test_tensor_empty_no_from_elements(self, check_ir):
        """Test tensor.empty does not emit tensor.from_elements"""
        @ml_function
        def empty_no_from() -> i32:
            t = Tensor.empty(i32, 3)
            t = t.at[0].set(1)
            return t[0]

        empty_no_from()

        check_ir("""
        // CHECK: tensor.empty
        // CHECK-NOT: tensor.from_elements
        """)

    def test_tensor_empty_2d_shape(self, check_ir):
        """Test 2D tensor.empty preserves shape"""
        @ml_function
        def empty_2d_shape() -> i32:
            t = Tensor.empty(i32, 2, 3)
            t = t.at[0, 0].set(1)
            return t[0, 0]

        empty_2d_shape()

        check_ir("""
        // CHECK: tensor.empty() : tensor<2x3xi32>
        """)

    def test_tensor_empty_bufferizes(self, check_lowered_ir):
        """Test that tensor.empty bufferizes correctly"""
        @ml_function
        def empty_buf() -> i32:
            t = Tensor.empty(i32, 4)
            t = t.at[0].set(1)
            return t[0]

        empty_buf()

        check_lowered_ir("""
        // CHECK: memref.alloc
        // CHECK-NOT: tensor.empty
        """, after="one-shot-bufferize")


class TestDynamicTensorIR:
    """Test IR for dynamic tensor operations"""

    def test_dynamic_empty_has_dynamic_dim(self, check_ir):
        """Test tensor.empty with dynamic dim emits tensor<?x...> type"""
        @ml_function
        def dyn_empty(n: int) -> i32:
            t = Tensor.empty(i32, n)
            t = t.at[0].set(42)
            return t[0]

        dyn_empty(4)

        check_ir("""
        // CHECK: tensor.empty
        // CHECK-SAME: tensor<?xi32>
        """)

    def test_dynamic_empty_bufferizes(self, check_lowered_ir):
        """Test dynamic tensor.empty bufferizes to memref.alloc with dynamic dim"""
        @ml_function
        def dyn_buf(n: int) -> i32:
            t = Tensor.empty(i32, n)
            t = t.at[0].set(1)
            return t[0]

        dyn_buf(4)

        check_lowered_ir("""
        // CHECK: memref.alloc
        // CHECK-SAME: memref<?xi32>
        // CHECK-NOT: tensor.empty
        """, after="one-shot-bufferize")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
