"""FileCheck-based IR tests for memref (array) operations

Validates MLIR IR structure for array allocation, load/store,
element-wise loops, multi-dimensional arrays, and broadcasting.
"""

import pytest
from mlir_edsl import ml_function, Array, i32, f32


# ==================== ARRAY BASICS ====================

class TestArrayBasicIR:
    """Test IR patterns for basic array operations"""

    def test_array_literal_uses_alloca(self, check_ir):
        """Test array literal emits memref.alloca + stores"""
        @ml_function
        def array_access() -> i32:
            arr = Array[4, i32]([10, 20, 30, 40])
            return arr[2]

        array_access()

        check_ir("""
        // CHECK: func.func @array_access() -> i32
        // CHECK: memref.alloca() : memref<4xi32>
        // CHECK: memref.store
        // CHECK: memref.load
        // CHECK: return
        """)

    def test_array_store_emits_store(self, check_ir):
        """Test .at[].set() emits memref.store for the update"""
        @ml_function
        def array_store() -> i32:
            arr = Array[3, i32]([1, 2, 3])
            arr = arr.at[1].set(99)
            return arr[1]

        array_store()

        check_ir("""
        // CHECK: func.func @array_store() -> i32
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: arith.constant 99 : i32
        // CHECK: memref.store
        // CHECK: memref.load
        // CHECK: return
        """)

    def test_float_array_uses_f32_memref(self, check_ir):
        """Test float array emits memref with f32 element type"""
        @ml_function
        def float_array() -> f32:
            arr = Array[3, f32]([1.5, 2.5, 3.5])
            return arr[1]

        float_array()

        check_ir("""
        // CHECK: func.func @float_array() -> f32
        // CHECK: memref.alloca() : memref<3xf32>
        // CHECK: memref.store
        // CHECK: memref.load
        // CHECK-SAME: memref<3xf32>
        // CHECK: return
        """)


# ==================== ELEMENT-WISE OPERATIONS ====================

class TestElementwiseIR:
    """Test IR patterns for element-wise array operations"""

    def test_array_add_uses_scf_for(self, check_ir):
        """Test element-wise add emits scf.for with addi"""
        @ml_function
        def array_add() -> i32:
            arr1 = Array[3, i32]([1, 2, 3])
            arr2 = Array[3, i32]([10, 20, 30])
            result = arr1 + arr2
            return result[1]

        array_add()

        check_ir("""
        // CHECK: func.func @array_add() -> i32
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: scf.for
        // CHECK:   memref.load
        // CHECK:   memref.load
        // CHECK:   arith.addi
        // CHECK:   memref.store
        // CHECK: }
        // CHECK: memref.load
        // CHECK: return
        """)

    def test_scalar_broadcast_uses_scf_for(self, check_ir):
        """Test scalar broadcast emits scf.for with muli"""
        @ml_function
        def scalar_broadcast() -> i32:
            arr = Array[3, i32]([10, 20, 30])
            result = arr * 2
            return result[1]

        scalar_broadcast()

        check_ir("""
        // CHECK: func.func @scalar_broadcast() -> i32
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: arith.constant 2 : i32
        // CHECK: memref.alloca() : memref<3xi32>
        // CHECK: scf.for
        // CHECK:   memref.load
        // CHECK:   arith.muli
        // CHECK:   memref.store
        // CHECK: }
        // CHECK: return
        """)


# ==================== MULTI-DIMENSIONAL ====================

class TestMultiDimIR:
    """Test IR patterns for 2D and 3D arrays"""

    def test_2d_array_uses_2d_memref(self, check_ir):
        """Test 2D array emits memref<2x3xi32> type"""
        @ml_function
        def array_2d() -> i32:
            arr = Array[2, 3, i32]([[1, 2, 3], [4, 5, 6]])
            return arr[1, 2]

        array_2d()

        check_ir("""
        // CHECK: func.func @array_2d() -> i32
        // CHECK: memref.alloca() : memref<2x3xi32>
        // CHECK: memref.store
        // CHECK: memref.load
        // CHECK-SAME: memref<2x3xi32>
        // CHECK: return
        """)

    def test_2d_array_no_tensor_ops(self, check_ir):
        """Test 2D array uses memref dialect, not tensor"""
        @ml_function
        def array_2d_only() -> i32:
            arr = Array[2, 2, i32]([[1, 2], [3, 4]])
            return arr[0, 0]

        array_2d_only()

        check_ir("""
        // CHECK: func.func @array_2d_only
        // CHECK: memref.alloca
        // CHECK-NOT: tensor.from_elements
        // CHECK-NOT: tensor.extract
        // CHECK: return
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
