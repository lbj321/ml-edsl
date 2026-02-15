"""FileCheck-based IR tests for memref (array) operations

Tests structural IR properties that runtime tests cannot catch.
See tests/CLAUDE.md for guidelines on what belongs here.
"""

import pytest
from mlir_edsl import ml_function, Array, i32


class TestDialectBoundaryIR:
    """Test that arrays use memref dialect exclusively"""

    def test_2d_array_no_tensor_ops(self, check_ir):
        """Test 2D array uses memref dialect, not tensor"""
        @ml_function
        def array_2d_only() -> i32:
            arr = Array[i32, 2, 2]([[1, 2], [3, 4]])
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
