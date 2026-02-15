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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
