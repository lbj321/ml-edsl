"""FileCheck-based IR tests for core operations

Tests structural IR properties that runtime tests cannot catch.
See tests/CLAUDE.md for guidelines on what belongs here.
"""

import pytest
from mlir_edsl import ml_function, i32
from mlir_edsl import If


class TestControlFlowIR:
    """Test IR structural properties for conditionals"""

    def test_let_binding_no_duplicate_comparison(self, check_ir):
        """Test that let-binding prevents duplicate IR for reused values

        Without value caching, clamp would emit the lt comparison
        and its scf.if twice. With let-binding, the first If result
        is reused via SSA value.
        """
        @ml_function
        def clamp(x: int, lo: int, hi: int) -> int:
            clamped = If(x < lo, lo, x)
            return If(clamped > hi, hi, clamped)

        clamp(5, 0, 10)

        check_ir("""
        // CHECK: func.func @clamp_{{[0-9]+}}(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
        // CHECK: arith.cmpi slt
        // CHECK: scf.if
        // CHECK: arith.cmpi sgt
        // CHECK: scf.if
        // CHECK-NOT: arith.cmpi slt
        // CHECK: return
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
