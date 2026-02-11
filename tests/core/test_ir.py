"""FileCheck-based IR tests for core operations

Validates MLIR IR structure for scalar ops, control flow, recursion,
and type casting. Catches IR regressions that runtime tests miss.
"""

import pytest
from mlir_edsl import ml_function, i32, f32, cast
from mlir_edsl import add, sub, mul, eq, lt, gt, If, call


# ==================== SCALAR OPERATIONS ====================

class TestScalarIR:
    """Test IR patterns for scalar arithmetic"""

    def test_integer_add(self, check_ir):
        """Test integer addition emits arith.addi"""
        @ml_function
        def int_add(x: int, y: int) -> int:
            return add(x, y)

        int_add(1, 2)

        check_ir("""
        // CHECK: func.func @int_add(%arg0: i32, %arg1: i32) -> i32
        // CHECK: arith.addi %arg0, %arg1 : i32
        // CHECK: return
        """)

    def test_float_add(self, check_ir):
        """Test float addition emits arith.addf"""
        @ml_function
        def float_add(x: float, y: float) -> float:
            return add(x, y)

        float_add(1.0, 2.0)

        check_ir("""
        // CHECK: func.func @float_add(%arg0: f32, %arg1: f32) -> f32
        // CHECK: arith.addf %arg0, %arg1 : f32
        // CHECK: return
        """)

    def test_cast_int_to_float(self, check_ir):
        """Test cast emits arith.sitofp"""
        @ml_function
        def cast_itof(x: int) -> float:
            return cast(x, f32)

        cast_itof(5)

        check_ir("""
        // CHECK: func.func @cast_itof(%arg0: i32) -> f32
        // CHECK: arith.sitofp %arg0 : i32 to f32
        // CHECK: return
        """)


# ==================== CONTROL FLOW ====================

class TestControlFlowIR:
    """Test IR patterns for conditionals"""

    def test_if_else_structure(self, check_ir):
        """Test if/else emits arith.cmpi + scf.if with yields"""
        @ml_function
        def max_val(x: int, y: int) -> int:
            return If(gt(x, y), x, y)

        max_val(3, 5)

        check_ir("""
        // CHECK: func.func @max_val(%arg0: i32, %arg1: i32) -> i32
        // CHECK: arith.cmpi sgt, %arg0, %arg1 : i32
        // CHECK: scf.if
        // CHECK:   scf.yield %arg0
        // CHECK: } else {
        // CHECK:   scf.yield %arg1
        // CHECK: return
        """)

    def test_let_binding_no_duplicate_comparison(self, check_ir):
        """Test that let-binding prevents duplicate IR for reused values

        Without value caching, clamp would emit the lt comparison
        and its scf.if twice. With let-binding, the first If result
        is reused via SSA value.
        """
        @ml_function
        def clamp(x: int, lo: int, hi: int) -> int:
            clamped = If(lt(x, lo), lo, x)
            return If(gt(clamped, hi), hi, clamped)

        clamp(5, 0, 10)

        check_ir("""
        // CHECK: func.func @clamp(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
        // CHECK: arith.cmpi slt
        // CHECK: scf.if
        // CHECK: arith.cmpi sgt
        // CHECK: scf.if
        // CHECK-NOT: arith.cmpi slt
        // CHECK: return
        """)


# ==================== RECURSION ====================

class TestRecursionIR:
    """Test IR patterns for recursive functions"""

    def test_recursive_call_emits_func_call(self, check_ir):
        """Test recursive function emits func.call to itself"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, call("factorial", [sub(n, 1)], i32)))

        factorial(5)

        check_ir("""
        // CHECK: func.func @factorial(%arg0: i32) -> i32
        // CHECK: scf.if
        // CHECK: func.call @factorial
        // CHECK: return
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
