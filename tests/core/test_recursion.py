"""Tests for recursive function support in MLIR EDSL

This test suite validates recursive function capabilities:
- Basic recursive functions (factorial, fibonacci, countdown)
- Tail recursion patterns with accumulators
- Symbol resolution for self-referencing functions
- MLIR/LLVM code generation for recursive calls

Recursion uses the `call()` function with explicit function name references.
"""

import pytest
from mlir_edsl import ml_function, add, sub, mul, eq, le, If, call
from mlir_edsl import i32, i1


# ==================== BASIC RECURSION ====================

class TestBasicRecursion:
    """Test basic recursive function patterns"""

    def test_simple_recursive_factorial(self, backend):
        """Test basic recursive factorial: fact(n) = n == 0 ? 1 : n * fact(n-1)"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0),
                      1,
                      mul(n, call("factorial", [sub(n, 1)], i32)))

        result = factorial(5)
        assert result == 120

    def test_fibonacci_recursion(self, backend):
        """Test fibonacci: fib(n) = n <= 1 ? n : fib(n-1) + fib(n-2)"""
        @ml_function
        def fibonacci(n: int) -> int:
            return If(le(n, 1),
                      n,
                      add(call("fibonacci", [sub(n, 1)], i32),
                          call("fibonacci", [sub(n, 2)], i32)))

        result = fibonacci(7)
        assert result == 13

    def test_countdown_recursion(self, backend):
        """Test simple countdown: countdown(n) = n <= 0 ? 0 : countdown(n-1)"""
        @ml_function
        def countdown(n: int) -> int:
            return If(le(n, 0),
                      0,
                      call("countdown", [sub(n, 1)], i32))

        result = countdown(10)
        assert result == 0


# ==================== TAIL RECURSION ====================

class TestTailRecursion:
    """Test tail recursive patterns with accumulators"""

    def test_tail_recursive_sum(self, backend):
        """Test tail recursive sum: sum(n, acc) = n == 0 ? acc : sum(n-1, acc+n)"""
        @ml_function
        def tail_sum(n: int, acc: int) -> int:
            return If(eq(n, 0),
                      acc,
                      call("tail_sum", [sub(n, 1), add(acc, n)], i32))

        result = tail_sum(10, 0)
        assert result == 55  # 1+2+3+...+10

    def test_tail_recursive_factorial(self, backend):
        """Test tail recursive factorial with accumulator"""
        @ml_function
        def tail_factorial(n: int, acc: int) -> int:
            return If(eq(n, 0),
                      acc,
                      call("tail_factorial", [sub(n, 1), mul(acc, n)], i32))

        result = tail_factorial(5, 1)
        assert result == 120


# ==================== SYMBOL RESOLUTION ====================

class TestRecursionSymbolResolution:
    """Test symbol table handling for recursive function definitions"""

    def test_recursive_function_symbol_resolution(self, backend):
        """Test that recursive functions can reference themselves during definition"""
        # This tests that a function is available in the symbol table
        # while its body is being defined, allowing self-reference
        @ml_function
        def self_test(x: int) -> int:
            # Self-reference during definition
            # Using If to avoid infinite recursion at runtime
            return If(eq(x, 0),
                      x,
                      add(x, call("self_test", [sub(x, 1)], i32)))

        # Compilation should succeed
        result = self_test(3)
        assert result == 6  # 3 + 2 + 1 + 0

    def test_nested_recursive_calls(self, backend):
        """Test multiple recursive calls in a single expression"""
        @ml_function
        def nested_fib(n: int) -> int:
            # Multiple recursive calls in one expression
            return If(le(n, 1),
                      n,
                      add(call("nested_fib", [sub(n, 1)], i32),
                          call("nested_fib", [sub(n, 2)], i32)))

        result = nested_fib(6)
        assert result == 8  # Fibonacci(6) = 8


# ==================== MUTUAL RECURSION ====================

class TestMutualRecursion:
    """Test mutually recursive functions calling each other"""

    def test_mutual_recursion_even_odd(self):
        """Test mutually recursive is_even/is_odd functions"""
        pytest.skip("Mutual recursion requires multiple function definitions in single compilation")

        # Future implementation pattern:
        # @ml_function
        # def is_even(n: int) -> bool:
        #     return If(eq(n, 0), True, call("is_odd", [sub(n, 1)], i1))
        #
        # @ml_function
        # def is_odd(n: int) -> bool:
        #     return If(eq(n, 0), False, call("is_even", [sub(n, 1)], i1))


# ==================== MLIR/LLVM OUTPUT VALIDATION ====================

class TestRecursionCodeGeneration:
    """Test MLIR and LLVM IR generation for recursive functions"""

    def test_recursive_factorial_compiles(self, backend):
        """Test that recursive factorial compiles without errors"""
        @ml_function
        def fact(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, call("fact", [sub(n, 1)], i32)))

        # Just verify compilation succeeds
        result = fact(4)
        assert result == 24


# ==================== EDGE CASES ====================

class TestRecursionEdgeCases:
    """Test edge cases and boundary conditions for recursion"""

    def test_recursion_base_case_immediate(self, backend):
        """Test recursion where base case is hit immediately"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, call("factorial", [sub(n, 1)], i32)))

        # Base case: n = 0
        result = factorial(0)
        assert result == 1

    def test_recursion_single_level(self, backend):
        """Test recursion with single recursive call"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, call("factorial", [sub(n, 1)], i32)))

        # One level of recursion: n = 1 -> factorial(0)
        result = factorial(1)
        assert result == 1

    def test_deep_recursion_compilation(self, backend):
        """Test that deeper recursion compiles correctly"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, call("factorial", [sub(n, 1)], i32)))

        # Test with moderate depth (10 levels)
        result = factorial(10)
        assert result == 3628800


if __name__ == "__main__":
    pytest.main([__file__])