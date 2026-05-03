"""Tests for recursive function support in MLIR EDSL

This test suite validates recursive function capabilities:
- Basic recursive functions (factorial, fibonacci, countdown)
- Tail recursion patterns with accumulators
- Symbol resolution for self-referencing functions
- MLIR/LLVM code generation for recursive calls

Recursion works by calling the decorated MLFunction directly inside the function
body. In symbolic context, MLFunction.__call__ returns CallOp(self.signature.name, ...)
which uses the correct registered name (including the unique _func_id suffix).
"""

import pytest
from mlir_edsl import ml_function, add, sub, mul, eq, le, If
from mlir_edsl import i32


# ==================== BASIC RECURSION ====================

class TestBasicRecursion:
    """Test basic recursive function patterns"""

    def test_simple_recursive_factorial(self, backend):
        """Test basic recursive factorial: fact(n) = n == 0 ? 1 : n * fact(n-1)"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0),
                      1,
                      mul(n, factorial(sub(n, 1))))

        result = factorial(5)
        assert result == 120

    def test_fibonacci_recursion(self, backend):
        """Test fibonacci: fib(n) = n <= 1 ? n : fib(n-1) + fib(n-2)"""
        @ml_function
        def fibonacci(n: int) -> int:
            return If(le(n, 1),
                      n,
                      add(fibonacci(sub(n, 1)),
                          fibonacci(sub(n, 2))))

        result = fibonacci(7)
        assert result == 13

    def test_countdown_recursion(self, backend):
        """Test simple countdown: countdown(n) = n <= 0 ? 0 : countdown(n-1)"""
        @ml_function
        def countdown(n: int) -> int:
            return If(le(n, 0),
                      0,
                      countdown(sub(n, 1)))

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
                      tail_sum(sub(n, 1), add(acc, n)))

        result = tail_sum(10, 0)
        assert result == 55  # 1+2+3+...+10

    def test_tail_recursive_factorial(self, backend):
        """Test tail recursive factorial with accumulator"""
        @ml_function
        def tail_factorial(n: int, acc: int) -> int:
            return If(eq(n, 0),
                      acc,
                      tail_factorial(sub(n, 1), mul(acc, n)))

        result = tail_factorial(5, 1)
        assert result == 120


# ==================== SYMBOL RESOLUTION ====================

class TestRecursionSymbolResolution:
    """Test symbol table handling for recursive function definitions"""

    def test_recursive_function_symbol_resolution(self, backend):
        """Test that recursive functions can reference themselves during definition"""
        @ml_function
        def self_test(x: int) -> int:
            return If(eq(x, 0),
                      x,
                      add(x, self_test(sub(x, 1))))

        result = self_test(3)
        assert result == 6  # 3 + 2 + 1 + 0

    def test_nested_recursive_calls(self, backend):
        """Test multiple recursive calls in a single expression"""
        @ml_function
        def nested_fib(n: int) -> int:
            return If(le(n, 1),
                      n,
                      add(nested_fib(sub(n, 1)),
                          nested_fib(sub(n, 2))))

        result = nested_fib(6)
        assert result == 8  # Fibonacci(6) = 8


# ==================== MLIR/LLVM OUTPUT VALIDATION ====================

class TestRecursionCodeGeneration:
    """Test MLIR and LLVM IR generation for recursive functions"""

    def test_recursive_factorial_compiles(self, backend):
        """Test that recursive factorial compiles without errors"""
        @ml_function
        def fact(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, fact(sub(n, 1))))

        result = fact(4)
        assert result == 24


# ==================== EDGE CASES ====================

class TestRecursionEdgeCases:
    """Test edge cases and boundary conditions for recursion"""

    def test_recursion_base_case_immediate(self, backend):
        """Test recursion where base case is hit immediately"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, factorial(sub(n, 1))))

        result = factorial(0)
        assert result == 1

    def test_recursion_single_level(self, backend):
        """Test recursion with single recursive call"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, factorial(sub(n, 1))))

        result = factorial(1)
        assert result == 1

    def test_deep_recursion_compilation(self, backend):
        """Test that deeper recursion compiles correctly"""
        @ml_function
        def factorial(n: int) -> int:
            return If(eq(n, 0), 1, mul(n, factorial(sub(n, 1))))

        result = factorial(10)
        assert result == 3628800


if __name__ == "__main__":
    pytest.main([__file__])
