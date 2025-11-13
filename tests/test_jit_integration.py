"""Test high-level JIT integration with @ml_function decorator

This test suite validates:
- JIT execution through @ml_function decorator
- Operator overloading (+, -, *, /)
- Integer and float operations
- Complex expressions with multiple operations
- Reverse operator support (__radd__, __rmul__)
- Mixed type operations with explicit casting
"""

import pytest
from mlir_edsl import ml_function, cast
from mlir_edsl import f32
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend")
class TestJITIntegration(MLIRTestBase):
    """Test JIT execution with @ml_function decorator"""

    def test_ml_function_jit_execution(self):
        """Test JIT execution through @ml_function decorator with operator overloading"""
        @ml_function
        def add_example(x: int, y: int) -> int:
            return x + y

        # Test direct function call (JIT execution)
        result = add_example(4, 6)
        assert result == 10

    def test_ml_function_float_jit(self):
        """Test JIT execution with float operations"""
        @ml_function
        def float_mul(x: float, y: float) -> float:
            return x * y

        result = float_mul(2.5, 4.0)
        assert abs(result - 10.0) < 1e-6

    def test_ml_function_subtraction_jit(self):
        """Test JIT execution with subtraction using operator overloading"""
        @ml_function
        def sub_example(x: int, y: int) -> int:
            return x - y

        result = sub_example(20, 8)
        assert result == 12

    def test_ml_function_division_jit(self):
        """Test JIT execution with division using operator overloading"""
        @ml_function
        def div_example(x: float, y: float) -> float:
            return x / y

        result = div_example(15.0, 3.0)
        assert abs(result - 5.0) < 1e-6

    def test_ml_function_complex_expression(self):
        """Test JIT execution with complex expression using operator overloading"""
        @ml_function
        def complex_example(a: int, b: int, c: int, d: int) -> int:
            return (a + b) - (c * d)

        result = complex_example(10, 5, 2, 3)
        assert result == 9  # (10 + 5) - (2 * 3) = 15 - 6 = 9

    def test_ml_function_mixed_types(self):
        """Test JIT execution with mixed int/float types using explicit cast"""
        @ml_function
        def mixed_types(x: int, y: float) -> float:
            return cast(x, f32) + y  # Explicit cast required for strict typing

        result = mixed_types(5, 2.5)
        assert abs(result - 7.5) < 1e-6

    def test_ml_function_reverse_operations(self):
        """Test reverse operator overloading (literal on left side)"""
        @ml_function
        def reverse_add(x: int) -> int:
            return 10 + x  # Tests __radd__

        result = reverse_add(5)
        assert result == 15

        @ml_function
        def reverse_mul(x: int) -> int:
            return 3 * x  # Tests __rmul__

        result2 = reverse_mul(4)
        assert result2 == 12