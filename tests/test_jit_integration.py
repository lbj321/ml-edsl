"""Test high-level JIT integration with @ml_function decorator

This test suite uses only the user-facing API with operator overloading.
Tests should remain stable even when backend implementation changes.
"""
import pytest
from mlir_edsl import ml_function

def test_ml_function_jit_execution():
    """Test JIT execution through @ml_function decorator with operator overloading"""

    @ml_function
    def add_example(x, y):
        return x + y

    # Test direct function call (JIT execution)
    result = add_example(4, 6)
    assert result == 10

def test_ml_function_float_jit():
    """Test JIT execution with float operations"""

    @ml_function
    def float_mul(x, y):
        return x * y

    result = float_mul(2.5, 4.0)
    assert abs(result - 10.0) < 1e-6

def test_ml_function_subtraction_jit():
    """Test JIT execution with subtraction using operator overloading"""

    @ml_function
    def sub_example(x, y):
        return x - y

    result = sub_example(20, 8)
    assert result == 12

def test_ml_function_division_jit():
    """Test JIT execution with division using operator overloading"""

    @ml_function
    def div_example(x, y):
        return x / y

    result = div_example(15.0, 3.0)
    assert abs(result - 5.0) < 1e-6

def test_ml_function_complex_expression():
    """Test JIT execution with complex expression using operator overloading"""

    @ml_function
    def complex_example(a, b, c, d):
        return (a + b) - (c * d)

    result = complex_example(10, 5, 2, 3)
    assert result == 9  # (10 + 5) - (2 * 3) = 15 - 6 = 9

def test_ml_function_mixed_types():
    """Test JIT execution with mixed int/float types (type promotion)"""

    @ml_function
    def mixed_types(x, y):
        return x + y  # int + float should promote to float

    result = mixed_types(5, 2.5)
    assert abs(result - 7.5) < 1e-6

def test_ml_function_reverse_operations():
    """Test reverse operator overloading (literal on left side)"""

    @ml_function
    def reverse_add(x):
        return 10 + x  # Tests __radd__

    result = reverse_add(5)
    assert result == 15

    @ml_function
    def reverse_mul(x):
        return 3 * x  # Tests __rmul__

    result2 = reverse_mul(4)
    assert result2 == 12