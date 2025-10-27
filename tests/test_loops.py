"""Tests for for loop and while loop implementations using high-level API

This test suite uses @ml_function decorator with For() and While() helpers.
Tests focus on JIT execution results rather than backend implementation details.
"""

import pytest
from mlir_edsl import ml_function, For, While
from mlir_edsl.backend import HAS_CPP_BACKEND

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


def test_for_loop_basic():
    """Test basic for loop with addition"""

    @ml_function
    def sum_range():
        # for(i = 0; i < 5; i += 1) with accumulator starting at 10
        # Computes: 10 + 0 + 1 + 2 + 3 + 4 = 20
        return For(start=0, end=5, init=10, operation="add")

    result = sum_range()
    assert result == 20, f"Expected 20, got {result}"


def test_for_loop_multiplication():
    """Test for loop with multiplication (factorial-like)"""

    @ml_function
    def factorial_like():
        # for(i = 1; i < 5; i++) result *= i
        # Computes: 1 * 1 * 2 * 3 * 4 = 24
        return For(start=1, end=5, init=1, operation="mul")

    result = factorial_like()
    assert result == 24, f"Expected 24, got {result}"


def test_for_loop_float():
    """Test for loop with float operations"""

    @ml_function
    def float_sum():
        # Sum: 0.5 + 0.0 + 1.0 + 2.0 = 3.5
        return For(start=0, end=3, init=0.5, operation="add")

    result = float_sum()
    expected = 3.5
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"


def test_for_loop_subtraction():
    """Test for loop with subtraction"""

    @ml_function
    def subtract_range():
        # Subtract: 10 - 0 - 1 - 2 = 7
        return For(start=0, end=3, init=10, operation="sub")

    result = subtract_range()
    assert result == 7, f"Expected 7, got {result}"


def test_for_loop_division():
    """Test for loop with division"""

    @ml_function
    def divide_range():
        # Divide: 24 / 2 / 3 = 4
        return For(start=2, end=4, init=24, operation="div")

    result = divide_range()
    assert result == 4, f"Expected 4, got {result}"


# ========== WHILE LOOP TESTS ==========

def test_while_loop_basic():
    """Test basic while loop counting up"""

    @ml_function
    def count_up():
        # while(current < 5) { current = current + 1 } starting from 0
        # Computes: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (stop)
        return While(init=0, target=5, operation="add", predicate="slt")

    result = count_up()
    assert result == 5, f"Expected 5, got {result}"


def test_while_loop_multiplication():
    """Test while loop with multiplication (doubling)"""

    @ml_function
    def double_until():
        # while(current < 8) { current = current * 2 } starting from 1
        # Computes: 1 -> 2 -> 4 -> 8 (stop)
        return While(init=1, target=8, operation="mul", predicate="slt")

    result = double_until()
    assert result == 8, f"Expected 8, got {result}"


def test_while_loop_float():
    """Test while loop with float operations"""

    @ml_function
    def float_count():
        # while(current < 2.5) { current = current + 1 } starting from 0.5
        # Computes: 0.5 -> 1.5 -> 2.5 (stop)
        return While(init=0.5, target=2.5, operation="add", predicate="olt")

    result = float_count()
    expected = 2.5
    assert abs(result - expected) < 0.001, f"Expected {expected}, got {result}"


def test_while_loop_subtraction():
    """Test while loop with subtraction (counting down)"""

    @ml_function
    def count_down():
        # while(current > 0) { current = current - 1 } starting from 5
        # Computes: 5 -> 4 -> 3 -> 2 -> 1 -> 0 (stop)
        return While(init=5, target=0, operation="sub", predicate="sgt")

    result = count_down()
    assert result == 0, f"Expected 0, got {result}"


def test_while_loop_division():
    """Test while loop with division (halving)"""

    @ml_function
    def halve_until():
        # while(current > 1) { current = current / 2 } starting from 16
        # Computes: 16 -> 8 -> 4 -> 2 -> 1 (stop)
        return While(init=16, target=1, operation="div", predicate="sgt")

    result = halve_until()
    assert result == 1, f"Expected 1, got {result}"


def test_while_loop_equality_condition():
    """Test while loop with != condition"""

    @ml_function
    def until_equal():
        # while(current != 3) { current = current + 1 } starting from 0
        # Computes: 0 -> 1 -> 2 -> 3 (stop)
        return While(init=0, target=3, operation="add", predicate="ne")

    result = until_equal()
    assert result == 3, f"Expected 3, got {result}"


if __name__ == "__main__":
    pytest.main([__file__])