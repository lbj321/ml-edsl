"""Tests for conditional operations using high-level API

This test suite uses @ml_function decorator with operator overloading and If() helper.
Tests focus on JIT execution results rather than backend implementation details.
"""

import pytest
from mlir_edsl import ml_function, If
from mlir_edsl.backend import HAS_CPP_BACKEND

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


# ==================== BASIC CONDITIONAL TESTS ====================

def test_if_greater_than():
    """Test If with greater than comparison"""

    @ml_function
    def max_value(x, y):
        return If(x > y, then_value=x, else_value=y)

    assert max_value(10, 5) == 10
    assert max_value(3, 7) == 7
    assert max_value(5, 5) == 5


def test_if_less_than():
    """Test If with less than comparison"""

    @ml_function
    def min_value(x, y):
        return If(x < y, then_value=x, else_value=y)

    assert min_value(10, 5) == 5
    assert min_value(3, 7) == 3
    assert min_value(5, 5) == 5


def test_if_equality():
    """Test If with equality comparison"""

    @ml_function
    def check_equal(x, y):
        return If(x == y, then_value=1, else_value=0)

    assert check_equal(5, 5) == 1
    assert check_equal(5, 3) == 0


def test_if_not_equal():
    """Test If with not-equal comparison"""

    @ml_function
    def check_not_equal(x, y):
        return If(x != y, then_value=1, else_value=0)

    assert check_not_equal(5, 3) == 1
    assert check_not_equal(5, 5) == 0


# ==================== CONDITIONAL WITH EXPRESSIONS ====================

def test_if_with_arithmetic_branches():
    """Test If with arithmetic in branches"""

    @ml_function
    def conditional_calc(x, y):
        return If(x > y, then_value=x * 2, else_value=y * 2)

    assert conditional_calc(10, 5) == 20  # 10 > 5, so 10 * 2
    assert conditional_calc(3, 7) == 14   # 3 < 7, so 7 * 2


def test_if_with_complex_condition():
    """Test If with complex expression in condition"""

    @ml_function
    def complex_condition(a, b, c):
        # Condition: (a + b) > c
        return If((a + b) > c, then_value=a + b, else_value=c)

    assert complex_condition(10, 5, 12) == 15  # (10+5)=15 > 12, return 15
    assert complex_condition(3, 2, 10) == 10   # (3+2)=5 < 10, return 10


def test_if_with_complex_branches():
    """Test If with complex expressions in both branches"""

    @ml_function
    def nested_arithmetic(x, y):
        # if ((10 + 5) > 12) return (2 * 3) else return (20 - 5)
        condition = (x + 5) > 12
        then_branch = 2 * 3
        else_branch = 20 - 5
        return If(condition, then_value=then_branch, else_value=else_branch)

    assert nested_arithmetic(10, 0) == 6   # (10+5)=15 > 12, return 2*3=6
    assert nested_arithmetic(5, 0) == 15   # (5+5)=10 < 12, return 20-5=15


# ==================== FLOAT CONDITIONALS ====================

def test_if_float_comparison():
    """Test If with float comparisons"""

    @ml_function
    def float_max(x, y):
        return If(x > y, then_value=x, else_value=y)

    result = float_max(3.5, 2.1)
    assert abs(result - 3.5) < 0.001

    result = float_max(1.5, 2.8)
    assert abs(result - 2.8) < 0.001


def test_if_float_operations():
    """Test If with float arithmetic in branches"""

    @ml_function
    def float_conditional(x, y):
        return If(x > y, then_value=x + 1.5, else_value=y + 2.5)

    result = float_conditional(10.0, 5.0)
    assert abs(result - 11.5) < 0.001  # 10 > 5, so 10 + 1.5

    result = float_conditional(3.0, 7.0)
    assert abs(result - 9.5) < 0.001   # 3 < 7, so 7 + 2.5


def test_if_mixed_type_comparison():
    """Test If with mixed int/float (type promotion)"""

    @ml_function
    def mixed_conditional(x, y):
        # x is int, y is float - should promote to float comparison
        return If(x > y, then_value=x, else_value=y)

    result = mixed_conditional(10, 5.5)
    # 10.0 > 5.5, return 10 (promoted to float)
    assert abs(result - 10.0) < 0.001


# ==================== COMPARISON OPERATORS ====================

def test_all_comparison_operators():
    """Test all comparison operators via overloading"""

    @ml_function
    def test_gt(x, y):
        return If(x > y, then_value=1, else_value=0)

    @ml_function
    def test_lt(x, y):
        return If(x < y, then_value=1, else_value=0)

    @ml_function
    def test_ge(x, y):
        return If(x >= y, then_value=1, else_value=0)

    @ml_function
    def test_le(x, y):
        return If(x <= y, then_value=1, else_value=0)

    # Greater than
    assert test_gt(10, 5) == 1
    assert test_gt(5, 10) == 0

    # Less than
    assert test_lt(5, 10) == 1
    assert test_lt(10, 5) == 0

    # Greater or equal
    assert test_ge(10, 5) == 1
    assert test_ge(5, 5) == 1
    assert test_ge(3, 5) == 0

    # Less or equal
    assert test_le(5, 10) == 1
    assert test_le(5, 5) == 1
    assert test_le(10, 5) == 0


# ==================== PARAMETERIZED CONDITIONALS ====================

def test_if_with_parameters():
    """Test If using function parameters in all parts"""

    @ml_function
    def param_conditional(a, b, x, y):
        # Use parameters in condition and branches
        return If(a > b, then_value=x, else_value=y)

    assert param_conditional(10, 5, 100, 200) == 100
    assert param_conditional(3, 7, 100, 200) == 200


def test_if_chained_logic():
    """Test multiple conditional-like logic"""

    @ml_function
    def clamp(value, min_val, max_val):
        # Clamp value between min and max
        # First check if value < min_val
        adjusted = If(value < min_val, then_value=min_val, else_value=value)
        # Then check if adjusted > max_val
        return If(adjusted > max_val, then_value=max_val, else_value=adjusted)

    assert clamp(5, 0, 10) == 5    # Within range
    assert clamp(-5, 0, 10) == 0   # Below min
    assert clamp(15, 0, 10) == 10  # Above max


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
