"""Tests for conditional operations using high-level API

This test suite validates:
- If() conditional expressions with comparison operators
- Integer and float conditional operations
- Complex conditions and branch expressions
- All comparison operators (>, <, >=, <=, ==, !=)
- Parameterized conditionals
- Chained conditional logic
"""

import pytest
from mlir_edsl import ml_function, If, cast
from mlir_edsl import f32
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


# ==================== BASIC CONDITIONAL TESTS ====================

class TestBasicConditionals(MLIRTestBase):
    """Test basic conditional operations"""

    def test_if_greater_than(self):
        """Test If with greater than comparison"""
        @ml_function
        def max_value(x: int, y: int) -> int:
            return If(x > y, then_value=x, else_value=y)

        assert max_value(10, 5) == 10
        assert max_value(3, 7) == 7
        assert max_value(5, 5) == 5

    def test_if_less_than(self):
        """Test If with less than comparison"""
        @ml_function
        def min_value(x: int, y: int) -> int:
            return If(x < y, then_value=x, else_value=y)

        assert min_value(10, 5) == 5
        assert min_value(3, 7) == 3
        assert min_value(5, 5) == 5

    def test_if_equality(self):
        """Test If with equality comparison"""
        @ml_function
        def check_equal(x: int, y: int) -> int:
            return If(x == y, then_value=1, else_value=0)

        assert check_equal(5, 5) == 1
        assert check_equal(5, 3) == 0

    def test_if_not_equal(self):
        """Test If with not-equal comparison"""
        @ml_function
        def check_not_equal(x: int, y: int) -> int:
            return If(x != y, then_value=1, else_value=0)

        assert check_not_equal(5, 3) == 1
        assert check_not_equal(5, 5) == 0


# ==================== CONDITIONAL WITH EXPRESSIONS ====================

class TestConditionalExpressions(MLIRTestBase):
    """Test conditionals with complex expressions"""

    def test_if_with_arithmetic_branches(self):
        """Test If with arithmetic in branches"""
        @ml_function
        def conditional_calc(x: int, y: int) -> int:
            return If(x > y, then_value=x * 2, else_value=y * 2)

        assert conditional_calc(10, 5) == 20  # 10 > 5, so 10 * 2
        assert conditional_calc(3, 7) == 14   # 3 < 7, so 7 * 2

    def test_if_with_complex_condition(self):
        """Test If with complex expression in condition"""
        @ml_function
        def complex_condition(a: int, b: int, c: int) -> int:
            # Condition: (a + b) > c
            return If((a + b) > c, then_value=a + b, else_value=c)

        assert complex_condition(10, 5, 12) == 15  # (10+5)=15 > 12, return 15
        assert complex_condition(3, 2, 10) == 10   # (3+2)=5 < 10, return 10

    def test_if_with_complex_branches(self):
        """Test If with complex expressions in both branches"""
        @ml_function
        def nested_arithmetic(x: int, y: int) -> int:
            # if ((10 + 5) > 12) return (2 * 3) else return (20 - 5)
            condition = (x + 5) > 12
            then_branch = 2 * 3
            else_branch = 20 - 5
            return If(condition, then_value=then_branch, else_value=else_branch)

        assert nested_arithmetic(10, 0) == 6   # (10+5)=15 > 12, return 2*3=6
        assert nested_arithmetic(5, 0) == 15   # (5+5)=10 < 12, return 20-5=15


# ==================== FLOAT CONDITIONALS ====================

class TestFloatConditionals(MLIRTestBase):
    """Test conditional operations with floats"""

    def test_if_float_comparison(self):
        """Test If with float comparisons"""
        @ml_function
        def float_max(x: float, y: float) -> float:
            return If(x > y, then_value=x, else_value=y)

        result = float_max(3.5, 2.1)
        assert abs(result - 3.5) < 0.001

        result = float_max(1.5, 2.8)
        assert abs(result - 2.8) < 0.001

    def test_if_float_operations(self):
        """Test If with float arithmetic in branches"""
        @ml_function
        def float_conditional(x: float, y: float) -> float:
            return If(x > y, then_value=x + 1.5, else_value=y + 2.5)

        result = float_conditional(10.0, 5.0)
        assert abs(result - 11.5) < 0.001  # 10 > 5, so 10 + 1.5

        result = float_conditional(3.0, 7.0)
        assert abs(result - 9.5) < 0.001   # 3 < 7, so 7 + 2.5

    def test_if_mixed_type_comparison(self):
        """Test If with mixed int/float using explicit cast"""
        @ml_function
        def mixed_conditional(x: int, y: float) -> float:
            # x is int, y is float - explicit cast required for strict typing
            x_float = cast(x, f32)
            return If(x_float > y, then_value=x_float, else_value=y)

        result = mixed_conditional(10, 5.5)
        # 10.0 > 5.5, return 10.0
        assert abs(result - 10.0) < 0.001


# ==================== COMPARISON OPERATORS ====================

class TestComparisonOperators(MLIRTestBase):
    """Test all comparison operators"""

    def test_all_comparison_operators(self):
        """Test all comparison operators via overloading"""
        @ml_function
        def test_gt(x: int, y: int) -> int:
            return If(x > y, then_value=1, else_value=0)

        @ml_function
        def test_lt(x: int, y: int) -> int:
            return If(x < y, then_value=1, else_value=0)

        @ml_function
        def test_ge(x: int, y: int) -> int:
            return If(x >= y, then_value=1, else_value=0)

        @ml_function
        def test_le(x: int, y: int) -> int:
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

class TestParameterizedConditionals(MLIRTestBase):
    """Test conditionals with function parameters"""

    def test_if_with_parameters(self):
        """Test If using function parameters in all parts"""
        @ml_function
        def param_conditional(a: int, b: int, x: int, y: int) -> int:
            # Use parameters in condition and branches
            return If(a > b, then_value=x, else_value=y)

        assert param_conditional(10, 5, 100, 200) == 100
        assert param_conditional(3, 7, 100, 200) == 200

    def test_if_chained_logic(self):
        """Test multiple conditional-like logic"""
        @ml_function
        def clamp(value: int, min_val: int, max_val: int) -> int:
            # Clamp value between min and max
            # First check if value < min_val
            adjusted = If(value < min_val, then_value=min_val, else_value=value)
            # Then check if adjusted > max_val
            return If(adjusted > max_val, then_value=max_val, else_value=adjusted)

        assert clamp(5, 0, 10) == 5    # Within range
        assert clamp(-5, 0, 10) == 0   # Below min
        assert clamp(15, 0, 10) == 10  # Above max
