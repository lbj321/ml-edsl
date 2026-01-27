"""Tests for control flow constructs (Phase 5)

Comprehensive test suite for control flow operations including:
- Conditional expressions (If) with all comparison operators
- While loops with various operations and predicates
- Complex conditions and nested logic
- Integer and float operations
- Parameterized control flow

Test Organization:
1. Conditionals (If) - Basic, expressions, floats, comparisons
2. While Loops - Basic operations and various predicates
3. (Future) For Loops - Will be added when buildForEach is integrated
"""

import pytest
from mlir_edsl import ml_function, If, While, cast, f32
from mlir_edsl import ADD, SUB, MUL, DIV, SLT, SGT, NE
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


# ==================== CONDITIONALS (IF) ====================

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


# ==================== WHILE LOOPS ====================

class TestWhileLoop(MLIRTestBase):
    """Test while loop implementations with various operations"""

    def test_while_loop_basic(self):
        """Test basic while loop counting up"""
        @ml_function
        def count_up() -> int:
            # while(current < 5) { current = current + 1 } starting from 0
            # Computes: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (stop)
            return While(init=0, target=5, operation=ADD, predicate=SLT)

        result = count_up()
        assert result == 5, f"Expected 5, got {result}"

    def test_while_loop_multiplication(self):
        """Test while loop with multiplication (doubling)"""
        @ml_function
        def double_until() -> int:
            # while(current < 8) { current = current * 2 } starting from 1
            # Computes: 1 -> 2 -> 4 -> 8 (stop)
            return While(init=1, target=8, operation=MUL, predicate=SLT)

        result = double_until()
        assert result == 8, f"Expected 8, got {result}"

    def test_while_loop_subtraction(self):
        """Test while loop with subtraction (counting down)"""
        @ml_function
        def count_down() -> int:
            # while(current > 0) { current = current - 1 } starting from 5
            # Computes: 5 -> 4 -> 3 -> 2 -> 1 -> 0 (stop)
            return While(init=5, target=0, operation=SUB, predicate=SGT)

        result = count_down()
        assert result == 0, f"Expected 0, got {result}"

    def test_while_loop_division(self):
        """Test while loop with division (halving)"""
        @ml_function
        def halve_until() -> int:
            # while(current > 1) { current = current / 2 } starting from 16
            # Computes: 16 -> 8 -> 4 -> 2 -> 1 (stop)
            return While(init=16, target=1, operation=DIV, predicate=SGT)

        result = halve_until()
        assert result == 1, f"Expected 1, got {result}"

    def test_while_loop_equality_condition(self):
        """Test while loop with != condition"""
        @ml_function
        def until_equal() -> int:
            # while(current != 3) { current = current + 1 } starting from 0
            # Computes: 0 -> 1 -> 2 -> 3 (stop)
            return While(init=0, target=3, operation=ADD, predicate=NE)

        result = until_equal()
        assert result == 3, f"Expected 3, got {result}"


# ==================== FOR LOOPS (FUTURE) ====================
# Note: For loop tests will be added once buildForEach() is integrated with Python frontend
# See ROADMAP.md Phase 7 for array iteration implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
