"""Test parameter functionality for ML functions with strict typing"""

import pytest
from mlir_edsl import ml_function, add, sub, mul, div, cast
from mlir_edsl import i32, f32, i1
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase

class TestParameterFunctionality(MLIRTestBase):
    """Test Phase 6.3: Function Parameters"""
    
    def test_basic_two_parameters(self):
        """Test function with two integer parameters"""
        @ml_function
        def add_params(a: int, b: int) -> int:
            return add(a, b)

        # Test with first set of parameters
        result1 = add_params(2, 3)
        assert result1 == 5

        # Test with different parameters (should not collide)
        result2 = add_params(10, 20)
        assert result2 == 30
    
    def test_single_parameter(self):
        """Test function with single parameter"""
        @ml_function
        def double_value(x: int) -> int:
            return mul(x, 2)

        result = double_value(5)
        assert result == 10
    
    def test_three_parameters(self):
        """Test function with three parameters"""
        @ml_function
        def three_param_ops(a: int, b: int, c: int) -> int:
            temp = add(a, b)
            return sub(temp, c)

        result = three_param_ops(10, 5, 3)  # (10 + 5) - 3 = 12
        assert result == 12
    
    def test_complex_expressions(self):
        """Test complex expressions with parameters"""
        @ml_function
        def complex_calc(x: int, y: int) -> int:
            # ((x * 2) + y) - (x / 2)
            doubled = mul(x, 2)      # x * 2
            halved = div(x, 2)       # x / 2
            sum_val = add(doubled, y) # (x * 2) + y
            return sub(sum_val, halved) # result - (x / 2)

        result = complex_calc(4, 3)  # ((4*2) + 3) - (4/2) = (8 + 3) - 2 = 9
        assert result == 9
    
    def test_explicit_type_casting(self):
        """Test explicit type casting in strict typing system"""
        @ml_function
        def mixed_types_with_cast(a: int, b: float) -> float:
            # Strict typing requires explicit cast - no auto promotion
            return add(cast(a, f32), b)

        # Test that casting works correctly
        result = mixed_types_with_cast(7, 8.0)
        assert abs(result - 15.0) < 0.001
    
    def test_multiple_function_calls_no_collision(self):
        """Test multiple function calls don't interfere with each other"""
        @ml_function
        def func1(a: int, b: int) -> int:
            return add(a, b)

        @ml_function
        def func2(x: int, y: int) -> int:
            return mul(x, y)

        # Call both functions with different parameters
        result1 = func1(2, 3)
        result2 = func2(4, 5)

        assert result1 == 5
        assert result2 == 20

        # Call first function again with new parameters
        result3 = func1(10, 15)
        assert result3 == 25
    
    def test_parameter_reuse_same_function(self):
        """Test calling same function multiple times with different parameters"""
        @ml_function
        def reusable_func(a: int, b: int) -> int:
            return sub(mul(a, 3), b)  # (a * 3) - b

        # First call: (2 * 3) - 1 = 5
        result1 = reusable_func(2, 1)
        assert result1 == 5

        # Second call: (5 * 3) - 2 = 13
        result2 = reusable_func(5, 2)
        assert result2 == 13

        # Third call: (1 * 3) - 4 = -1
        result3 = reusable_func(1, 4)
        assert result3 == -1
    
    def test_zero_parameters(self):
        """Test function with no parameters (regression test)"""
        @ml_function
        def constant_func() -> int:
            return add(5, 3)

        result = constant_func()
        assert result == 8
    
    def test_parameter_order_matters(self):
        """Test that parameter order is preserved"""
        @ml_function
        def order_test(a: int, b: int) -> int:
            return sub(a, b)  # a - b != b - a

        result1 = order_test(10, 3)  # 10 - 3 = 7
        assert result1 == 7

        result2 = order_test(3, 10)  # 3 - 10 = -7
        assert result2 == -7