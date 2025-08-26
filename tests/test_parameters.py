"""Test parameter functionality for ML functions"""

import pytest
from mlir_edsl import ml_function, add, sub, mul, div

class TestParameterFunctionality:
    """Test Phase 6.3: Function Parameters"""
    
    def test_basic_two_parameters(self):
        """Test function with two integer parameters"""
        @ml_function
        def add_params(a, b):
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
        def double_value(x):
            return mul(x, 2)
        
        result = double_value(5)
        assert result == 10
    
    def test_three_parameters(self):
        """Test function with three parameters"""
        @ml_function
        def three_param_ops(a, b, c):
            temp = add(a, b)
            return sub(temp, c)
        
        result = three_param_ops(10, 5, 3)  # (10 + 5) - 3 = 12
        assert result == 12
    
    def test_complex_expressions(self):
        """Test complex expressions with parameters"""
        @ml_function
        def complex_calc(x, y):
            # ((x * 2) + y) - (x / 2)
            doubled = mul(x, 2)      # x * 2
            halved = div(x, 2)       # x / 2  
            sum_val = add(doubled, y) # (x * 2) + y
            return sub(sum_val, halved) # result - (x / 2)
        
        result = complex_calc(4, 3)  # ((4*2) + 3) - (4/2) = (8 + 3) - 2 = 9
        assert result == 9
    
    def test_parameter_type_inference(self):
        """Test that parameter types are correctly inferred"""
        @ml_function
        def mixed_types(a, b):
            return add(a, b)  # Should promote int + float -> float
        
        # This would test int + float promotion when we support floats
        # For now, test with integers
        result = mixed_types(7, 8)
        assert result == 15
    
    def test_multiple_function_calls_no_collision(self):
        """Test multiple function calls don't interfere with each other"""
        @ml_function
        def func1(a, b):
            return add(a, b)
        
        @ml_function 
        def func2(x, y):
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
        def reusable_func(a, b):
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
        def constant_func():
            return add(5, 3)
        
        result = constant_func()
        assert result == 8
    
    def test_parameter_order_matters(self):
        """Test that parameter order is preserved"""
        @ml_function
        def order_test(a, b):
            return sub(a, b)  # a - b != b - a
        
        result1 = order_test(10, 3)  # 10 - 3 = 7
        assert result1 == 7
        
        result2 = order_test(3, 10)  # 3 - 10 = -7
        assert result2 == -7

if __name__ == "__main__":
    # Run tests directly
    test = TestParameterFunctionality()
    
    print("Testing Phase 6.3: Function Parameters...")
    
    try:
        test.test_basic_two_parameters()
        print("✅ Basic two parameters")
        
        test.test_single_parameter()  
        print("✅ Single parameter")
        
        test.test_three_parameters()
        print("✅ Three parameters")
        
        test.test_complex_expressions()
        print("✅ Complex expressions")
        
        test.test_parameter_type_inference()
        print("✅ Parameter type inference")
        
        test.test_multiple_function_calls_no_collision()
        print("✅ Multiple function calls")
        
        test.test_parameter_reuse_same_function()
        print("✅ Parameter reuse")
        
        test.test_zero_parameters()
        print("✅ Zero parameters")
        
        test.test_parameter_order_matters()
        print("✅ Parameter order")
        
        print("\n🎉 All parameter tests passed! Phase 6.3 Complete!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()