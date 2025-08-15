"""Test high-level JIT integration with @ml_function decorator"""
import pytest
from mlir_edsl import ml_function, add, sub, mul, div

def test_ml_function_jit_execution():
    """Test JIT execution through @ml_function decorator"""
    
    @ml_function
    def add_example():
        return add(4, 6)
    
    # Test JIT execution
    result = add_example.execute()
    assert result == 10
    
    # Test that regular call still works (prints MLIR)
    ast_result = add_example()
    assert hasattr(ast_result, 'op')  # AST BinaryOp object

def test_ml_function_float_jit():
    """Test JIT execution with float operations"""
    
    @ml_function
    def float_mul():
        return mul(2.5, 4.0)
    
    result = float_mul.execute()
    assert abs(result - 10.0) < 1e-6

def test_ml_function_subtraction_jit():
    """Test JIT execution with subtraction"""
    
    @ml_function
    def sub_example():
        return sub(20, 8)
    
    result = sub_example.execute()
    assert result == 12

def test_ml_function_division_jit():
    """Test JIT execution with division"""
    
    @ml_function
    def div_example():
        return div(15.0, 3.0)
    
    result = div_example.execute()
    assert abs(result - 5.0) < 1e-6

def test_ml_function_complex_expression():
    """Test JIT execution with complex expression"""
    
    @ml_function
    def complex_example():
        a = add(10, 5)
        b = mul(2, 3)
        return sub(a, b)
    
    result = complex_example.execute()
    assert result == 9  # (10 + 5) - (2 * 3) = 15 - 6 = 9