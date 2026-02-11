"""Tests for C++ MLIR backend - AST compilation and execution

This test suite directly tests the backend's AST compilation and execution.
It uses AST nodes directly rather than @ml_function to isolate backend testing.
"""

import pytest
from mlir_edsl.ast import Constant, BinaryOp, CompareOp, IfOp, Parameter
from mlir_edsl.types import i32, f32, i1
from mlir_edsl import cast, ADD, SUB, MUL, DIV, SGT


# ==================== BACKEND AVAILABILITY ====================

class TestBackendAvailability:
    """Test backend instantiation and error handling"""

    def test_backend_available(self, backend):
        """Test that C++ backend is available and initialized"""
        assert backend is not None

    def test_backend_starts_empty(self, backend):
        """Test that backend starts with no compiled functions"""
        assert backend.list_functions() == []


# ==================== BASIC ARITHMETIC ====================

class TestBasicArithmetic:
    """Test basic arithmetic operations via AST compilation"""

    def test_constant_addition(self, backend):
        """Test compiling and executing constant addition"""
        result = BinaryOp(ADD, Constant(5), Constant(3))
        backend.compile_function_from_ast("test_add", [], i32, result)
        assert backend.execute_function("test_add") == 8

    def test_all_arithmetic_operations(self, backend):
        """Test all four arithmetic operations"""
        add_result = BinaryOp(ADD, Constant(10), Constant(5))
        backend.compile_function_from_ast("test_add", [], i32, add_result)
        assert backend.execute_function("test_add") == 15

        sub_result = BinaryOp(SUB, Constant(10), Constant(3))
        backend.compile_function_from_ast("test_sub", [], i32, sub_result)
        assert backend.execute_function("test_sub") == 7

        mul_result = BinaryOp(MUL, Constant(6), Constant(4))
        backend.compile_function_from_ast("test_mul", [], i32, mul_result)
        assert backend.execute_function("test_mul") == 24

        div_result = BinaryOp(DIV, Constant(20), Constant(4))
        backend.compile_function_from_ast("test_div", [], i32, div_result)
        assert backend.execute_function("test_div") == 5

    def test_float_addition(self, backend):
        """Test float addition execution"""
        result = BinaryOp(ADD, Constant(5.5), Constant(2.5))
        backend.compile_function_from_ast("test_float_add", [], f32, result)
        assert abs(backend.execute_function("test_float_add") - 8.0) < 0.001

    def test_type_promotion(self, backend):
        """Test mixed type promotion (cast int to float, then add)"""
        int_val = Constant(5)
        float_val = Constant(2.5)
        result = BinaryOp(ADD, cast(int_val, f32), float_val)

        assert result.infer_type() == f32

        backend.compile_function_from_ast("test_promotion", [], f32, result)
        assert abs(backend.execute_function("test_promotion") - 7.5) < 0.001

    def test_complex_expression(self, backend):
        """Test nested expression: (5 + 3) * 2 = 16"""
        add_result = BinaryOp(ADD, Constant(5), Constant(3))
        final_result = BinaryOp(MUL, add_result, Constant(2))

        backend.compile_function_from_ast("test_complex", [], i32, final_result)
        assert backend.execute_function("test_complex") == 16


# ==================== PARAMETERIZED FUNCTIONS ====================

class TestParameterizedFunctions:
    """Test functions with parameters"""

    def test_integer_parameters(self, backend):
        """Test function with integer parameters: add(x, y) = x + y"""
        param_x = Parameter("x", i32)
        param_y = Parameter("y", i32)
        result = BinaryOp(ADD, param_x, param_y)

        backend.compile_function_from_ast(
            "test_param_add", [("x", i32), ("y", i32)], i32, result
        )

        assert backend.execute_function("test_param_add", 10, 5) == 15
        assert backend.execute_function("test_param_add", 100, 200) == 300

    def test_float_parameters(self, backend):
        """Test function with float parameters: mul(x, y) = x * y"""
        param_x = Parameter("x", f32)
        param_y = Parameter("y", f32)
        result = BinaryOp(MUL, param_x, param_y)

        backend.compile_function_from_ast(
            "test_float_param", [("x", f32), ("y", f32)], f32, result
        )

        assert abs(backend.execute_function("test_float_param", 2.5, 4.0) - 10.0) < 0.001


# ==================== CONDITIONAL OPERATIONS ====================

class TestConditionalOperations:
    """Test comparison and if-else operations"""

    def test_comparison_in_conditional(self, backend):
        """Test comparison used in if-else returns correct branch"""
        comparison = CompareOp(SGT, Constant(5), Constant(3))
        assert comparison.infer_type() == i1

        result = IfOp(comparison, Constant(1), Constant(0))
        backend.compile_function_from_ast("test_compare", [], i32, result)
        assert backend.execute_function("test_compare") == 1

    def test_if_else_execution(self, backend):
        """Test if-else returns correct branch value"""
        condition = CompareOp(SGT, Constant(10), Constant(5))
        result = IfOp(condition, Constant(100), Constant(200))

        backend.compile_function_from_ast("test_if", [], i32, result)
        assert backend.execute_function("test_if") == 100


# ==================== MODULE MANAGEMENT ====================

class TestModuleManagement:
    """Test module management operations"""

    def test_has_function(self, backend):
        """Test has_function reports correctly before and after compilation"""
        assert not backend.has_function("test_func")

        result = BinaryOp(ADD, Constant(1), Constant(2))
        backend.compile_function_from_ast("test_func", [], i32, result)
        assert backend.has_function("test_func")

    def test_list_functions(self, backend):
        """Test list_functions returns all compiled function names"""
        result1 = BinaryOp(ADD, Constant(1), Constant(2))
        backend.compile_function_from_ast("func1", [], i32, result1)

        result2 = BinaryOp(MUL, Constant(3), Constant(4))
        backend.compile_function_from_ast("func2", [], i32, result2)

        functions = backend.list_functions()
        assert "func1" in functions
        assert "func2" in functions

    def test_clear_module(self, backend):
        """Test clear_module removes all compiled functions"""
        result = BinaryOp(ADD, Constant(1), Constant(2))
        backend.compile_function_from_ast("test_func", [], i32, result)
        assert backend.has_function("test_func")

        backend.clear_module()
        assert not backend.has_function("test_func")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
