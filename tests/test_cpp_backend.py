"""Tests for C++ MLIR backend - AST compilation and execution

This test suite directly tests the backend's AST compilation and execution.
It uses AST nodes directly rather than @ml_function to isolate backend testing.
"""

import pytest
from mlir_edsl.backend import HAS_CPP_BACKEND
from mlir_edsl.ast import Constant, BinaryOp, CompareOp, IfOp, Parameter
from mlir_edsl.types import i32, f32, i1
from mlir_edsl import cast, ADD, SUB, MUL, DIV, SGT
from tests.test_base import MLIRTestBase

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


# ==================== BACKEND AVAILABILITY ====================

class TestBackendAvailability(MLIRTestBase):
    """Test backend instantiation and error handling"""

    def test_backend_available(self):
        """Test that C++ backend is available and initialized"""
        assert self.backend is not None

    def test_backend_starts_empty(self):
        """Test that backend starts with no compiled functions"""
        assert self.backend.list_functions() == []


# ==================== BASIC ARITHMETIC ====================

class TestBasicArithmetic(MLIRTestBase):
    """Test basic arithmetic operations via AST compilation"""

    def test_constant_addition(self):
        """Test compiling and executing constant addition"""
        result = BinaryOp(ADD, Constant(5), Constant(3))
        self.backend.compile_function_from_ast("test_add", [], i32, result)
        assert self.backend.execute_function("test_add") == 8

    def test_all_arithmetic_operations(self):
        """Test all four arithmetic operations"""
        add_result = BinaryOp(ADD, Constant(10), Constant(5))
        self.backend.compile_function_from_ast("test_add", [], i32, add_result)
        assert self.backend.execute_function("test_add") == 15

        sub_result = BinaryOp(SUB, Constant(10), Constant(3))
        self.backend.compile_function_from_ast("test_sub", [], i32, sub_result)
        assert self.backend.execute_function("test_sub") == 7

        mul_result = BinaryOp(MUL, Constant(6), Constant(4))
        self.backend.compile_function_from_ast("test_mul", [], i32, mul_result)
        assert self.backend.execute_function("test_mul") == 24

        div_result = BinaryOp(DIV, Constant(20), Constant(4))
        self.backend.compile_function_from_ast("test_div", [], i32, div_result)
        assert self.backend.execute_function("test_div") == 5

    def test_float_addition(self):
        """Test float addition execution"""
        result = BinaryOp(ADD, Constant(5.5), Constant(2.5))
        self.backend.compile_function_from_ast("test_float_add", [], f32, result)
        assert abs(self.backend.execute_function("test_float_add") - 8.0) < 0.001

    def test_type_promotion(self):
        """Test mixed type promotion (cast int to float, then add)"""
        int_val = Constant(5)
        float_val = Constant(2.5)
        result = BinaryOp(ADD, cast(int_val, f32), float_val)

        assert result.infer_type() == f32

        self.backend.compile_function_from_ast("test_promotion", [], f32, result)
        assert abs(self.backend.execute_function("test_promotion") - 7.5) < 0.001

    def test_complex_expression(self):
        """Test nested expression: (5 + 3) * 2 = 16"""
        add_result = BinaryOp(ADD, Constant(5), Constant(3))
        final_result = BinaryOp(MUL, add_result, Constant(2))

        self.backend.compile_function_from_ast("test_complex", [], i32, final_result)
        assert self.backend.execute_function("test_complex") == 16


# ==================== PARAMETERIZED FUNCTIONS ====================

class TestParameterizedFunctions(MLIRTestBase):
    """Test functions with parameters"""

    def test_integer_parameters(self):
        """Test function with integer parameters: add(x, y) = x + y"""
        param_x = Parameter("x", i32)
        param_y = Parameter("y", i32)
        result = BinaryOp(ADD, param_x, param_y)

        self.backend.compile_function_from_ast(
            "test_param_add", [("x", i32), ("y", i32)], i32, result
        )

        assert self.backend.execute_function("test_param_add", 10, 5) == 15
        assert self.backend.execute_function("test_param_add", 100, 200) == 300

    def test_float_parameters(self):
        """Test function with float parameters: mul(x, y) = x * y"""
        param_x = Parameter("x", f32)
        param_y = Parameter("y", f32)
        result = BinaryOp(MUL, param_x, param_y)

        self.backend.compile_function_from_ast(
            "test_float_param", [("x", f32), ("y", f32)], f32, result
        )

        assert abs(self.backend.execute_function("test_float_param", 2.5, 4.0) - 10.0) < 0.001


# ==================== CONDITIONAL OPERATIONS ====================

class TestConditionalOperations(MLIRTestBase):
    """Test comparison and if-else operations"""

    def test_comparison_in_conditional(self):
        """Test comparison used in if-else returns correct branch"""
        comparison = CompareOp(SGT, Constant(5), Constant(3))
        assert comparison.infer_type() == i1

        result = IfOp(comparison, Constant(1), Constant(0))
        self.backend.compile_function_from_ast("test_compare", [], i32, result)
        assert self.backend.execute_function("test_compare") == 1

    def test_if_else_execution(self):
        """Test if-else returns correct branch value"""
        condition = CompareOp(SGT, Constant(10), Constant(5))
        result = IfOp(condition, Constant(100), Constant(200))

        self.backend.compile_function_from_ast("test_if", [], i32, result)
        assert self.backend.execute_function("test_if") == 100


# ==================== MODULE MANAGEMENT ====================

class TestModuleManagement(MLIRTestBase):
    """Test module management operations"""

    def test_has_function(self):
        """Test has_function reports correctly before and after compilation"""
        assert not self.backend.has_function("test_func")

        result = BinaryOp(ADD, Constant(1), Constant(2))
        self.backend.compile_function_from_ast("test_func", [], i32, result)
        assert self.backend.has_function("test_func")

    def test_list_functions(self):
        """Test list_functions returns all compiled function names"""
        result1 = BinaryOp(ADD, Constant(1), Constant(2))
        self.backend.compile_function_from_ast("func1", [], i32, result1)

        result2 = BinaryOp(MUL, Constant(3), Constant(4))
        self.backend.compile_function_from_ast("func2", [], i32, result2)

        functions = self.backend.list_functions()
        assert "func1" in functions
        assert "func2" in functions

    def test_clear_module(self):
        """Test clear_module removes all compiled functions"""
        result = BinaryOp(ADD, Constant(1), Constant(2))
        self.backend.compile_function_from_ast("test_func", [], i32, result)
        assert self.backend.has_function("test_func")

        self.backend.clear_module()
        assert not self.backend.has_function("test_func")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
