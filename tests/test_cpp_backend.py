"""Tests for C++ MLIR backend - AST compilation and execution

This test suite directly tests the backend's AST compilation and execution.
It uses AST nodes directly rather than @ml_function to isolate backend testing.
"""

import pytest
from mlir_edsl.backend import HAS_CPP_BACKEND, get_backend, CppMLIRBackend
from mlir_edsl.ast import Constant, BinaryOp, CompareOp, IfOp, ForLoopOp, Parameter
from mlir_edsl.types import i32, f32, i1
from mlir_edsl import cast, ADD, SUB, MUL, DIV, SGT

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


@pytest.fixture
def backend():
    """Provide a clean backend instance for each test"""
    backend_instance = get_backend()
    backend_instance.clear_module()
    return backend_instance


# ==================== BACKEND AVAILABILITY ====================

def test_backend_availability():
    """Test that C++ backend is available and can be instantiated"""
    backend = get_backend()
    assert backend is not None
    assert isinstance(backend, CppMLIRBackend)


def test_backend_error_handling():
    """Test error handling when C++ backend is used incorrectly"""
    if not HAS_CPP_BACKEND:
        # Test that CppMLIRBackend raises error when backend unavailable
        with pytest.raises(RuntimeError, match="C++ backend not available"):
            CppMLIRBackend()
    else:
        # If backend is available, this should not raise
        backend = CppMLIRBackend()
        assert backend is not None


# ==================== BASIC ARITHMETIC COMPILATION ====================

def test_constant_addition(backend):
    """Test compiling constant addition"""
    # Create AST: 5 + 3
    left = Constant(5)
    right = Constant(3)
    result = BinaryOp(ADD, left, right)

    # Compile function
    backend.compile_function_from_ast("test_add", [], i32, result)
    mlir_code = backend.get_mlir_string()

    # Verify MLIR structure
    assert "func.func @test_add()" in mlir_code
    assert "-> i32" in mlir_code
    assert "arith.constant 5" in mlir_code
    assert "arith.constant 3" in mlir_code
    assert "arith.addi" in mlir_code

    # Test JIT execution
    executed_result = backend.execute_function("test_add")
    assert executed_result == 8


def test_constant_operations(backend):
    """Test all basic arithmetic operations"""
    # Addition
    add_result = BinaryOp(ADD, Constant(10), Constant(5))
    backend.compile_function_from_ast("test_ops_add", [], i32, add_result)
    assert backend.execute_function("test_ops_add") == 15

    # Subtraction
    sub_result = BinaryOp(SUB, Constant(10), Constant(3))
    backend.compile_function_from_ast("test_ops_sub", [], i32, sub_result)
    assert backend.execute_function("test_ops_sub") == 7

    # Multiplication
    mul_result = BinaryOp(MUL, Constant(6), Constant(4))
    backend.compile_function_from_ast("test_ops_mul", [], i32, mul_result)
    assert backend.execute_function("test_ops_mul") == 24

    # Division
    div_result = BinaryOp(DIV, Constant(20), Constant(4))
    backend.compile_function_from_ast("test_ops_div", [], i32, div_result)
    assert backend.execute_function("test_ops_div") == 5


def test_float_operations(backend):
    """Test float operations"""
    # Float addition
    left = Constant(5.5)
    right = Constant(2.5)
    result = BinaryOp(ADD, left, right)

    backend.compile_function_from_ast("test_float_add", [], f32, result)
    mlir_code = backend.get_mlir_string()

    # Verify float types in MLIR
    assert "-> f32" in mlir_code
    assert "arith.addf" in mlir_code

    # Test execution
    executed_result = backend.execute_function("test_float_add")
    assert abs(executed_result - 8.0) < 0.001


def test_type_promotion(backend):
    """Test mixed type promotion (int + float = float)"""
    int_val = Constant(5)
    float_val = Constant(2.5)
    result = BinaryOp(ADD, cast(int_val, f32), float_val)

    # Result should be float
    assert result.infer_type() == f32

    backend.compile_function_from_ast("test_promotion", [], f32, result)
    executed_result = backend.execute_function("test_promotion")
    assert abs(executed_result - 7.5) < 0.001


def test_complex_expression(backend):
    """Test complex nested expression: (5 + 3) * 2"""
    a = Constant(5)
    b = Constant(3)
    c = Constant(2)

    add_result = BinaryOp(ADD, a, b)
    final_result = BinaryOp(MUL, add_result, c)

    backend.compile_function_from_ast("test_complex", [], i32, final_result)
    mlir_code = backend.get_mlir_string()

    # Should contain multiple operations
    assert "arith.constant 5" in mlir_code
    assert "arith.constant 3" in mlir_code
    assert "arith.constant 2" in mlir_code
    assert mlir_code.count("arith.") >= 3

    # Test execution: (5 + 3) * 2 = 16
    executed_result = backend.execute_function("test_complex")
    assert executed_result == 16


# ==================== PARAMETERIZED FUNCTIONS ====================

def test_parameterized_function(backend):
    """Test function with parameters"""
    # Create function: add(x, y) = x + y
    param_x = Parameter("x", i32)
    param_y = Parameter("y", i32)
    result = BinaryOp(ADD, param_x, param_y)

    backend.compile_function_from_ast(
        "test_param_add",
        [("x", i32), ("y", i32)],
        i32,
        result
    )

    mlir_code = backend.get_mlir_string()
    assert "func.func @test_param_add(%arg0: i32, %arg1: i32)" in mlir_code

    # Test with different arguments
    assert backend.execute_function("test_param_add", 10, 5) == 15
    assert backend.execute_function("test_param_add", 100, 200) == 300


def test_float_parameters(backend):
    """Test function with float parameters"""
    param_x = Parameter("x", f32)
    param_y = Parameter("y", f32)
    result = BinaryOp(MUL, param_x, param_y)

    backend.compile_function_from_ast(
        "test_float_param",
        [("x", f32), ("y", f32)],
        f32,
        result
    )

    executed_result = backend.execute_function("test_float_param", 2.5, 4.0)
    assert abs(executed_result - 10.0) < 0.001


# ==================== CONDITIONAL OPERATIONS ====================

def test_comparison_compilation(backend):
    """Test comparison operation compilation"""
    left = Constant(5)
    right = Constant(3)
    comparison = CompareOp(SGT, left, right)

    # Comparisons return i1 (bool)
    assert comparison.infer_type() == i1

    # Use in an If to get a concrete value
    result = IfOp(comparison, Constant(1), Constant(0))
    backend.compile_function_from_ast("test_compare", [], i32, result)

    mlir_code = backend.get_mlir_string()
    assert "arith.cmpi" in mlir_code
    assert "scf.if" in mlir_code

    # 5 > 3 is true, so return 1
    executed_result = backend.execute_function("test_compare")
    assert executed_result == 1


def test_if_else_compilation(backend):
    """Test if-else operation compilation"""
    # if (10 > 5) return 100 else return 200
    condition = CompareOp(SGT, Constant(10), Constant(5))
    result = IfOp(condition, Constant(100), Constant(200))

    backend.compile_function_from_ast("test_if", [], i32, result)
    mlir_code = backend.get_mlir_string()

    assert "arith.cmpi" in mlir_code
    assert "scf.if" in mlir_code
    assert "scf.yield" in mlir_code

    # 10 > 5 is true, return 100
    executed_result = backend.execute_function("test_if")
    assert executed_result == 100


# ==================== LOOP OPERATIONS ====================

# def test_for_loop_compilation(backend):
#     """Test for loop compilation"""
#     # for(i = 0; i < 5; i += 1) accumulator += i, starting from 10
#     # Result: 10 + 0 + 1 + 2 + 3 + 4 = 20
#     start = Constant(0)
#     end = Constant(5)
#     step = Constant(1)
#     init_value = Constant(10)

#     result = ForLoopOp(start, end, step, init_value, "add")

#     backend.compile_function_from_ast("test_for", [], i32, result)
#     mlir_code = backend.get_mlir_string()

#     assert "scf.for" in mlir_code
#     assert "scf.yield" in mlir_code
#     assert "arith.addi" in mlir_code

#     executed_result = backend.execute_function("test_for")
#     assert executed_result == 20



# ==================== LLVM IR GENERATION ====================

def test_llvm_ir_generation(backend):
    """Test LLVM IR generation from MLIR"""
    # Simple addition
    result = BinaryOp(ADD, Constant(4), Constant(6))
    backend.compile_function_from_ast("add_fn", [], i32, result)

    llvm_ir = backend.get_llvm_ir_string()
    mlir_code = backend.get_mlir_string()

    # Verify LLVM IR structure
    assert "define" in llvm_ir
    assert "add_fn" in llvm_ir
    assert "ret" in llvm_ir
    assert not llvm_ir.startswith("ERROR:")

    print(f"\nGenerated MLIR:\n{mlir_code}")
    print(f"\nGenerated LLVM IR:\n{llvm_ir}")


def test_llvm_ir_float(backend):
    """Test LLVM IR generation with float operations"""
    result = BinaryOp(MUL, Constant(2.5), Constant(4.0))
    backend.compile_function_from_ast("mul_fn", [], f32, result)

    llvm_ir = backend.get_llvm_ir_string()

    # Verify float types
    assert "define" in llvm_ir
    assert "mul_fn" in llvm_ir
    assert ("float" in llvm_ir or "f32" in llvm_ir)
    assert "ret" in llvm_ir


# ==================== MODULE MANAGEMENT ====================

def test_has_function(backend):
    """Test has_function method"""
    assert not backend.has_function("test_func")

    result = BinaryOp(ADD, Constant(1), Constant(2))
    backend.compile_function_from_ast("test_func", [], i32, result)

    assert backend.has_function("test_func")


def test_list_functions(backend):
    """Test list_functions method"""
    # Compile multiple functions
    result1 = BinaryOp(ADD, Constant(1), Constant(2))
    backend.compile_function_from_ast("func1", [], i32, result1)

    result2 = BinaryOp(MUL, Constant(3), Constant(4))
    backend.compile_function_from_ast("func2", [], i32, result2)

    functions = backend.list_functions()
    assert "func1" in functions
    assert "func2" in functions


def test_clear_module(backend):
    """Test clear_module method"""
    # Compile a function
    result = BinaryOp(ADD, Constant(1), Constant(2))
    backend.compile_function_from_ast("test_func", [], i32, result)
    assert backend.has_function("test_func")

    # Clear module
    backend.clear_module()
    assert not backend.has_function("test_func")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
