"""Tests for for loop and while loop implementations"""

import pytest
from mlir_edsl.backend import HAS_CPP_BACKEND, get_backend, CppMLIRBuilder
from mlir_edsl.loop_ops import LoopOp

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


@pytest.fixture
def backend():
    """Provide a clean backend instance for each test"""
    backend_instance = get_backend()
    backend_instance.reset()
    return backend_instance


def test_cpp_for_loop_basic(backend):
    """Test basic for loop with C++ backend"""
    # Create loop bounds: for(i = 0; i < 5; i += 1) with accumulator starting at 10
    start = backend.constant(0)
    end = backend.constant(5) 
    step = backend.constant(1)
    init_value = backend.constant(10)
    
    # Use LoopOp.ADD to add induction variable to accumulator
    # This would compute: 10 + 0 + 1 + 2 + 3 + 4 = 20
    result = backend.for_loop(start, end, step, init_value, LoopOp.ADD)
    
    # Generate function and verify MLIR
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_for", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated For Loop MLIR:\n{mlir_code}")
    
    # Verify for loop structure
    assert "scf.for" in mlir_code
    assert "scf.yield" in mlir_code
    assert "arith.constant 0" in mlir_code
    assert "arith.constant 5" in mlir_code
    assert "arith.constant 1" in mlir_code
    assert "arith.constant 10" in mlir_code
    
    # Test JIT execution - should compute: 10 + 0 + 1 + 2 + 3 + 4 = 20
    executed_result = backend.execute_function("test_for", result)
    print(f"JIT execution result: {executed_result}")
    assert executed_result == 20, f"Expected 20, got {executed_result}"


def test_cpp_for_loop_multiplication(backend):
    """Test for loop with multiplication operation"""
    # for(i = 1; i <= 4; i++) result *= i  (factorial-like)
    start = backend.constant(1)
    end = backend.constant(5)  # exclusive upper bound
    step = backend.constant(1)
    init_value = backend.constant(1)  # Start with 1 for multiplication
    
    # Use LoopOp.MUL for multiplication
    result = backend.for_loop(start, end, step, init_value, LoopOp.MUL)
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_factorial", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated Factorial-like MLIR:\n{mlir_code}")
    
    # Verify structure
    assert "scf.for" in mlir_code
    assert "arith.muli" in mlir_code
    assert "scf.yield" in mlir_code
    
    # Test JIT execution - should compute: 1 * 1 * 2 * 3 * 4 = 24
    executed_result = backend.execute_function("test_factorial", result)
    print(f"JIT execution result: {executed_result}")
    assert executed_result == 24, f"Expected 24, got {executed_result}"


def test_cpp_for_loop_float(backend):
    """Test for loop with float operations"""
    start = backend.constant(0)
    end = backend.constant(3)
    step = backend.constant(1)
    init_value = backend.constant(0.5)  # Float initial value
    
    # Use LoopOp.ADD - the C++ backend will handle int->float conversion automatically
    result = backend.for_loop(start, end, step, init_value, LoopOp.ADD)
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_float_for", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated Float For Loop MLIR:\n{mlir_code}")
    
    # Verify float operations
    assert "scf.for" in mlir_code
    assert "arith.constant 5.000000e-01" in mlir_code or "arith.constant 0.5" in mlir_code
    assert "arith.addf" in mlir_code
    assert "-> f32" in mlir_code
    
    # Test JIT execution - should compute: 0.5 + 0.0 + 1.0 + 2.0 = 3.5
    executed_result = backend.execute_function("test_float_for", result)
    print(f"JIT execution result: {executed_result}")
    expected = 0.5 + 0.0 + 1.0 + 2.0  # 3.5
    assert abs(executed_result - expected) < 0.001, f"Expected {expected}, got {executed_result}"


def test_cpp_for_loop_subtraction(backend):
    """Test for loop with subtraction operation"""
    # for(i = 0; i < 3; i++) result = result - i
    # Should compute: 10 - 0 - 1 - 2 = 7
    start = backend.constant(0)
    end = backend.constant(3)
    step = backend.constant(1)
    init_value = backend.constant(10)
    
    result = backend.for_loop(start, end, step, init_value, LoopOp.SUB)
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_sub_for", result)
    
    # Test JIT execution - should compute: 10 - 0 - 1 - 2 = 7
    executed_result = backend.execute_function("test_sub_for", result)
    print(f"JIT subtraction result: {executed_result}")
    assert executed_result == 7, f"Expected 7, got {executed_result}"


def test_cpp_for_loop_division(backend):
    """Test for loop with division operation"""
    # for(i = 2; i < 4; i++) result = result / i
    # Should compute: 24 / 2 / 3 = 4
    start = backend.constant(2)  # Start from 2 to avoid division by 0
    end = backend.constant(4)
    step = backend.constant(1)
    init_value = backend.constant(24)  # Use 24 so we get clean integer division
    
    result = backend.for_loop(start, end, step, init_value, LoopOp.DIV)
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_div_for", result)
    
    # Test JIT execution - should compute: 24 / 2 / 3 = 4
    executed_result = backend.execute_function("test_div_for", result)
    print(f"JIT division result: {executed_result}")
    assert executed_result == 4, f"Expected 4, got {executed_result}"


# ========== WHILE LOOP TESTS ==========

def test_cpp_while_loop_basic(backend):
    """Test basic while loop with C++ backend"""
    # while(current < 5) { current = current + 1 } starting from 0
    # Should compute: 0 -> 1 -> 2 -> 3 -> 4 -> 5 (stop)
    init_value = backend.constant(0)
    target = backend.constant(5)
    
    result = backend.while_loop(init_value, target, LoopOp.ADD, "slt")
    
    # Generate function and verify MLIR
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated While Loop MLIR:\n{mlir_code}")
    
    # Verify while loop structure
    assert "scf.while" in mlir_code
    assert "scf.condition" in mlir_code
    assert "scf.yield" in mlir_code
    assert "arith.constant 0" in mlir_code
    assert "arith.constant 5" in mlir_code
    assert "arith.constant 1" in mlir_code  # step for ADD operation
    
    # Test JIT execution - should result in 5 (stops when current >= 5)
    executed_result = backend.execute_function("test_while", result)
    print(f"JIT execution result: {executed_result}")
    assert executed_result == 5, f"Expected 5, got {executed_result}"


def test_cpp_while_loop_multiplication(backend):
    """Test while loop with multiplication operation"""
    # while(current < 8) { current = current * 2 } starting from 1
    # Should compute: 1 -> 2 -> 4 -> 8 (stop)
    init_value = backend.constant(1)
    target = backend.constant(8)
    
    result = backend.while_loop(init_value, target, LoopOp.MUL, "slt")
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while_mul", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated While Multiplication MLIR:\n{mlir_code}")
    
    # Verify structure
    assert "scf.while" in mlir_code
    assert "arith.muli" in mlir_code
    assert "arith.constant 2" in mlir_code  # step for MUL operation
    
    # Test JIT execution - should result in 8
    executed_result = backend.execute_function("test_while_mul", result)
    print(f"JIT execution result: {executed_result}")
    assert executed_result == 8, f"Expected 8, got {executed_result}"


def test_cpp_while_loop_float(backend):
    """Test while loop with float operations"""
    # while(current < 2.5) { current = current + 1 } starting from 0.5
    # Should compute: 0.5 -> 1.5 -> 2.5 (stop)
    init_value = backend.constant(0.5)
    target = backend.constant(2.5)
    
    result = backend.while_loop(init_value, target, LoopOp.ADD, "olt")
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while_float", result)
    mlir_code = backend.get_mlir_string()
    
    print(f"\nGenerated While Float MLIR:\n{mlir_code}")
    
    # Verify float operations
    assert "scf.while" in mlir_code
    assert "arith.constant 5.000000e-01" in mlir_code or "arith.constant 0.5" in mlir_code
    assert "arith.addf" in mlir_code
    assert "-> f32" in mlir_code
    
    # Test JIT execution - should result in 2.5
    executed_result = backend.execute_function("test_while_float", result)
    print(f"JIT execution result: {executed_result}")
    expected = 2.5
    assert abs(executed_result - expected) < 0.001, f"Expected {expected}, got {executed_result}"


def test_cpp_while_loop_subtraction(backend):
    """Test while loop with subtraction operation"""
    # while(current > 0) { current = current - 1 } starting from 5
    # Should compute: 5 -> 4 -> 3 -> 2 -> 1 -> 0 (stop)
    init_value = backend.constant(5)
    target = backend.constant(0)
    
    result = backend.while_loop(init_value, target, LoopOp.SUB, "sgt")
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while_sub", result)
    
    # Test JIT execution - should result in 0
    executed_result = backend.execute_function("test_while_sub", result)
    print(f"JIT subtraction result: {executed_result}")
    assert executed_result == 0, f"Expected 0, got {executed_result}"


def test_cpp_while_loop_division(backend):
    """Test while loop with division operation"""
    # while(current > 1) { current = current / 2 } starting from 16
    # Should compute: 16 -> 8 -> 4 -> 2 -> 1 (stop)
    init_value = backend.constant(16)
    target = backend.constant(1)
    
    result = backend.while_loop(init_value, target, LoopOp.DIV, "sgt")
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while_div", result)
    
    # Test JIT execution - should result in 1
    executed_result = backend.execute_function("test_while_div", result)
    print(f"JIT division result: {executed_result}")
    assert executed_result == 1, f"Expected 1, got {executed_result}"


def test_cpp_while_loop_equality_condition(backend):
    """Test while loop with equality condition"""
    # while(current != 3) { current = current + 1 } starting from 0
    # Should compute: 0 -> 1 -> 2 -> 3 (stop)
    init_value = backend.constant(0)
    target = backend.constant(3)
    
    result = backend.while_loop(init_value, target, LoopOp.ADD, "ne")
    
    backend.create_function_with_params_setup([])
    backend.finalize_function_with_params("test_while_ne", result)
    
    # Test JIT execution - should result in 3
    executed_result = backend.execute_function("test_while_ne", result)
    print(f"JIT equality condition result: {executed_result}")
    assert executed_result == 3, f"Expected 3, got {executed_result}"


if __name__ == "__main__":
    pytest.main([__file__])