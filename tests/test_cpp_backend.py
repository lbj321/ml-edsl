"""Tests for C++ MLIR backend integration"""

import pytest
from mlir_edsl.backend import HAS_CPP_BACKEND, get_backend, CppMLIRBuilder

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


def test_backend_availability():
    """Test that C++ backend is available and can be instantiated"""
    backend = get_backend()
    assert backend is not None
    assert isinstance(backend, CppMLIRBuilder)


def test_cpp_constant_creation():
    """Test creating constants with C++ backend"""
    backend = get_backend()
    
    # Test integer constant
    int_val = backend.constant(42)
    assert int_val.type == "i32"
    
    # Test float constant
    float_val = backend.constant(3.14)
    assert float_val.type == "f32"


def test_cpp_addition():
    """Test addition operation with C++ backend"""
    backend = get_backend()
    
    left = backend.constant(5)
    right = backend.constant(3)
    result = backend.add(left, right)
    
    assert result.type == "i32"


def test_cpp_subtraction():
    """Test subtraction operation with C++ backend"""
    backend = get_backend()
    
    left = backend.constant(10)
    right = backend.constant(3)
    result = backend.sub(left, right)
    
    assert result.type == "i32"


def test_cpp_multiplication():
    """Test multiplication operation with C++ backend"""
    backend = get_backend()
    
    left = backend.constant(6)
    right = backend.constant(4)
    result = backend.mul(left, right)
    
    assert result.type == "i32"


def test_cpp_division():
    """Test division operation with C++ backend"""
    backend = get_backend()
    
    left = backend.constant(20)
    right = backend.constant(4)
    result = backend.div(left, right)
    
    assert result.type == "i32"


def test_cpp_float_operations():
    """Test operations with float types"""
    backend = get_backend()
    
    left = backend.constant(5.5)
    right = backend.constant(2.0)
    
    # Test all operations with floats
    add_result = backend.add(left, right)
    assert add_result.type == "f32"
    
    sub_result = backend.sub(left, right)
    assert sub_result.type == "f32"
    
    mul_result = backend.mul(left, right)
    assert mul_result.type == "f32"
    
    div_result = backend.div(left, right)
    assert div_result.type == "f32"


def test_cpp_mixed_type_promotion():
    """Test mixed type promotion (int + float = float)"""
    backend = get_backend()
    
    int_val = backend.constant(5)
    float_val = backend.constant(2.5)
    
    # int + float should result in float
    result = backend.add(int_val, float_val)
    assert result.type == "f32"
    
    # float + int should result in float
    result2 = backend.add(float_val, int_val)
    assert result2.type == "f32"


def test_cpp_function_generation():
    """Test generating a complete MLIR function"""
    backend = get_backend()
    
    # Create a simple addition
    left = backend.constant(10)
    right = backend.constant(20)
    result = backend.add(left, right)
    
    # Generate function
    mlir_code = backend.create_function("test_add", result)
    
    # Verify basic structure
    assert "func.func @test_add()" in mlir_code
    assert "-> i32" in mlir_code
    assert "return" in mlir_code  # MLIR uses "return", not "func.return"
    assert "arith.constant 10" in mlir_code
    assert "arith.constant 20" in mlir_code


def test_cpp_float_function_generation():
    """Test generating MLIR function with float result"""
    backend = get_backend()
    
    # Create float operation
    left = backend.constant(3.14)
    right = backend.constant(2.86)
    result = backend.mul(left, right)
    
    # Generate function
    mlir_code = backend.create_function("test_float_mul", result)
    
    # Verify float types
    assert "func.func @test_float_mul()" in mlir_code
    assert "-> f32" in mlir_code
    assert "arith.constant 3.14" in mlir_code
    assert "arith.constant 2.86" in mlir_code


def test_cpp_llvm_ir_generation():
    """Test generating LLVM IR from MLIR"""
    backend = get_backend()
    
    # Create a simple addition
    left = backend.constant(4)
    right = backend.constant(6)
    result = backend.add(left, right)
    
    # Create function but get LLVM IR before reset
    backend.builder.create_function("add_fn", result.cpp_value)
    llvm_ir = backend.get_llvm_ir_string()
    
    # Also get MLIR for comparison
    mlir_code = backend.builder.get_mlir_string()
    backend.reset()  # Manual reset
    
    # Verify LLVM IR structure
    assert "define" in llvm_ir
    assert "add_fn" in llvm_ir
    assert "ret" in llvm_ir
    assert not llvm_ir.startswith("ERROR:")
    assert not llvm_ir.startswith("TODO:")
    
    print(f"\nGenerated MLIR:\n{mlir_code}")
    print(f"\nGenerated LLVM IR:\n{llvm_ir}")


def test_cpp_llvm_ir_float():
    """Test LLVM IR generation with float operations"""
    backend = get_backend()
    
    # Create float multiplication
    left = backend.constant(2.5)
    right = backend.constant(4.0)
    result = backend.mul(left, right)
    
    # Create function but get LLVM IR before reset
    backend.builder.create_function("mul_fn", result.cpp_value)
    llvm_ir = backend.get_llvm_ir_string()
    
    # Also get MLIR for comparison
    mlir_code = backend.builder.get_mlir_string()
    backend.reset()  # Manual reset
    
    # Verify LLVM IR contains float operations
    assert "define" in llvm_ir
    assert "mul_fn" in llvm_ir
    assert "float" in llvm_ir or "f32" in llvm_ir
    assert "ret" in llvm_ir
    assert not llvm_ir.startswith("ERROR:")
    
    print(f"\nGenerated Float MLIR:\n{mlir_code}")
    print(f"\nGenerated Float LLVM IR:\n{llvm_ir}")


def test_cpp_complex_expression():
    """Test building a complex expression with C++ backend"""
    backend = get_backend()
    
    # Build: (5 + 3) * 2
    a = backend.constant(5)
    b = backend.constant(3)
    c = backend.constant(2)
    
    add_result = backend.add(a, b)
    final_result = backend.mul(add_result, c)
    
    mlir_code = backend.create_function("complex_expr", final_result)
    
    # Should contain multiple constants and operations
    assert "arith.constant 5" in mlir_code
    assert "arith.constant 3" in mlir_code
    assert "arith.constant 2" in mlir_code
    assert mlir_code.count("arith.") >= 3  # At least 3 arith operations


def test_cpp_reset_functionality():
    """Test that reset allows creating multiple functions"""
    backend = get_backend()
    
    # Create first function
    val1 = backend.constant(10)
    val2 = backend.constant(5)
    result1 = backend.add(val1, val2)
    mlir1 = backend.create_function("func1", result1)
    
    # Reset and create second function
    backend.reset()
    val3 = backend.constant(20)
    val4 = backend.constant(10)
    result2 = backend.sub(val3, val4)
    mlir2 = backend.create_function("func2", result2)
    
    # Verify both functions are different
    assert "func1" in mlir1
    assert "func2" in mlir2
    assert "func1" not in mlir2
    assert "func2" not in mlir1


@pytest.mark.skipif(HAS_CPP_BACKEND, reason="Testing fallback when C++ backend unavailable")
def test_backend_fallback():
    """Test fallback behavior when C++ backend is not available"""
    backend = get_backend()
    assert backend is None


def test_cpp_backend_error_handling():
    """Test error handling when C++ backend is used incorrectly"""
    if not HAS_CPP_BACKEND:
        # Test that CppMLIRBuilder raises error when backend unavailable
        with pytest.raises(RuntimeError, match="C++ backend not available"):
            CppMLIRBuilder()
    else:
        # If backend is available, this should not raise
        backend = CppMLIRBuilder()
        assert backend is not None


if __name__ == "__main__":
    pytest.main([__file__])