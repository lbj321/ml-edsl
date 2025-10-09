"""Tests for conditional operations (comparisons and if-else)"""

import pytest
from mlir_edsl.backend import HAS_CPP_BACKEND, get_backend

# Skip all tests if C++ backend is not available
pytestmark = pytest.mark.skipif(not HAS_CPP_BACKEND, reason="C++ backend not available")


@pytest.fixture
def backend():
    """Provide a clean backend instance for each test"""
    backend_instance = get_backend()
    backend_instance.reset()
    return backend_instance


def test_cpp_comparison_operations(backend):
    """Test basic comparison operations"""
    # Test integer comparisons
    five = backend.constant(5)
    three = backend.constant(3)
    
    # Test greater than (5 > 3 = true)
    gt_result = backend.compare("sgt", five, three)
    assert gt_result.type == "i1"  # Boolean result
    
    # Test less than (5 < 3 = false) 
    lt_result = backend.compare("slt", five, three)
    assert lt_result.type == "i1"
    
    # Test equality (5 == 3 = false)
    eq_result = backend.compare("eq", five, three)
    assert eq_result.type == "i1"


def test_cpp_float_comparison(backend):
    """Test float comparison operations"""
    # Test float comparisons with promoted predicates
    pi = backend.constant(3.14)
    e = backend.constant(2.71)
    
    # Note: Float comparisons use "ordered" predicates
    gt_result = backend.compare("ogt", pi, e)  # 3.14 > 2.71 = true
    assert gt_result.type == "i1"
    
    eq_result = backend.compare("oeq", pi, pi)  # 3.14 == 3.14 = true
    assert eq_result.type == "i1"


def test_cpp_mixed_type_comparison(backend):
    """Test comparison with mixed types (int vs float)"""
    # Mixed types should be promoted for comparison
    int_val = backend.constant(5)      # i32
    float_val = backend.constant(5.0)  # f32
    
    # Should promote int to float, then use float comparison
    eq_result = backend.compare("oeq", int_val, float_val)
    assert eq_result.type == "i1"


def test_cpp_basic_if_else(backend):
    """Test basic if-else operation"""
    # Create: if (5 > 3) return 10 else return 20
    five = backend.constant(5)
    three = backend.constant(3)
    condition = backend.compare("sgt", five, three)  # true
    
    ten = backend.constant(10)
    twenty = backend.constant(20)
    
    if_result = backend.if_else(condition, ten, twenty)
    assert if_result.type == "i32"  # Should match then/else type


def test_cpp_conditional_mlir_generation(backend):
    """Test MLIR generation for conditional operations"""
    # Create: if (7 > 2) return 100 else return 200
    seven = backend.constant(7)
    two = backend.constant(2)
    condition = backend.compare("sgt", seven, two)
    
    hundred = backend.constant(100)
    two_hundred = backend.constant(200)
    result = backend.if_else(condition, hundred, two_hundred)
    
    backend.create_function("test_conditional", [], "i32")
    backend.finalize_function("test_conditional", result)
    mlir_code = backend.get_mlir_string()
    
    # Verify MLIR contains conditional constructs
    assert "arith.cmpi" in mlir_code or "arith.cmp" in mlir_code     # Comparison operation
    assert "scf.if" in mlir_code         # Conditional operation
    assert "scf.yield" in mlir_code      # Yield in regions
    assert "value = 100" in mlir_code    # Constant 100 (updated format)
    assert "value = 200" in mlir_code    # Constant 200 (updated format)
    
    print(f"\nGenerated Conditional MLIR:\n{mlir_code}")


def test_cpp_nested_conditional(backend):
    """Test nested expressions with conditionals"""
    # Create: if ((10 + 5) > 12) return (2 * 3) else return (20 - 5)
    ten = backend.constant(10)
    five = backend.constant(5)
    twelve = backend.constant(12)
    
    # Left side: (10 + 5) > 12
    sum_result = backend.add(ten, five)  # 15
    condition = backend.compare("sgt", sum_result, twelve)  # 15 > 12 = true
    
    # Then branch: 2 * 3
    two = backend.constant(2)
    three = backend.constant(3)
    then_value = backend.mul(two, three)  # 6
    
    # Else branch: 20 - 5  
    twenty = backend.constant(20)
    five2 = backend.constant(5)
    else_value = backend.sub(twenty, five2)  # 15
    
    result = backend.if_else(condition, then_value, else_value)
    backend.create_function("nested_conditional", [], "i32")
    backend.finalize_function("nested_conditional", result)
    mlir_code = backend.get_mlir_string()
    
    # Should contain multiple operations
    assert "arith.addi" in mlir_code     # Addition
    assert "arith.cmpi" in mlir_code     # Comparison
    assert "scf.if" in mlir_code         # Conditional
    assert "arith.muli" in mlir_code     # Multiplication in then
    assert "arith.subi" in mlir_code     # Subtraction in else
    
    print(f"\nGenerated Nested Conditional MLIR:\n{mlir_code}")


def test_cpp_float_conditional(backend):
    """Test conditional with float operations"""
    # Create: if (3.5 > 2.1) return 10.0 else return 20.0
    val1 = backend.constant(3.5)
    val2 = backend.constant(2.1)
    condition = backend.compare("ogt", val1, val2)  # float comparison
    
    then_val = backend.constant(10.0)
    else_val = backend.constant(20.0)
    
    result = backend.if_else(condition, then_val, else_val)
    assert result.type == "f32"
    
    backend.create_function("float_conditional", [], "f32")
    backend.finalize_function("float_conditional", result)
    mlir_code = backend.get_mlir_string()
    
    # Verify float operations
    assert "arith.cmpf" in mlir_code     # Float comparison
    assert "scf.if" in mlir_code
    assert "-> f32" in mlir_code         # Float return type
    
    print(f"\nGenerated Float Conditional MLIR:\n{mlir_code}")


def test_cpp_comparison_predicates(backend):
    """Test different comparison predicates"""
    a = backend.constant(10)
    b = backend.constant(5)
    
    # Test all integer predicates
    sgt = backend.compare("sgt", a, b)  # 10 > 5
    slt = backend.compare("slt", a, b)  # 10 < 5  
    eq = backend.compare("eq", a, a)    # 10 == 10
    ne = backend.compare("ne", a, b)    # 10 != 5
    sge = backend.compare("sge", a, b)  # 10 >= 5
    sle = backend.compare("sle", b, a)  # 5 <= 10
    
    # All should return i1 type
    for comp_result in [sgt, slt, eq, ne, sge, sle]:
        assert comp_result.type == "i1"


def test_cpp_conditional_llvm_ir(backend):
    """Test LLVM IR generation for conditionals"""
    # Use parameters instead of constants to prevent constant folding
    backend.create_function("conditional_fn", [("param1", "i32"), ("param2", "i32")], "i32")
    param1 = backend.get_parameter("param1")
    param2 = backend.get_parameter("param2")

    condition = backend.compare("sgt", param1, param2)
    result = backend.if_else(condition, backend.constant(42), backend.constant(24))

    backend.finalize_function("conditional_fn", result)
    llvm_ir = backend.get_llvm_ir_string()
    
    # Verify LLVM IR contains conditional constructs
    assert "define" in llvm_ir
    assert "conditional_fn" in llvm_ir
    assert "icmp" in llvm_ir  # Now there should be an actual comparison
    assert "br" in llvm_ir                        # Branch instruction
    assert "ret" in llvm_ir
    assert not llvm_ir.startswith("ERROR:")
    print(f"\nConditional LLVM IR:\n{llvm_ir}")


if __name__ == "__main__":
    pytest.main([__file__])