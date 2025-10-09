"""Tests for recursive function support in MLIR EDSL"""

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


def test_simple_recursive_factorial(backend):
    """Test basic recursive factorial function: fact(n) = n == 0 ? 1 : n * fact(n-1)"""
    # Create function with signature
    backend.create_function("factorial", [("n", "i32")], "i32")
    
    # Get function parameter
    n = backend.get_parameter("n")
    
    # Base case: n == 0
    zero = backend.constant(0)
    one = backend.constant(1)
    is_zero = backend.compare("eq", n, zero)
    
    # Recursive case: n * factorial(n-1)
    n_minus_one = backend.sub(n, one)
    recursive_call = backend.call_function("factorial", [n_minus_one])
    recursive_result = backend.mul(n, recursive_call)
    
    # Conditional: is_zero ? 1 : recursive_result
    result = backend.if_else(is_zero, one, recursive_result)
    
    backend.add_return(result)
    mlir_code = backend.get_mlir_string()
    
    # Verify MLIR contains expected elements
    assert '"func.func"' in mlir_code and "factorial" in mlir_code
    assert '"func.call"' in mlir_code and "@factorial" in mlir_code
    assert '"scf.if"' in mlir_code
    assert '"func.return"' in mlir_code


def test_fibonacci_recursion(backend):
    """Test fibonacci function: fib(n) = n <= 1 ? n : fib(n-1) + fib(n-2)"""
    # Create function with signature
    backend.create_function("fibonacci", [("n", "i32")], "i32")
    
    n = backend.get_parameter("n")
    zero = backend.constant(0)
    one = backend.constant(1)
    two = backend.constant(2)
    
    # Base cases: n <= 1 (simplified: just n == 0 or n == 1)
    is_zero = backend.compare("eq", n, zero)
    is_one = backend.compare("eq", n, one)
    
    # For base cases, return n itself
    # For recursive case: fib(n-1) + fib(n-2)
    n_minus_one = backend.sub(n, one)
    n_minus_two = backend.sub(n, two)
    fib_n_1 = backend.call_function("fibonacci", [n_minus_one])
    fib_n_2 = backend.call_function("fibonacci", [n_minus_two])
    recursive_result = backend.add(fib_n_1, fib_n_2)
    
    # Nested conditionals: is_zero ? 0 : (is_one ? 1 : recursive_result)
    inner_result = backend.if_else(is_one, one, recursive_result)
    result = backend.if_else(is_zero, zero, inner_result)
    
    backend.add_return(result)
    mlir_code = backend.get_mlir_string()
    
    # Verify MLIR contains expected elements
    assert '"func.func"' in mlir_code and "fibonacci" in mlir_code
    assert '"func.call"' in mlir_code and "@fibonacci" in mlir_code
    assert '"scf.if"' in mlir_code
    assert '"func.return"' in mlir_code


def test_countdown_recursion(backend):
    """Test simple countdown: countdown(n) = n <= 0 ? 0 : countdown(n-1)"""
    # Create function with signature
    backend.create_function("countdown", [("n", "i32")], "i32")
    
    n = backend.get_parameter("n")
    zero = backend.constant(0)
    one = backend.constant(1)
    
    # Base case: n <= 0 (using n == 0 for simplicity)
    is_done = backend.compare("sle", n, zero)  # signed less or equal
    n_minus_one = backend.sub(n, one)
    recursive_call = backend.call_function("countdown", [n_minus_one])
    
    result = backend.if_else(is_done, zero, recursive_call)
    backend.add_return(result)
    mlir_code = backend.get_mlir_string()
    
    # Verify MLIR contains expected elements
    assert '"func.func"' in mlir_code and "countdown" in mlir_code
    assert '"func.call"' in mlir_code and "@countdown" in mlir_code
    assert '"scf.if"' in mlir_code
    assert '"func.return"' in mlir_code


def test_tail_recursive_sum(backend):
    """Test tail recursive sum with accumulator: sum(n, acc) = n == 0 ? acc : sum(n-1, acc+n)"""
    # Create function with signature (two parameters)
    backend.create_function("tail_sum", [("n", "i32"), ("acc", "i32")], "i32")
    
    n = backend.get_parameter("n")
    acc = backend.get_parameter("acc")
    zero = backend.constant(0)
    one = backend.constant(1)
    
    # Base case: n == 0
    is_done = backend.compare("eq", n, zero)
    n_minus_one = backend.sub(n, one)
    new_acc = backend.add(acc, n)
    recursive_call = backend.call_function("tail_sum", [n_minus_one, new_acc])
    
    result = backend.if_else(is_done, acc, recursive_call)
    backend.add_return(result)
    mlir_code = backend.get_mlir_string()
    
    # Verify MLIR contains expected elements
    assert '"func.func"' in mlir_code and "tail_sum" in mlir_code
    assert '"func.call"' in mlir_code and "@tail_sum" in mlir_code
    assert '"scf.if"' in mlir_code
    assert '"func.return"' in mlir_code


def test_mutual_recursion_even_odd(backend):
    """Test mutually recursive functions: is_even/is_odd"""
    # More complex case: functions calling each other
    # is_even(n) = n == 0 ? true : is_odd(n-1)
    # is_odd(n) = n == 0 ? false : is_even(n-1)
    
    # This tests:
    # 1. Multiple function definitions
    # 2. Forward references
    # 3. Cross-function calls
    
    # Expected pattern for is_even:
    # backend.start_function("is_even", [("n", "i32")], "i1")  # i1 for boolean
    # n = backend.get_param("n")
    # zero = backend.constant(0)
    # one = backend.constant(1)
    # true_val = backend.constant_bool(True)
    # false_val = backend.constant_bool(False)
    # 
    # is_zero = backend.compare_eq(n, zero)
    # n_minus_one = backend.sub(n, one)
    # odd_call = backend.call_function("is_odd", [n_minus_one])
    # 
    # result = backend.if_else(is_zero, true_val, odd_call)
    # backend.finish_function(result)
    
    # Similar pattern for is_odd...
    
    pytest.skip("Mutual recursion not yet implemented")


def test_recursive_factorial_mlir_output(backend):
    """Test that recursive factorial generates correct MLIR structure"""
    # Create a simple factorial function
    backend.create_function("fact", [("n", "i32")], "i32")
    
    n = backend.get_parameter("n")
    zero = backend.constant(0)
    one = backend.constant(1)
    
    is_zero = backend.compare("eq", n, zero)
    n_minus_one = backend.sub(n, one)
    recursive_call = backend.call_function("fact", [n_minus_one])
    recursive_result = backend.mul(n, recursive_call)
    
    result = backend.if_else(is_zero, one, recursive_result)
    backend.add_return(result)
    
    mlir_code = backend.get_mlir_string()
    
    # Print MLIR for debugging
    print("Generated MLIR:")
    print(mlir_code)
    
    # Detailed MLIR structure verification
    assert "function_type = (i32) -> i32" in mlir_code  # Function signature
    assert 'sym_name = "fact"' in mlir_code  # Function name
    assert "%arg0: i32" in mlir_code  # Parameter
    assert '"arith.cmpi"' in mlir_code  # Comparison operation
    # Look for function call in any format
    assert ("func.call" in mlir_code and "fact" in mlir_code) or ("@fact" in mlir_code)  # Recursive call
    assert '"arith.muli"' in mlir_code  # Multiplication
    assert '"scf.if"' in mlir_code  # Conditional
    assert '"func.return"' in mlir_code  # Return statement


def test_recursive_function_symbol_resolution(backend):
    """Test that recursive functions can reference themselves during definition"""
    # This tests the core symbol table challenge:
    # Function must be callable from within its own definition
    
    # Create function - it should be immediately available for self-calls
    backend.create_function("self_test", [("x", "i32")], "i32")
    
    x = backend.get_parameter("x")
    one = backend.constant(1)
    
    # The fact that we can call the function being defined proves symbol resolution works
    self_call = backend.call_function("self_test", [one])  # This should not fail
    result = backend.add(x, self_call)  # Simple operation to avoid infinite recursion in test
    
    backend.add_return(result)
    mlir_code = backend.get_mlir_string()
    
    # Print MLIR for debugging
    print("Generated MLIR:")
    print(mlir_code)
    
    # Verify the function can call itself (look for any function call format)
    assert ("func.call" in mlir_code and "self_test" in mlir_code) or ("@self_test" in mlir_code)


def test_recursive_llvm_ir_generation(backend):
    """Test LLVM IR generation from recursive MLIR functions"""
    # Verify that recursive MLIR lowers correctly to LLVM IR
    # Should contain proper function calls and stack management
    
    pytest.skip("LLVM lowering for recursion not yet implemented")


def test_deep_recursion_stack_safety(backend):
    """Test that deep recursion doesn't break compilation"""
    # Not testing runtime stack overflow, just compilation correctness
    # For very deep call chains in generated IR
    
    pytest.skip("Deep recursion compilation not yet implemented")


if __name__ == "__main__":
    pytest.main([__file__])