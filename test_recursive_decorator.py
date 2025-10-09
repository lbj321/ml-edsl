#!/usr/bin/env python3
"""Test script for recursive functions with decorator"""

from mlir_edsl.functions import ml_function
from mlir_edsl.ops import add, sub, mul, call
from mlir_edsl.ast import Parameter, Constant, CompareOp, IfOp

@ml_function
def recursive_factorial(n: Parameter) -> int:
    # Base case: if n <= 1, return 1
    condition = CompareOp("sle", n, Constant(1))  # n <= 1 (signed less-equal)
    base_case = Constant(1)

    # Recursive case: n * factorial(n-1)
    n_minus_one = sub(n, Constant(1))
    recursive_result = call("recursive_factorial", [n_minus_one])
    recursive_case = mul(n, recursive_result)

    # Choose between base case and recursive case
    return IfOp(condition, base_case, recursive_case)

if __name__ == "__main__":
    print("Testing recursive factorial with decorator...")
    
    try:
        # Test the recursive function
        print("Calling recursive_factorial...")
        
        # Create a parameter and call the function
        param = Parameter("n", 5)
        print(f"Parameter type: {param.type}")
        result = recursive_factorial(5)  # factorial(5) - pass the actual value, not the Parameter object
        print(f"Factorial result: {result}")
        print("SUCCESS: Recursive function call worked!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Let's also check what MLIR was generated
        try:
            from mlir_edsl.backend import get_backend
            backend = get_backend()
            mlir_code = backend.get_mlir_string()
            print("Generated MLIR:")
            print(mlir_code)
            print("\n" + "="*50)
            print("DIAGNOSIS: The function signature says 1 parameter but block has 2!")
            print("This suggests an issue in C++ createFunction() or parameter setup")
            print("="*50)
        except:
            print("Could not get MLIR output")