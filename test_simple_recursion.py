#!/usr/bin/env python3
"""Test simple recursion: countdown function"""

from mlir_edsl.functions import ml_function
from mlir_edsl.ops import sub, call
from mlir_edsl.ast import Parameter, Constant, CompareOp, IfOp

@ml_function
def countdown(n: Parameter) -> int:
    # Base case: if n <= 0, return 0
    condition = CompareOp("sle", n, Constant(0))  # n <= 0
    base_case = Constant(0)

    # Recursive case: countdown(n-1)
    n_minus_one = sub(n, Constant(1))
    recursive_result = call("countdown", [n_minus_one])

    return IfOp(condition, base_case, recursive_result)

if __name__ == "__main__":
    print("Testing simple recursion (countdown)...")

    try:
        # Test small values first
        result = countdown(2)  # Should recurse: 2 -> 1 -> 0, return 0
        print(f"countdown(2) = {result}")
        print("SUCCESS: Simple recursion worked!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()