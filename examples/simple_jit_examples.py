#!/usr/bin/env python3
"""
Simple JIT Examples - Getting Started
=====================================

Quick examples showing basic JIT compilation and execution.
"""

from mlir_edsl import ml_function, add, sub, mul, div

# Example 1: Basic addition with JIT
@ml_function
def simple_add():
    return add(5, 3)

print("=== Example 1: Basic Addition ===")
print("Expression: add(5, 3)")
result = simple_add.execute()  # JIT compile and execute
print(f"JIT Result: {result}")  # Should print 8

# Example 2: Float multiplication  
@ml_function
def float_multiply():
    return mul(2.5, 4.0)

print("\n=== Example 2: Float Multiplication ===") 
print("Expression: mul(2.5, 4.0)")
result = float_multiply.execute()
print(f"JIT Result: {result}")  # Should print 10.0

# Example 3: Complex expression
@ml_function  
def complex_math():
    # Calculate: (10 + 5) * (8 - 2) / 3 = 15 * 6 / 3 = 30
    sum_part = add(10, 5)      # 15
    diff_part = sub(8, 2)      # 6
    product = mul(sum_part, diff_part)  # 90
    return div(product, 3)     # 30

print("\n=== Example 3: Complex Expression ===")
print("Expression: (10 + 5) * (8 - 2) / 3")
result = complex_math.execute()
print(f"JIT Result: {result}")  # Should print 30

# Example 4: Show MLIR generation
@ml_function
def show_mlir():
    return sub(20, 8)

print("\n=== Example 4: MLIR Generation ===")
print("Regular call shows generated MLIR:")
show_mlir()  # This prints the MLIR code

print("\nJIT execution:")
result = show_mlir.execute()  # This executes at native speed
print(f"Result: {result}")  # Should print 12

print("\n=== All Examples Complete! ===")
print("Your ML-EDSL is working with JIT compilation!")