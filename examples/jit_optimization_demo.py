#!/usr/bin/env python3
"""
JIT Optimization Examples - ML-EDSL Phase 5
============================================

This example demonstrates the JIT compilation and optimization capabilities
of the ML-EDSL, showing performance improvements from LLVM optimization passes.
"""

import time
from mlir_edsl import ml_function, add, sub, mul, div

def basic_jit_example():
    """Basic JIT execution example"""
    print("=== Basic JIT Execution ===")
    
    @ml_function  
    def simple_math():
        return add(10, 5)
    
    # JIT execution - compiles to native code
    result = simple_math.execute()
    print(f"JIT Result: {result}")
    
    # Regular call - prints MLIR IR
    print("\nGenerated MLIR:")
    simple_math()

def complex_expression_example():
    """More complex mathematical expression"""
    print("\n=== Complex Expression JIT ===")
    
    @ml_function
    def complex_calculation():
        # (20 + 10) * (8 - 3) / 5 = 30 * 5 / 5 = 30
        a = add(20, 10)     # 30
        b = sub(8, 3)       # 5  
        c = mul(a, b)       # 150
        return div(c, 5)    # 30
    
    result = complex_calculation.execute()
    print(f"Complex JIT Result: {result}")

def float_computation_example():
    """Floating-point computations with JIT"""
    print("\n=== Float Computation JIT ===")
    
    @ml_function
    def physics_calculation():
        # Kinetic energy: KE = 0.5 * m * v²
        # mass = 2.0, velocity = 10.0
        mass = 2.0
        velocity = 10.0
        half = 0.5
        
        v_squared = mul(velocity, velocity)  # v² = 100.0
        mv_squared = mul(mass, v_squared)    # m*v² = 200.0
        return mul(half, mv_squared)         # 0.5 * 200.0 = 100.0
    
    kinetic_energy = physics_calculation.execute()
    print(f"Kinetic Energy: {kinetic_energy} J")

def optimization_level_comparison():
    """Compare different optimization levels"""
    print("\n=== Optimization Level Comparison ===")
    
    @ml_function  
    def benchmark_expression():
        # Complex nested expression for benchmarking
        a = add(100, 50)      # 150
        b = sub(200, 75)      # 125  
        c = mul(a, b)         # 18750
        d = div(c, 25)        # 750
        e = add(d, 250)       # 1000
        return sub(e, 500)    # 500
    
    # Test different optimization levels using low-level API
    from mlir_edsl import _mlir_backend
    
    def benchmark_optimization_level(opt_level, iterations=10000):
        builder = _mlir_backend.MLIRBuilder()
        builder.initialize_module()
        
        # Build the expression manually
        c100 = builder.build_constant(100)
        c50 = builder.build_constant(50)
        c200 = builder.build_constant(200)
        c75 = builder.build_constant(75)
        c25 = builder.build_constant(25)
        c250 = builder.build_constant(250)
        c500 = builder.build_constant(500)
        
        a = builder.build_add(c100, c50)
        b = builder.build_sub(c200, c75)
        c = builder.build_mul(a, b)
        d = builder.build_div(c, c25)
        e = builder.build_add(d, c250)
        result = builder.build_sub(e, c500)
        
        builder.create_function_with_params_setup([])
        builder.finalize_function_with_params('bench_fn', result)
        llvm_ir = builder.get_llvm_ir_string()
        
        executor = _mlir_backend.MLIRExecutor()
        executor.initialize()
        executor.set_optimization_level(opt_level)
        
        func_ptr = executor.compile_function(llvm_ir, 'bench_fn')
        
        # Benchmark execution time
        start = time.perf_counter()
        for _ in range(iterations):
            res = executor.call_int32_function(func_ptr)
        end = time.perf_counter()
        
        return (end - start), res
    
    iterations = 50000
    print(f"Benchmarking {iterations} iterations...")
    
    o0_time, o0_result = benchmark_optimization_level(0, iterations)
    o2_time, o2_result = benchmark_optimization_level(2, iterations)  
    o3_time, o3_result = benchmark_optimization_level(3, iterations)
    
    print(f"O0 (no opt):  {o0_time:.4f}s → Result: {o0_result}")
    print(f"O2 (default): {o2_time:.4f}s → Result: {o2_result}")
    print(f"O3 (aggress): {o3_time:.4f}s → Result: {o3_result}")
    
    print(f"\nSpeedup vs O0:")
    print(f"O2: {o0_time/o2_time:.2f}x faster")
    print(f"O3: {o0_time/o3_time:.2f}x faster")

def constant_folding_demo():
    """Demonstrate compile-time constant folding"""
    print("\n=== Constant Folding Demo ===")
    
    @ml_function
    def constant_expression():
        # This should be completely folded at compile time
        return add(42, 8)  # Should become just "50"
    
    print("Expression: add(42, 8)")
    print("Expected optimization: Folded to constant 50 at compile time")
    
    # Show the MLIR generation
    constant_expression()
    
    # Execute with JIT
    result = constant_expression.execute()
    print(f"JIT Result: {result}")

def mixed_type_example():
    """Mixed integer and float operations"""
    print("\n=== Mixed Type Operations ===")
    
    @ml_function
    def mixed_calculation():
        # Temperature conversion: F = C * 9/5 + 32
        celsius = 25
        nine = 9.0
        five = 5.0  
        thirty_two = 32.0
        
        ratio = div(nine, five)        # 9.0 / 5.0 = 1.8
        scaled = mul(celsius, ratio)   # 25 * 1.8 = 45.0 (int promoted to float)
        return add(scaled, thirty_two) # 45.0 + 32.0 = 77.0
    
    fahrenheit = mixed_calculation.execute()
    print(f"25°C = {fahrenheit}°F")

def performance_showcase():
    """Showcase near-native performance"""
    print("\n=== Performance Showcase ===")
    
    def python_version():
        """Pure Python equivalent for comparison"""
        return ((100 + 50) * (200 - 75)) // 25 + 250 - 500
    
    @ml_function
    def jit_version():
        a = add(100, 50)
        b = sub(200, 75)  
        c = mul(a, b)
        d = div(c, 25)
        e = add(d, 250)
        return sub(e, 500)
    
    iterations = 100000
    
    # Benchmark Python
    start = time.perf_counter()
    for _ in range(iterations):
        py_result = python_version()
    py_time = time.perf_counter() - start
    
    # Benchmark JIT (compile once, execute many times)
    jit_result = jit_version.execute()  # First call compiles
    
    from mlir_edsl.backend import get_backend
    backend = get_backend()
    
    # Get the compiled function for repeated calls
    builder = backend
    c100 = builder.constant(100)
    c50 = backend.constant(50)  
    c200 = backend.constant(200)
    c75 = backend.constant(75)
    c25 = backend.constant(25)
    c250 = backend.constant(250) 
    c500 = backend.constant(500)
    
    a = backend.add(c100, c50)
    b = backend.sub(c200, c75)
    c = backend.mul(a, b)
    d = backend.div(c, c25) 
    e = backend.add(d, c250)
    final = backend.sub(e, c500)
    
    start = time.perf_counter()
    for _ in range(iterations):
        jit_res = backend.execute_function('perf_test', final)
    jit_time = time.perf_counter() - start
    
    print(f"Python ({iterations} iterations): {py_time:.4f}s → {py_result}")
    print(f"JIT ({iterations} iterations):    {jit_time:.4f}s → {jit_res}")
    print(f"JIT Speedup: {py_time/jit_time:.2f}x faster than Python!")

def main():
    """Run all examples"""
    print("ML-EDSL JIT Optimization Examples")
    print("=" * 50)
    
    basic_jit_example()
    complex_expression_example()
    float_computation_example() 
    constant_folding_demo()
    mixed_type_example()
    optimization_level_comparison()
    
    try:
        performance_showcase()
    except Exception as e:
        print(f"\nPerformance showcase skipped: {e}")

if __name__ == "__main__":
    main()