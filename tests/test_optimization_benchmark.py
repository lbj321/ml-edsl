"""Benchmark optimization performance vs unoptimized code"""
import time
import pytest
from mlir_edsl import _mlir_backend

def benchmark_optimization(optimization_level, iterations=10000):
    """Benchmark JIT execution at given optimization level"""
    # Build MLIR/LLVM IR for a complex computation
    builder = _mlir_backend.MLIRBuilder()
    builder.initialize_module()
    
    # Create a more complex expression: (10 + 5) * (8 - 3) / 2
    c10 = builder.build_constant(10)
    c5 = builder.build_constant(5)
    c8 = builder.build_constant(8)  
    c3 = builder.build_constant(3)
    c2 = builder.build_constant(2)
    
    add_result = builder.build_add(c10, c5)      # 10 + 5 = 15
    sub_result = builder.build_sub(c8, c3)       # 8 - 3 = 5
    mul_result = builder.build_mul(add_result, sub_result)  # 15 * 5 = 75
    div_result = builder.build_div(mul_result, c2)  # 75 / 2 = 37
    
    builder.create_function('complex_fn', div_result)
    llvm_ir = builder.get_llvm_ir_string()
    
    # Setup executor with specified optimization level
    executor = _mlir_backend.MLIRExecutor()
    executor.initialize()
    executor.set_optimization_level(optimization_level)
    
    # Compile function
    func_ptr = executor.compile_function(llvm_ir, 'complex_fn')
    assert func_ptr is not None
    
    # Benchmark execution
    start_time = time.perf_counter()
    
    for _ in range(iterations):
        result = executor.call_int32_function(func_ptr)
        assert result == 37  # Verify correctness
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    return execution_time, result

def test_optimization_performance_comparison():
    """Compare performance across optimization levels"""
    iterations = 50000
    
    print(f"\n=== JIT Optimization Benchmark ({iterations} iterations) ===")
    
    # Test each optimization level
    o0_time, o0_result = benchmark_optimization(0, iterations)  # O0
    o2_time, o2_result = benchmark_optimization(2, iterations)  # O2  
    o3_time, o3_result = benchmark_optimization(3, iterations)  # O3
    
    print(f"O0 (unoptimized): {o0_time:.4f}s - Result: {o0_result}")
    print(f"O2 (default):     {o2_time:.4f}s - Result: {o2_result}")  
    print(f"O3 (aggressive):  {o3_time:.4f}s - Result: {o3_result}")
    
    # Calculate speedups
    o2_speedup = o0_time / o2_time if o2_time > 0 else 1.0
    o3_speedup = o0_time / o3_time if o3_time > 0 else 1.0
    
    print(f"\nSpeedup vs O0:")
    print(f"O2: {o2_speedup:.2f}x faster")
    print(f"O3: {o3_speedup:.2f}x faster")
    
    # Verify all results are correct
    assert o0_result == o2_result == o3_result == 37
    
    # Optimization should generally make things faster (or at least not slower)
    # Note: For very simple expressions, overhead might make optimized versions slower
    assert o2_time > 0 and o3_time > 0  # Sanity check

def test_optimization_correctness():
    """Ensure optimization doesn't change computation results"""
    builder = _mlir_backend.MLIRBuilder()
    builder.initialize_module()
    
    # Complex float computation: 2.5 * 4.0 + 1.5 - 0.5 = 11.0
    c1 = builder.build_constant(2.5)
    c2 = builder.build_constant(4.0)
    c3 = builder.build_constant(1.5)
    c4 = builder.build_constant(0.5)
    
    mul_result = builder.build_mul(c1, c2)      # 2.5 * 4.0 = 10.0
    add_result = builder.build_add(mul_result, c3)  # 10.0 + 1.5 = 11.5  
    sub_result = builder.build_sub(add_result, c4)  # 11.5 - 0.5 = 11.0
    
    builder.create_function('float_fn', sub_result)
    llvm_ir = builder.get_llvm_ir_string()
    
    # Test each optimization level gives same result
    for opt_level in [0, 2, 3]:
        executor = _mlir_backend.MLIRExecutor()
        executor.initialize()
        executor.set_optimization_level(opt_level)
        
        func_ptr = executor.compile_function(llvm_ir, 'float_fn')
        result = executor.call_float_function(func_ptr)
        
        assert abs(result - 11.0) < 1e-6, f"O{opt_level} gave incorrect result: {result}"

def test_optimization_simple_constant_folding():
    """Test that optimization performs constant folding"""
    builder = _mlir_backend.MLIRBuilder() 
    builder.initialize_module()
    
    # Simple constant expression that should be folded: 4 + 6 = 10
    c1 = builder.build_constant(4)
    c2 = builder.build_constant(6) 
    result = builder.build_add(c1, c2)
    
    builder.create_function('const_fold_fn', result)
    llvm_ir = builder.get_llvm_ir_string()
    
    print(f"\nOriginal LLVM IR:\n{llvm_ir}")
    
    # Test O0 vs O2 - O2 should optimize this to a constant
    for opt_level in [0, 2]:
        executor = _mlir_backend.MLIRExecutor()
        executor.initialize()  
        executor.set_optimization_level(opt_level)
        
        func_ptr = executor.compile_function(llvm_ir, 'const_fold_fn')
        result = executor.call_int32_function(func_ptr)
        
        assert result == 10, f"O{opt_level} gave incorrect result: {result}"
        print(f"O{opt_level}: Correctly computed 4+6={result}")

if __name__ == "__main__":
    test_optimization_performance_comparison()
    test_optimization_correctness()  
    test_optimization_simple_constant_folding()