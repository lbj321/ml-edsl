# ML-EDSL Examples

This directory contains examples demonstrating the capabilities of the ML-EDSL with JIT compilation.

## Quick Start

### Simple Examples
```bash
python3 examples/simple_jit_examples.py
```

Basic examples showing:
- Simple arithmetic with JIT execution
- Float operations  
- Complex expressions
- MLIR code generation

### JIT Optimization Demo
```bash
python3 examples/jit_optimization_demo.py
```

Advanced examples showcasing:
- Optimization level comparison (O0 vs O2 vs O3)
- Performance benchmarking
- Constant folding demonstration
- Mixed-type operations
- Performance vs pure Python

## Example Output

### Simple JIT Execution
```python
@ml_function
def simple_add():
    return add(5, 3)

result = simple_add.execute()  # Returns 8 at native speed
```

### MLIR Generation
```python
simple_add()  # Prints generated MLIR:
```
```mlir
module {
  func.func @simple_add() -> i32 {
    %c5_i32 = arith.constant 5 : i32
    %c3_i32 = arith.constant 3 : i32  
    %0 = arith.addi %c5_i32, %c3_i32 : i32
    return %0 : i32
  }
}
```

### Optimization Levels
```python
# Control optimization level for performance
executor.set_optimization_level(0)  # O0 - No optimization
executor.set_optimization_level(2)  # O2 - Default (balanced)  
executor.set_optimization_level(3)  # O3 - Aggressive optimization
```

### Performance Results
Typical performance improvements with optimization:
- **O2**: 1.05x - 1.2x faster than O0
- **O3**: 1.07x - 1.5x faster than O0  
- **JIT vs Python**: 2x - 10x faster for mathematical computations

## Features Demonstrated

### Phase 5 Capabilities ✅
- [x] **JIT Compilation**: MLIR → LLVM → Native execution
- [x] **LLVM Optimization**: mem2reg, instcombine, simplifycfg passes
- [x] **Optimization Levels**: O0/O2/O3 control
- [x] **Type Safety**: Automatic int/float handling
- [x] **Performance**: Near-native execution speed
- [x] **Constant Folding**: Compile-time optimization

### Current Operations
- `add(a, b)` - Addition
- `sub(a, b)` - Subtraction  
- `mul(a, b)` - Multiplication
- `div(a, b)` - Division
- Mixed int/float operations with automatic type promotion

### API Usage

#### High-Level API (Recommended)
```python
@ml_function
def my_computation():
    return add(mul(5, 3), 2)  # (5 * 3) + 2 = 17

result = my_computation.execute()  # JIT execution
```

#### Low-Level API (Advanced)
```python  
from mlir_edsl import _mlir_backend

builder = _mlir_backend.MLIRBuilder()
executor = _mlir_backend.MLIRExecutor()

# Build expression
c1 = builder.build_constant(5)
c2 = builder.build_constant(3)
result = builder.build_add(c1, c2)

# JIT compile and execute
builder.create_function("test", result)
llvm_ir = builder.get_llvm_ir_string()
func_ptr = executor.compile_function(llvm_ir, "test")
value = executor.call_int32_function(func_ptr)
```

## Next Steps: Phase 6

The examples show Phase 5 is complete. Ready for Phase 6 advanced features:
- Control flow (if/else, loops)
- Function calls and recursion
- Memory operations and arrays
- Tensor operations for ML workloads

## Performance Tips

1. **Use JIT for compute-heavy functions** - Best speedup for mathematical computations
2. **Enable O2/O3 optimization** - Usually provides measurable performance gains  
3. **Compile once, execute many times** - Amortize compilation cost
4. **Profile your code** - Use the benchmarking examples as templates