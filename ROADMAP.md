# ML-EDSL Development Roadmap

**Architecture**: Python frontend with C++ MLIR backend  
**Current Status**: Foundation Complete - Phase 6 (Control Flow) in progress  
**Vision**: Comprehensive ML compilation framework supporting modern AI architectures

---

## Foundation Complete ✅ (Phases 1-5)

**Implemented Architecture:**
- **Python Frontend**: AST with type system, operations (add/sub/mul/div), `@ml_function` decorator
- **C++ MLIR Backend**: Real MLIR IR generation with `MLIRBuilder` class  
- **LLVM Lowering**: Complete MLIR → LLVM IR transformation pipeline
- **JIT Execution**: Native-speed execution via `MLIRExecutor` with O0/O2/O3 optimization levels

**Current Capabilities:**
```python
@ml_function
def example():
    a = add(10, 5)      # 15
    b = mul(2, 3)       # 6  
    return sub(a, b)    # 9

result = example.execute()  # Native speed execution!
```

---

## Phase 6: Control Flow and Functions 🚧 CURRENT FOCUS

**Goal**: Add control flow constructs and function definitions to enable complex ML algorithms.

### Implementation Plan:

#### 1. **Conditional Operations**
```python
@ml_function
def conditional_example(x):
    if x > 0:
        return mul(x, 2)
    else:
        return sub(0, x)
```

**Technical Requirements:**
- `scf.if` dialect integration in MLIR backend
- Boolean type support and comparison operations (`arith.cmpi`, `arith.cmpf`)
- Conditional branch lowering to LLVM

#### 2. **Loop Constructs**
```python
@ml_function
def loop_example(n):
    result = 0
    for i in range(n):
        result = add(result, i)
    return result
```

**Technical Requirements:**
- `scf.for` and `scf.while` dialect integration
- Loop induction variable management
- Loop optimization passes (unrolling, vectorization)

#### 3. **Function Definitions with Parameters**
```python
@ml_function
def math_function(a: int, b: float):
    c = add(a, 5)
    d = mul(b, 2.0)
    return add(c, d)

result = math_function.execute(10, 3.5)  # Native execution with args
```

**Technical Requirements:**
- Function parameter parsing and type checking
- Local variable scope management
- Multi-parameter JIT execution support

#### 4. **Recursion Support**
```python
@ml_function
def factorial(n):
    if n <= 1:
        return 1
    else:
        return mul(n, factorial(sub(n, 1)))
```

**Technical Requirements:**
- Recursive function call lowering
- Tail call optimization
- Stack management for deep recursion

### Expected Deliverables:
- Extended AST nodes for control flow and functions
- MLIR SCF (Structured Control Flow) dialect integration
- Enhanced Python frontend with control flow syntax
- Comprehensive test suite for all control flow patterns
- Performance benchmarks vs equivalent Python code

---

## Future Development Phases 📋

### Phase 7: Memory & Arrays
- Array/tensor support with `memref`, `tensor`, `linalg` dialects
- Dynamic memory management and NumPy/PyTorch integration
- Multi-dimensional operations (slicing, broadcasting, reshaping)

### Phase 8: ML Operations  
- Linear algebra primitives (`matmul`, activation functions)
- Automatic differentiation with gradient computation
- Neural network building blocks and optimization

### Phase 9: Production & Performance
- Multi-threading, GPU/CUDA backend support
- Advanced optimization (PGO, vectorization, polyhedral)
- Profiling tools and debug capabilities

### Phase 10: Advanced AI Architectures
- Transformer primitives (attention, layer norm, FFN)
- Graph neural networks and diffusion model support  
- Integration with PyTorch/TensorFlow, HuggingFace Transformers
- Custom operator definition framework

---

## MLIR Syntax Reference

**Current MLIR Output Format:**
```mlir
module {
  func.func @add_fn() -> i32 {
    %c4_i32 = arith.constant 4 : i32
    %c6_i32 = arith.constant 6 : i32
    %0 = arith.addi %c4_i32, %c6_i32 : i32
    return %0 : i32
  }
}
```

**Target LLVM IR Output:**
```llvm
define i32 @add_fn() {
entry:
  ret i32 10
}
```

---

## Technical Status

### Current Stack:
- **Dependencies**: LLVM/MLIR 18+, CMake 3.20+, pybind11, C++17, Python 3.8+
- **MLIR Dialects**: `arith`, `func`, `builtin` | **Next**: `scf` (control flow)
- **Testing**: Comprehensive pytest suite with JIT integration tests
- **Examples**: Working demos in `examples/` directory

### Architecture Overview:
```
Python Frontend (AST) → C++ MLIR Backend → LLVM Lowering → JIT Execution
     @ml_function           MLIRBuilder        MLIRLowering    MLIRExecutor
```