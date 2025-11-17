# ML-EDSL Development Roadmap

**Architecture**: Python frontend with C++ MLIR backend
**Current Status**: Phase 7 (Memory & Arrays) - Tensor support in progress
**Vision**: Comprehensive ML compilation framework supporting modern AI architectures

---

## Foundation Complete ✅ (Phases 1-6)

**Implemented Architecture:**
- **Python Frontend**: AST with strict type system, `@ml_function` decorator
- **C++ MLIR Backend**: Real MLIR IR generation with `MLIRBuilder` class
- **LLVM Lowering**: Complete MLIR → LLVM IR transformation pipeline
- **JIT Execution**: Native-speed execution via `MLIRExecutor` with O0/O2/O3 optimization levels
- **Strict Typing**: Required type hints, explicit cast() for conversions, no auto-promotion
- **Control Flow**: If/For/While constructs with comparison operators
- **Recursion**: Self-referencing functions via call() with symbol resolution

**Current Type System:**
- **MLIR Types**: `i32`, `f32`, `i1` (boolean)
- **Python Aliases**: `int` → `i32`, `float` → `f32`, `bool` → `i1`
- **Strict Enforcement**: No implicit conversions, explicit `cast()` required

**Current Capabilities:**
```python
@ml_function
def factorial(n: int) -> int:
    return If(n <= 1, 1, n * factorial(n - 1))

@ml_function
def sum_range(start: int, end: int) -> int:
    return For(start=start, end=end, init=0, operation="add")

@ml_function
def mixed_types(x: int, y: float) -> float:
    return cast(x, f32) + y  # Explicit cast required

result = factorial(5)     # 120 - Native speed!
result = sum_range(1, 10) # 55 - Native speed!
```

---

## Phase 7: Memory & Arrays 🚧 CURRENT FOCUS

**Goal**: Add array and tensor support to enable ML workloads, bridging from scalar operations to vectorized/tensor computations.

### Implementation Plan:

#### 1. **Fixed-Size Array Support (memref dialect)**
```python
@ml_function
def array_operations() -> tensor[4, i32]:
    # Create a fixed-size array [1, 2, 3, 4]
    arr = array([1, 2, 3, 4], dtype=i32)

    # Element access and modification
    arr[0] = 10
    value = arr[2]

    # Array arithmetic (element-wise)
    doubled = arr * 2
    return doubled
```

**Technical Requirements:**
- `memref` dialect for stack-allocated arrays (fixed size)
- Array type: `memref<NxT>` (e.g., `memref<4xi32>`)
- Operations: `memref.alloc`, `memref.load`, `memref.store`
- Index operations: `arith.index_cast` for array indexing
- Element-wise operations support

#### 2. **Multi-Dimensional Arrays (2D/3D)**
```python
@ml_function
def matrix_operations() -> tensor[2, 3, i32]:
    # 2D matrix [2 rows x 3 cols]
    matrix = array([[1, 2, 3],
                    [4, 5, 6]], dtype=i32)

    # Access element at [row, col]
    value = matrix[1, 2]  # 6

    # Row/column slicing
    row = matrix[0, :]    # [1, 2, 3]
    col = matrix[:, 1]    # [2, 5]

    return matrix
```

**Technical Requirements:**
- Multi-dimensional `memref` types: `memref<MxNxT>`
- Strided memory layout for slicing
- Affine expressions for indexing (`affine` dialect)
- Memory layout transformations

#### 3. **Dynamic-Size Tensors (tensor dialect)**
```python
@ml_function
def dynamic_tensor(size: int) -> tensor[?, f32]:
    # Create tensor with runtime-determined size
    data = tensor.empty([size], dtype=f32)

    # Fill with values using loop
    for i in range(size):
        data[i] = cast(i, f32) * 2.0

    return data
```

**Technical Requirements:**
- `tensor` dialect for value-semantic tensors
- Dynamic shapes: `tensor<?xf32>`
- Tensor operations: `tensor.empty`, `tensor.extract`, `tensor.insert`
- Buffer allocation and deallocation
- Integration with loops for tensor initialization

#### 4. **NumPy/PyTorch Interoperability**
```python
import numpy as np

@ml_function
def process_numpy(arr: tensor[?, f32]) -> tensor[?, f32]:
    # Process elements: arr[i] * 2 + 1
    result = arr * 2.0 + 1.0
    return result

# Usage with NumPy
np_array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
result = process_numpy(np_array)  # Returns NumPy array
```

**Technical Requirements:**
- Python buffer protocol integration (PEP 3118)
- Zero-copy data passing between Python and MLIR
- Type mapping: `numpy.float32` → `f32`, `numpy.int32` → `i32`
- Shape inference and validation
- Conversion: NumPy → `memref`/`tensor` → computation → NumPy

#### 5. **Array Reduction Operations**
```python
@ml_function
def array_sum(arr: tensor[?, i32]) -> i32:
    # Sum all elements in array
    result = 0
    for i in range(len(arr)):
        result = result + arr[i]
    return result

@ml_function
def array_max(arr: tensor[?, i32]) -> i32:
    # Find maximum element
    max_val = arr[0]
    for i in range(1, len(arr)):
        max_val = If(arr[i] > max_val, arr[i], max_val)
    return max_val
```

**Technical Requirements:**
- Array length/shape queries: `tensor.dim`, `memref.dim`
- Reduction patterns with loops
- Future: `linalg.reduce` for optimized reductions

### Expected Deliverables:
- Array/tensor type system with fixed and dynamic sizes
- `memref` dialect integration for stack-allocated arrays
- `tensor` dialect integration for value-semantic tensors
- Element access, slicing, and modification operations
- NumPy interoperability with zero-copy data passing
- Comprehensive test suite for array operations
- Documentation and examples for array usage

### MLIR Dialects to Integrate:
- **memref**: Stack-allocated, mutable buffers (fixed size)
- **tensor**: Value-semantic, immutable tensors (dynamic size)
- **affine**: Affine expressions for loop bounds and indexing
- **linalg** (future): High-level linear algebra operations

---

## Future Development Phases 📋

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

**Scalar Operations:**
```mlir
module {
  func.func @add_fn(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
```

**Control Flow (If/Else):**
```mlir
module {
  func.func @conditional(%arg0: i32, %arg1: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %cmp = arith.cmpi sgt, %arg0, %c0 : i32
    %result = scf.if %cmp -> i32 {
      %doubled = arith.muli %arg0, %c2_i32 : i32
      scf.yield %doubled : i32
    } else {
      scf.yield %arg1 : i32
    }
    return %result : i32
  }
}
```

**Loops (scf.for):**
```mlir
module {
  func.func @sum_range(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %result = scf.for %iv = %c0 to %arg0 step %c1 iter_args(%acc = %c0) -> i32 {
      %next = arith.addi %acc, %iv : i32
      scf.yield %next : i32
    }
    return %result : i32
  }
}
```

**Recursion:**
```mlir
module {
  func.func @factorial(%arg0: i32) -> i32 {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cmp = arith.cmpi sle, %arg0, %c1 : i32
    %result = scf.if %cmp -> i32 {
      scf.yield %c1 : i32
    } else {
      %n_minus_1 = arith.subi %arg0, %c1 : i32
      %rec_result = func.call @factorial(%n_minus_1) : (i32) -> i32
      %product = arith.muli %arg0, %rec_result : i32
      scf.yield %product : i32
    }
    return %result : i32
  }
}
```

**Type Casting:**
```mlir
module {
  func.func @cast_example(%arg0: i32, %arg1: f32) -> f32 {
    %arg0_float = arith.sitofp %arg0 : i32 to f32
    %result = arith.addf %arg0_float, %arg1 : f32
    return %result : f32
  }
}
```

---

## Technical Status

### Current Stack:
- **Dependencies**: LLVM/MLIR 18+, CMake 3.20+, pybind11, C++17, Python 3.8+
- **MLIR Dialects**: `arith`, `func`, `builtin`, `scf`, `cf` | **Next**: `memref`, `tensor`, `affine`
- **Testing**: Comprehensive pytest suite with JIT integration tests
  - `test_strict_typing.py` - Type system validation
  - `test_parameters.py` - Function parameter tests
  - `test_conditionals.py` - If/comparison operations
  - `test_loops.py` - For/While loop tests
  - `test_recursion.py` - Recursive function tests
  - `test_cpp_backend.py` - C++ backend integration
- **Examples**: Working demos in `examples/` directory

### Current Operations:
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `cast`
- **Comparison**: `lt`, `le`, `gt`, `ge`, `eq`, `ne`
- **Control Flow**: `If`, `For`, `While`
- **Functions**: `call` (recursion), `@ml_function` decorator

### Architecture Overview:
```
Python Frontend (AST) → C++ MLIR Backend → LLVM Lowering → JIT Execution
     @ml_function           MLIRBuilder        MLIRLowering    MLIRExecutor
      (strict types)        (arith/scf/cf)      (llvm dialect)  (O0/O2/O3)
```