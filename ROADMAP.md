# ML-EDSL Development Roadmap

**Architecture**: Python frontend with C++ MLIR backend
**Current Status**: Phase 9 - Performance
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

## Phase 7: Memory & Arrays ✅ Complete

**Goal**: Add array and tensor support to enable ML workloads, bridging from scalar operations to vectorized/tensor computations.

### Completed ✅

#### 1. Fixed-Size Array Support (memref dialect)
- `Array[N, dtype]` and `Array[M, N, dtype]` (1D/2D/3D)
- `memref.alloca`, `memref.load`, `memref.store`
- JAX-style `.at[i].set(v)` functional update semantics
- Element-wise arithmetic with broadcasting

#### 2. Multi-Dimensional Arrays (2D/3D)
- `memref<MxNxT>` with multi-index access `matrix[i, j]`
- Nested list initialization with shape inference

#### 3. Dynamic-Size Tensors (tensor dialect)
- `tensor<?xf32>` with runtime-determined shapes
- `tensor.empty`, `tensor.extract`, `tensor.insert`
- Bufferization pipeline: `tensor` → `memref` → LLVM

#### 4. Aggregate Types as Function Parameters
- `Array` and `Tensor` types accepted as `@ml_function` parameters
- Python lists packed as MLIR memref descriptors `(alloc_ptr, aligned_ptr, offset, sizes..., strides...)`
- Shape/ndim validation against declared `ArrayType`/`TensorType` at call time
- `Array`/`Tensor` added to `_get_type_hints` namespace for PEP 563 compatibility

### MLIR Dialects Integrated:
- **memref**: Stack-allocated, mutable buffers ✅
- **tensor**: Value-semantic, immutable tensors ✅
- **bufferization**: tensor → memref lowering ✅
- **linalg**: High-level linear algebra → Phase 8

---

## Phase 8: Linear Algebra (linalg dialect) ✅ Complete

**Goal**: Introduce `linalg` as the core abstraction for ML operations. This is the right
architectural level — `linalg` ops compose with tiling, vectorization, and GPU lowering
in later phases. Implementing matmul as explicit `scf.for` loops would be slower and
architecturally wrong.

### Completed ✅

#### 8.1 Array Return Types
- `Array[dtype, dims]` accepted as `@ml_function` return type
- C++ transforms aggregate-returning functions: appends hidden out-param, switches to void return
- Python allocates zeroed ctypes output buffer, passes as memref descriptor (same layout as input params)
- Result copied via `memref.copy` into Python-owned buffer; reconstructed as nested list via `_unflatten`
- Scalar return path unchanged

### Remaining 🚧

#### 8.2 linalg Dialect Integration
```python
@ml_function
def dot(a: Array[4, f32], b: Array[4, f32]) -> f32:
    return linalg.dot(a, b)  # Scalar return — no tensor return needed

@ml_function
def matmul(A: Tensor[4, 4, f32], B: Tensor[4, 4, f32]) -> Tensor[4, 4, f32]:
    return linalg.matmul(A, B)  # Requires 8.1
```

**Technical Requirements:**
- Register `linalg` dialect in `MLIRBuilder` and lowering pipeline
- `linalg.dot` (scalar return — first linalg op, no dependency on 8.1)
- `linalg.matmul`, `linalg.fill` (depend on tensor return from 8.1)
- Python frontend ops: `dot()`, `matmul()`
- Lowering: `linalg` → `loops` → `memref` → LLVM (existing bufferization handles the rest)

#### 8.3 Activation Functions
```python
@ml_function
def relu(x: Tensor[?, f32]) -> Tensor[?, f32]:
    return tensor_map(x, lambda v: If(v > 0.0, v, 0.0))
```

**Technical Requirements:**
- Element-wise tensor map operation (`linalg.generic` or explicit For loop)
- Common activations: relu, sigmoid, tanh (composable from existing arith ops)
- No new dialect needed — these are element-wise ops on tensors

#### 8.4 Reduction Operations
```python
@ml_function
def tensor_sum(x: Tensor[?, f32]) -> f32:
    return linalg.reduce(x, init=0.0, op=lambda a, b: a + b)

@ml_function
def tensor_max(x: Tensor[?, f32]) -> f32:
    return linalg.reduce(x, init=x[0], op=lambda a, b: If(a > b, a, b))

@ml_function
def mean(x: Tensor[?, f32], n: int) -> f32:
    return tensor_sum(x) / cast(n, f32)
```

**Technical Requirements:**
- `linalg.reduce` for general reductions (scalar output — no dependency on 8.1)
- Reduction ops: `sum`, `max`, `min` as built-in convenience wrappers
- `tensor.dim` / `memref.dim` for dynamic size queries (needed for `mean`)
- Note: reductions to scalar are independent of 8.1 (tensor returns); reductions
  to lower-rank tensors (e.g. row-wise sum) require 8.1

### MLIR Dialects:
- **linalg**: High-level linear algebra — current target ✅ (registered)
- **vector**: Vectorization → Phase 9

---

## Future Development Phases 📋

### Phase 8.5: I/O & NumPy Interoperability

**Goal**: Connect the JIT-compiled functions to real data. Now that Phase 8 provides useful
ML operations, this phase makes them actually usable with Python ecosystem data.

```python
import numpy as np

@ml_function
def process(A: Tensor[4, 4, f32], B: Tensor[4, 4, f32]) -> Tensor[4, 4, f32]:
    return matmul(A, B)

A = np.random.rand(4, 4).astype(np.float32)
B = np.random.rand(4, 4).astype(np.float32)
result = process(A, B)  # Returns np.ndarray
```

**Technical Requirements:**
- Python buffer protocol (PEP 3118) for zero-copy input passing
- numpy dtype → MLIR type mapping (`np.float32` → `f32`, `np.int32` → `i32`)
- Shape validation against declared parameter type
- Result reconstruction: raw pointer + shape → numpy array (with correct ownership/copy)

---

## Phase 9: Performance 🚧 CURRENT FOCUS

**Goal**: Make the compiler produce fast code on CPU. `linalg` is the key — it's designed
to be the target of tiling and vectorization passes. GPU is deferred; maintaining two
execution backends at this stage adds complexity without proportional value.

#### 9.1 Vectorization
- `linalg` → `vector` dialect transformation (MLIR's built-in vectorization pass)
- Auto-vectorization of matmul and element-wise ops (AVX2/AVX-512 on x86)

#### 9.2 Loop Tiling & Fusion
- Tiling matmul for cache efficiency (`linalg` tiling pass)
- Loop fusion for element-wise + matmul chains

#### 9.3 OpenMP Multithreading (Optional)
- `omp` dialect for shared-memory parallelism — annotates parallel `linalg` loops
- Integrates naturally with `linalg` parallel iterator types (no MPI/distributed needed)
- Requires linking against OpenMP runtime (`-fopenmp`)
- Pursue if vectorization alone leaves a meaningful gap vs numpy on multicore machines

#### 9.4 NumPy Benchmark Suite
- Compare MLIR-compiled ops against equivalent numpy operations
- **Data passing**: zero-copy via Python buffer protocol — numpy C-contiguous arrays match
  MLIR memref layout; enforce `np.ascontiguousarray()` at boundary for non-contiguous input
- **Scope**: element-wise ops (Python overhead favors MLIR), matmul (numpy/BLAS wins —
  goal is to understand the gap, not close it)
- Benchmark harness measures compute time only (pre-allocate arrays outside timing loop)

#### 9.5 GPU/CUDA Backend (Deferred)
- Deferred to a later phase when model-scale workloads justify the complexity
- `linalg` → `gpu` dialect → NVVM/CUBIN
- CUDA kernel launch via MLIR's GPU execution engine
- Memory transfer: host ↔ device

---

### Phase 10: Advanced Architectures

**Goal**: High-level ML building blocks and ecosystem integration.

#### 10.1 Transformer Primitives
- Multi-head attention (matmul + softmax, composable from Phase 8)
- Layer normalization (reduction + element-wise, composable from Phase 8)
- FFN layer (two matmuls + activation)

#### 10.2 Automatic Differentiation (Research Track)
- **Scope**: Forward-mode AD for scalar and simple tensor ops (achievable)
- **Deferred**: Reverse-mode AD is a research-level project; consider Enzyme-MLIR
  integration rather than implementing from scratch
- Gradient computation for basic ops: `arith`, `linalg.matmul`

#### 10.3 Ecosystem Integration
- PyTorch/TensorFlow model import (via torch-mlir or tf2mlir)
- HuggingFace Transformers: run inference on imported models
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
- **Dependencies**: LLVM/MLIR 18+, CMake 3.20+, pybind11, protobuf, C++17, Python 3.8+
- **MLIR Dialects (active)**: `arith`, `func`, `builtin`, `scf`, `cf`, `memref`, `tensor`, `bufferization`
- **Next dialect**: `linalg`
- **Testing**: pytest suite organized by feature area (`tests/core/`, `tests/memref/`, `tests/tensor/`)

### Current Operations:
- **Arithmetic**: `add`, `sub`, `mul`, `div`, `cast`
- **Comparison**: `lt`, `le`, `gt`, `ge`, `eq`, `ne`
- **Control Flow**: `If`, `For` (with `iter_args`), `While`
- **Functions**: `call` (recursion), `@ml_function` decorator
- **Arrays (memref)**: `Array[N, dtype]`, element access/store, element-wise arithmetic, JAX-style `.at[i].set(v)`
- **Tensors**: `Tensor.empty([size], dtype)`, `tensor.extract`, `tensor.insert`, dynamic dims

### Architecture Overview:
```
Python Frontend (AST) → Protobuf → C++ MLIR Backend → LLVM Lowering → JIT Execution
     @ml_function          bytes      MLIRBuilder         MLIRLowering    MLIRExecutor
      (strict types)                  (dialect builders)  (bufferize →    (O0/O2/O3)
                                                           memref → llvm)
```