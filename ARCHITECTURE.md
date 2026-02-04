# Architecture

This project is an MLIR-based Embedded Domain-Specific Language (EDSL) for Machine Learning. Users write functions in Python using a restricted set of operations, and the system compiles them through MLIR down to native machine code via LLVM JIT. The Python frontend builds an AST through symbolic execution, serializes it to protobuf, and sends it across a pybind11 boundary to a C++ backend that generates MLIR IR, lowers it to LLVM IR, and JIT-compiles it for execution.

## Compilation Pipeline

```mermaid
flowchart TD
    A["@ml_function decorator"] --> B["Symbolic execution"]
    B --> C["Python AST"]
    C --> D["Protobuf serialization"]
    D -->|pybind11 boundary| E["C++ MLIRBuilder"]
    E --> F["MLIR IR"]
    F --> G["MLIR lowering passes"]
    G --> H["LLVM IR"]
    H --> I["LLVM ORC JIT"]
    I --> J["Native machine code"]
    J -->|function pointer via ctypes| K["Python result"]
```

The pipeline has two phases. The **definition phase** happens at decoration time: the decorator runs the function body with symbolic `Parameter` objects to build an AST and validate types. The **execution phase** happens on first call: the AST is serialized, compiled through MLIR/LLVM, and JIT-compiled. Subsequent calls reuse the compiled native code.

## Directory Structure

```
mlir_edsl/                          # Python frontend
    __init__.py                     # Public API exports
    types.py                        # Type system (ScalarType, ArrayType)
    backend.py                      # C++ backend wrapper, ctypes execution
    ast_pb2.py                      # Generated protobuf Python code
    ast/                            # AST node implementations
        base.py                     # Value base class
        operators.py                # Operator overloads (__add__, __le__, etc.)
        serialization.py            # Protobuf serialization context (SSA reuse)
        helpers.py                  # JAX-style .at[] array indexing
        nodes/
            scalars.py              # Constant, BinaryOp, CompareOp, CastOp
            arrays.py               # ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp
            control_flow.py         # IfOp, ForLoopOp
            functions.py            # Parameter, CallOp
    ops/                            # User-facing operation builders
        arithmetic.py               # add, sub, mul, div
        comparison.py               # lt, le, gt, ge, eq, ne
        control_flow.py             # If, For
        conversion.py               # cast, call
    functions/                      # Function decoration and compilation
        decorator.py                # @ml_function → MLFunction wrapper
        signature.py                # Type hint extraction
        validation.py               # Symbolic execution and type checking
        compilation.py              # AST → backend compilation
        context.py                  # Symbolic execution context manager

cpp/                                # C++ backend
    schemas/
        ast.proto                   # Protobuf schema (single source of truth for types)
    include/mlir_edsl/
        MLIRBuilder.h               # AST → MLIR IR generation
        ArithBuilder.h              # arith dialect operations
        SCFBuilder.h                # scf dialect (if, for)
        MemRefBuilder.h             # memref dialect (arrays)
        MLIRLowering.h              # MLIR → LLVM IR lowering
        MLIRExecutor.h              # LLVM ORC JIT engine
    src/
        MLIRBuilder.cpp             # Core IR generation
        MLIRExecutor.cpp            # JIT compilation and execution
        MLIRLowering.cpp            # Lowering passes
        python_bindings.cpp         # pybind11 glue
        builders/
            ArithBuilder.cpp        # arith.addi, arith.muli, arith.sitofp, etc.
            SCFBuilder.cpp          # scf.if, scf.for
            MemRefBuilder.cpp       # memref.alloca, memref.load, memref.store

tests/                              # pytest suite
```

## Walkthrough: Factorial

Here is a concrete example traced through every stage.

### 1. User code

```python
from mlir_edsl import ml_function, mul, sub, eq, If, call, i32

@ml_function
def factorial(n: int) -> int:
    return If(eq(n, 0),
              1,
              mul(n, call("factorial", [sub(n, 1)], i32)))

result = factorial(5)  # returns 120
```

### 2. Decoration: symbolic execution

When `@ml_function` is applied, `MLFunction.__init__` runs the function body with a symbolic `Parameter("n", i32)` object instead of a real integer. Every operation (`eq`, `sub`, `mul`, `If`, `call`) detects it is inside a `symbolic_execution()` context and returns an AST node instead of computing a value. The result is a tree:

```
IfOp
├── condition: CompareOp(EQ, Parameter("n"), Constant(0))
├── then: Constant(1)
└── else: BinaryOp(MUL,
               Parameter("n"),
               CallOp("factorial", [BinaryOp(SUB, Parameter("n"), Constant(1))]))
```

Type inference runs on this tree to verify the return type matches the declared `-> int`.

### 3. First call: protobuf serialization

On the first call to `factorial(5)`, the cached AST is serialized to protobuf via `to_proto_with_reuse()`. The `SerializationContext` detects that `Parameter("n")` appears multiple times in the tree and emits a `LetBinding`/`ValueReference` pair so the C++ side can generate proper SSA form. The serialized `FunctionDef` protobuf is sent across the pybind11 boundary as raw bytes.

### 4. C++ MLIR generation

`MLIRBuilder::compileFunctionFromDef` parses the protobuf and dispatches each node to a category handler:

| AST category | Handler | MLIR dialect |
|---|---|---|
| `ScalarNode` | `buildFromScalarNode()` | `arith` (constants, binary ops, casts) |
| `ArrayNode` | `buildFromArrayNode()` | `memref` (alloca, load, store) |
| `ControlFlowNode` | `buildFromControlFlowNode()` | `scf` (if, for) |
| `FunctionNode` | `buildFromFunctionNode()` | `func` (parameters, calls) |
| `BindingNode` | `buildFromBindingNode()` | SSA value reuse (let/ref) |

Each handler delegates to a specialized builder (`ArithBuilder`, `SCFBuilder`, `MemRefBuilder`) that creates MLIR operations. The factorial example produces IR like:

```mlir
module {
  func.func @factorial(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32
    %1 = scf.if %0 -> i32 {
      %c1_i32 = arith.constant 1 : i32
      scf.yield %c1_i32 : i32
    } else {
      %c1_i32 = arith.constant 1 : i32
      %2 = arith.subi %arg0, %c1_i32 : i32
      %3 = func.call @factorial(%2) : (i32) -> i32
      %4 = arith.muli %arg0, %3 : i32
      scf.yield %4 : i32
    }
    return %1 : i32
  }
}
```

### 5. Lowering and JIT

`MLIRLowering` applies passes to convert from the MLIR dialects (`arith`, `scf`, `func`, `memref`) down to the `LLVM` dialect, then emits LLVM IR as a string. `MLIRExecutor` feeds this LLVM IR to the ORC JIT engine with the configured optimization level (O0, O2, or O3), producing native machine code. The function pointer is stored and returned to Python as a `uintptr_t`.

### 6. Execution

`CppMLIRBackend.execute_function` retrieves the function pointer, wraps it with `ctypes.CFUNCTYPE` using the registered signature (mapping `i32 → ctypes.c_int32`), and calls it with the Python arguments. The native return value is marshalled back to a Python `int`.

## Python/C++ Boundary

The two sides communicate through **protobuf** and **function pointers**:

```mermaid
flowchart LR
    subgraph Python
        A[AST nodes] -->|"SerializeToString()"| B[protobuf bytes]
        B --> C
        D -->|"get_llvm_ir_string()"| E[LLVM IR string]
        E --> F
        G -->|"get_function_pointer()"| H["uintptr_t"]
        H --> I[ctypes wrapper]
        I -->|"call native fn"| J[Python result]
    end
    subgraph C++
        C -->|"ParseFromString()"| D[MLIRBuilder]
        F -->|"compile_module()"| G[MLIRExecutor]
    end
```

Python orchestrates both C++ objects. It does not hand off control — it mediates every step:

1. **Python → MLIRBuilder**: Serialized `FunctionDef` protobuf bytes sent via pybind11
2. **MLIRBuilder → Python**: LLVM IR returned as a string via `get_llvm_ir_string()`
3. **Python → MLIRExecutor**: LLVM IR string passed to `compile_module()`
4. **MLIRExecutor → Python**: Function pointer returned as `uintptr_t`, called via `ctypes.CFUNCTYPE`

The protobuf schema (`cpp/schemas/ast.proto`) is the single source of truth for AST node types, operation enums, and type definitions.

## Type System

All types inherit from the abstract `Type` base class:

```
Type (ABC)
├── ScalarType
│   ├── i32   (32-bit signed integer, Python int)
│   ├── f32   (32-bit float, Python float)
│   └── i1    (boolean, Python bool)
└── ArrayType
    ├── Array[4, i32]         (1D: memref<4xi32>)
    ├── Array[2, 3, f32]      (2D: memref<2x3xf32>)
    └── Array[2, 3, 4, i32]   (3D: memref<2x3x4xi32>)
```

Key rules:
- **No implicit type promotion.** `add(i32_val, f32_val)` is a type error. Use `cast()` explicitly.
- **Type errors are caught at decoration time**, not at runtime, via symbolic execution.
- **Python type hints map to MLIR types**: `int → i32`, `float → f32`, `bool → i1`.
- **Arrays cannot be cast.** Only scalar-to-scalar casts are allowed.

## MLIR Dialects Used

| Dialect | Purpose | Key operations |
|---|---|---|
| `arith` | Arithmetic and comparisons | `addi`, `muli`, `subi`, `divsi`, `cmpi`, `sitofp`, `fptosi` |
| `func` | Function definitions and calls | `func.func`, `func.call`, `return` |
| `scf` | Structured control flow | `scf.if`, `scf.for`, `scf.yield` |
| `memref` | Fixed-size arrays | `memref.alloca`, `memref.load`, `memref.store` |
| `cf` | Control flow (branching) | Used during lowering |
| `llvm` | LLVM dialect | Target of lowering passes |

## Related Documents

- **[ROADMAP.md](ROADMAP.md)** — Implementation phases, current status, and future plans
- **[BUILD.md](BUILD.md)** — Build instructions and dependencies
- **[tests/CLAUDE.md](tests/CLAUDE.md)** — Test conventions and patterns
