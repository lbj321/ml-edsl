# ARCHITECTURE.md

## Overview

This document describes the architecture of the MLIR-based EDSL, focusing on the Python-C++ integration, component responsibilities, and data flow.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Python Frontend                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ functions.py │  │    ops.py    │  │    ast.py    │      │
│  │  (Decorator) │  │ (Operations) │  │ (AST Nodes)  │      │
│  └──────┬───────┘  └──────────────┘  └──────┬───────┘      │
│         │                                     │              │
│         │          ┌──────────────┐          │              │
│         └─────────>│  backend.py  │<─────────┘              │
│                    │  (C++ Wrapper)│                         │
│                    └───────┬───────┘                         │
└────────────────────────────┼─────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Protobuf      │
                    │  Serialization  │
                    └────────┬────────┘
                             │
┌────────────────────────────▼─────────────────────────────────┐
│                       C++ Backend                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │           python_bindings.cpp (pybind11)             │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│  ┌─────────────────────▼──────────────────────────────┐     │
│  │              MLIRBuilder.cpp                        │     │
│  │  - Module management                                │     │
│  │  - Function creation/finalization                   │     │
│  │  - AST → MLIR IR generation                         │     │
│  │  - MLIR → LLVM IR lowering                          │     │
│  └─────────────────────┬──────────────────────────────┘     │
│                        │                                     │
│  ┌─────────────────────▼──────────────────────────────┐     │
│  │              MLIRExecutor.cpp                       │     │
│  │  - JIT compilation                                  │     │
│  │  - Function execution                               │     │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Python Layer

#### `functions.py` - Function Compilation Orchestration
**Purpose:** Provides the `@ml_function` decorator and orchestrates the compilation pipeline.

**Key Class:** `MLFunction`
- Wraps Python functions for MLIR compilation
- Manages parameter mapping (Python values → AST Parameters)
- Orchestrates symbolic execution
- Controls the function compilation lifecycle
- Handles JIT execution and returns results

**Compilation Flow:**
1. Create parameter map from function arguments
2. Execute function symbolically to build AST
3. Infer return type from AST
4. Create function declaration in C++ backend
5. Build function body from AST
6. Finalize function with return statement
7. JIT compile and execute

#### `backend.py` - C++ Backend Wrapper
**Purpose:** Python interface to C++ MLIR builder and executor.

**Key Class:** `CppMLIRBuilder`
- Wraps C++ `MLIRBuilder` via pybind11
- Provides Python-friendly API for MLIR operations
- Manages `MLIRValue` wrapper objects
- Handles protobuf serialization (new approach)
- Maintains global builder instance

**Two API Approaches:**
1. **Direct pybind11 calls:** `constant()`, `add()`, `sub()`, etc.
2. **Protobuf-based:** `process_ast_protobuf()` - serialize entire AST and send to C++

#### `ast.py` - AST Node Definitions
**Purpose:** Defines the abstract syntax tree node types.

**Key Classes:**
- `Value` - Base class for all AST nodes
- `Constant` - Literal values (int/float)
- `Parameter` - Function parameters
- `BinaryOp` - Binary operations (add, sub, mul, div)
- `CompareOp` - Comparison operations (eq, ne, lt, gt, etc.)
- `IfOp` - Conditional expressions
- `CallOp` - Function calls

**Responsibilities:**
- Define AST structure
- Provide `to_proto()` method for protobuf serialization
- Store type information

#### `ops.py` - Operation Constructors
**Purpose:** User-facing API for creating operations.

**Functions:**
- Arithmetic: `add()`, `sub()`, `mul()`, `div()`
- Comparisons: `eq()`, `ne()`, `lt()`, `gt()`, `le()`, `ge()`
- Control flow: `if_then_else()`

**Behavior:**
- Returns AST nodes during symbolic execution
- Simple Python wrapper functions

#### `control_flow.py` - Control Flow Operations
**Purpose:** Loop and control flow constructs.

**Key Classes:**
- `LoopOp` - Loop operation types (add, mul)
- `for_loop()` - For loop constructor
- `while_loop()` - While loop constructor

### C++ Layer

#### `python_bindings.cpp` - Python-C++ Interface
**Purpose:** Exposes C++ classes and methods to Python via pybind11.

**Exposed Classes:**
- `MLIRBuilder` - Main MLIR generation class
- `MLIRExecutor` - JIT compilation and execution class
- `MLIRValue` - Opaque wrapper for C++ MLIR values

**Exposed Methods:**
- Module management: `initialize_module()`, `reset()`
- Function lifecycle: `create_function()`, `finalize_function()`
- Operations: `build_constant()`, `build_add()`, `build_sub()`, etc.
- Control flow: `build_if()`, `build_for_with_op()`, `build_while_with_op()`
- Execution: `compile_function()`, `call_int32_function()`, `call_float_function()`
- IR generation: `get_mlir_string()`, `get_llvm_ir_string()`
- Protobuf: `process_ast_from_protobuf()` (TODO)

#### `MLIRBuilder.cpp` - MLIR IR Generation
**Purpose:** Core MLIR IR generation and management.

**Responsibilities:**
- MLIR context, module, and builder management
- Function creation and body generation
- Type management (i32, f32, i1)
- Operation construction (arith, scf, func dialects)
- MLIR → LLVM IR lowering
- Parameter tracking and SSA value management

**Key State:**
- Current module
- Current function being built
- Parameter map (name → BlockArgument)
- Builder insertion point

**Critical Design Question:** Should state persist across multiple function compilations?

#### `MLIRExecutor.cpp` - JIT Compilation and Execution
**Purpose:** JIT compile MLIR/LLVM IR and execute functions.

**Responsibilities:**
- LLVM JIT engine initialization
- Function compilation from LLVM IR
- Function pointer management
- Type-safe function invocation (i32, f32 return types)
- Error handling and reporting

## Communication Protocol

### Approach 1: Direct pybind11 Calls (Current)

**Flow:**
```
Python AST → Python backend.py → pybind11 → C++ MLIRBuilder methods
```

**Advantages:**
- Direct control
- Type checking at pybind11 boundary
- Incremental MLIR construction

**Disadvantages:**
- Multiple Python→C++ calls per AST node
- More overhead for complex ASTs

### Approach 2: Protobuf Serialization (New)

**Flow:**
```
Python AST → to_proto() → Protobuf bytes → C++ deserialize → MLIR IR
```

**Protobuf Schema Location:** `cpp/schemas/ast.proto`

**Advantages:**
- Single Python→C++ call per AST
- Better performance for large ASTs
- Language-agnostic serialization
- Easier versioning

**Disadvantages:**
- Additional serialization/deserialization overhead
- Schema maintenance
- Less direct error reporting

### Design Decision: Protobuf for Function ASTs

**Decision:** Use **protobuf serialization** for passing complete function ASTs to C++.

**Rationale:**
- Each function's AST is serialized once and sent as a complete unit
- Better performance for complex functions with many operations
- Cleaner separation: Python builds AST, C++ builds MLIR
- Single protobuf message per function = single Python→C++ call

**Implementation:**
```python
# functions.py
backend.create_function(name, params, return_type)
backend.build_from_protobuf(ast.to_proto().SerializeToString())
backend.finalize_function(name, result)
```

## Function Lifecycle

### Single-Function Flow (Option C)

```python
@ml_function
def example(x: int):
    return x + 1

result = example(5)
```

**Steps:**
1. `MLFunction.__call__(5)` invoked
2. Check `backend.has_function("example")` → False (first call)
3. Create parameter map: `{"x": Parameter("x", 5, "i32")}`
4. Symbolic execution: `example(Parameter("x"))` → `BinaryOp("add", Parameter("x"), Constant(1))`
5. Infer return type: `"i32"`
6. **Compile function:**
   - `backend.create_function("example", [("x", "i32")], "i32")` - Declare function
   - `backend.build_from_protobuf(ast.to_proto().SerializeToString())` - Build body via protobuf
   - `backend.finalize_function("example", result)` - Add return statement
   - Backend marks "example" as compiled
7. `backend.execute_function("example", [5], [])` - JIT and execute
8. Return result: `6`

**Second call:** `result2 = example(10)`
1. Check `backend.has_function("example")` → True (already compiled)
2. Skip compilation, go directly to execution
3. `backend.execute_function("example", [10], [])`
4. Return result: `11`

### Multiple Function Flow (Option C)

**Scenario 1: Sequential Independent Functions**
```python
@ml_function
def add_one(x: int):
    return x + 1

@ml_function
def add_two(x: int):
    return x + 2

result1 = add_one(5)   # Returns 6
result2 = add_two(10)  # Returns 12
```

**Flow:**
1. **Call `add_one(5)`:**
   - Check `has_function("add_one")` → False
   - Compile `add_one` into module
   - Execute and return `6`

2. **Call `add_two(10)`:**
   - Check `has_function("add_two")` → False
   - Compile `add_two` into **same module** (no reset!)
   - Execute and return `12`

**Result:** Module now contains both functions, both remain available.

**Scenario 2: Function Calls Between Functions**
```python
@ml_function
def add_one(x: int):
    return x + 1

@ml_function
def add_two(x: int):
    return add_one(x) + 1  # CallOp("add_one")

result = add_two(5)  # Should return 7
```

**Flow:**
1. **Call `add_two(5)`:**
   - Check `has_function("add_two")` → False
   - Symbolic execution creates AST with `CallOp("add_one", ...)`
   - Detect dependency on `add_one`
   - Check `has_function("add_one")` → False

2. **Compile dependency first:**
   - Compile `add_one` into module
   - Mark as compiled

3. **Then compile `add_two`:**
   - Compile `add_two` with `CallOp("add_one")` - now valid!
   - Mark as compiled

4. **Execute `add_two(5)`:**
   - JIT compiles module (both functions)
   - Executes: `5 → add_one(5) → 6 → 6+1 → 7`
   - Return `7`

**Result:** ✅ Cross-function calls work! Module persists across compilations.

**Scenario 3: Recursive Functions**
```python
@ml_function
def factorial(n: int):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # Self-reference

result = factorial(5)  # Should return 120
```

**Flow:**
1. **Call `factorial(5)`:**
   - Check `has_function("factorial")` → False
   - Symbolic execution creates AST with `CallOp("factorial", ...)` (self-reference)
   - Detect recursive call to self

2. **Compile with three-phase approach:**
   - **Phase 1:** `create_function("factorial", ...)` - Declare signature
   - **Phase 2:** `build_from_protobuf(ast)` - Build body (can reference declared function)
   - **Phase 3:** `finalize_function("factorial", ...)` - Complete function
   - Mark as compiled

3. **Execute `factorial(5)`:**
   - JIT compiles module
   - Executes recursively: `5*4*3*2*1 = 120`
   - Return `120`

**Result:** ✅ Recursion works! Three-phase compilation enables self-reference.

### Program Structure: Multiple ASTs vs Single AST

**Question:** Is the entire program one AST, or does it consist of several?

**Answer:** The program consists of **multiple ASTs** (one per function), all compiled into **one MLIR module**.

**Structure:**
```python
@ml_function
def add_one(x: int):
    return add(x, 1)
    # AST 1: BinaryOp("add", Parameter("x"), Constant(1))

@ml_function
def double_then_add_one(y: int):
    doubled = mul(y, 2)
    return add_one(doubled)
    # AST 2: Sequence of BinaryOp("mul", ...) + CallOp("add_one", ...)
```

**Python Side:**
- Function 1 → AST 1 → `to_proto()` → protobuf bytes 1
- Function 2 → AST 2 → `to_proto()` → protobuf bytes 2

**C++ Side:**
- Receives protobuf bytes 1 → deserialize → build `func.func @add_one`
- Receives protobuf bytes 2 → deserialize → build `func.func @double_then_add_one`
- Both functions in **same MLIR module**

**Final MLIR Module:**
```mlir
module {
  func.func @add_one(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %result = arith.addi %x, %c1 : i32
    return %result : i32
  }

  func.func @double_then_add_one(%y: i32) -> i32 {
    %c2 = arith.constant 2 : i32
    %doubled = arith.muli %y, %c2 : i32
    %result = call @add_one(%doubled) : (i32) -> i32
    return %result : i32
  }
}
```

**Key Points:**
- ✅ Each function = separate AST (natural Python model)
- ✅ Each AST = separate protobuf message
- ✅ All functions = single MLIR module (enables cross-function calls)
- ✅ Module persists (no reset between functions)

### Chosen Approach: Option C (Hybrid - Multiple ASTs with Managed Program State)

**Decision:** Each function has its own AST, but the backend explicitly manages them as a unified program.

**Architecture:**
- **Multiple ASTs:** Each `@ml_function` has its own AST tree
- **One MLIR Module:** All functions compiled into a single persistent MLIR module
- **Explicit State Management:** Backend exposes program lifecycle methods
- **No Automatic Reset:** Module persists across function compilations

**Key Design Principles:**

1. **Persistent Module:**
   - Remove `backend.reset()` from `MLFunction.__call__()`
   - Backend maintains a single MLIR module for all functions
   - Functions accumulate in the module as they are compiled

2. **Function Tracking:**
   - Backend tracks which functions are already compiled
   - `backend.has_function(name)` - Check if function exists
   - Avoid recompiling already-defined functions

3. **Dependency Management:**
   - Functions can call previously compiled functions
   - Lazy compilation: compile dependencies as needed
   - Support for recursive functions (already works via two-phase creation)

4. **Explicit Lifecycle Control:**
   - `backend.clear_module()` - Reset module to empty state
   - `backend.list_functions()` - Get all compiled function names
   - `backend.get_module_state()` - Inspect module (future)

**Implementation Changes Required:**

**Python (`functions.py`):**
```python
def __call__(self, *args, **kwargs):
    backend = get_backend()
    # REMOVED: backend.reset()  ← Don't reset!

    # Check if already compiled
    if not backend.has_function(self.func_name):
        self._compile_function()

    # Execute
    return backend.execute_function(...)
```

**Python (`backend.py`):**
```python
class CppMLIRBuilder:
    def has_function(self, name: str) -> bool:
        """Check if function already compiled in module"""
        return self.builder.has_function(name)

    def clear_module(self):
        """Clear all functions from module (explicit reset)"""
        self.builder.clear_module()

    def list_functions(self) -> list[str]:
        """Get names of all compiled functions"""
        return self.builder.list_functions()
```

**C++ (`MLIRBuilder.cpp`):**
```cpp
class MLIRBuilder {
private:
    std::unordered_set<std::string> compiledFunctions;

public:
    bool hasFunction(const std::string& name);
    void clearModule();  // Clear module + compiledFunctions
    std::vector<std::string> listFunctions();
};
```

**Trade-offs:**
- ✅ Simple user API (no explicit program object)
- ✅ Supports cross-function calls naturally
- ✅ Supports recursion (already works)
- ✅ Explicit control when needed (`clear_module()`)
- ✅ Can inspect program state (`list_functions()`)
- ⚠️ Module grows over session (need explicit `clear_module()`)
- ⚠️ Name collisions if function redefined (can add warning/error)

## Protobuf Integration

### Schema Location
- **File:** `cpp/schemas/ast.proto`
- **Generated Python:** `mlir_edsl/ast_pb2.py`
- **Generated C++:** `cpp/schemas/ast.pb.h`, `cpp/schemas/ast.pb.cc`

### AST Serialization

Each AST node class implements `to_proto()`:
```python
class Constant(Value):
    def to_proto(self):
        node = ast_pb2.ASTNode()
        node.constant.value = self.value
        node.constant.type = self.type
        return node
```

### C++ Deserialization

**TODO:** Define C++ API for protobuf processing:
```cpp
// Option 1: Return MLIRValue (matches current buildFromAST)
mlir::Value MLIRBuilder::processASTFromProtobuf(const std::string& buffer);

// Option 2: Handle everything internally
void MLIRBuilder::processASTFromProtobuf(const std::string& buffer, const std::string& functionName);

// Option 3: Two-phase (recommended)
mlir::Value MLIRBuilder::buildFromProtobuf(const std::string& buffer);
```

### Integration with Function Lifecycle

**Current (pybind11):**
```python
backend.create_function(name, params, return_type)
mlir_result = backend.buildFromAST(ast_node)
backend.finalize_function(name, mlir_result)
```

**Proposed (protobuf):**
```python
backend.create_function(name, params, return_type)
mlir_result = backend.build_from_protobuf(ast_node.to_proto().SerializeToString())
backend.finalize_function(name, mlir_result)
```

**OR (simplified):**
```python
backend.create_function(name, params, return_type)
backend.build_function_body_protobuf(ast_node)  # Handles build + finalize
```

## Memory and Ownership

### Python Side
- `CppMLIRBuilder` maintains C++ object via pybind11 smart pointers
- `MLIRValue` wraps opaque C++ `mlir::Value` handles
- Global `_global_builder` instance keeps builder alive

### C++ Side
- MLIR context owns all IR objects (values, operations, types)
- `MLIRBuilder` owns context and module
- `MLIRExecutor` owns JIT engine
- **Question:** Who owns function pointers after compilation?
- **Question:** Lifetime of compiled functions relative to module?

## Build System

### CMake Configuration
- **Root:** `CMakeLists.txt`
- **C++ Backend:** `cpp/CMakeLists.txt`
- **Protobuf Generation:** `cmake/ProtobufSchemas.cmake`
- **FlatBuffers:** `cmake/FlatBuffersSchemas.cmake` (future?)

### Dependencies
- LLVM/MLIR libraries
- Protobuf
- pybind11
- Python development headers

### Build Process
```bash
./build.sh  # Runs CMake + generates protobuf + builds C++ + installs Python package
```

## Error Handling

### Python Layer
- Type errors in symbolic execution
- Missing parameters
- Invalid operations
- Backend unavailable

### C++ Layer
- MLIR verification failures
- LLVM IR lowering failures
- JIT compilation failures
- Function execution errors

### Error Propagation
- C++ exceptions → pybind11 → Python exceptions
- Error messages from `MLIRExecutor::getLastError()`

## Testing Strategy

### Python Tests
- `tests/test_basic_ops.py` - Basic operations
- `tests/test_cpp_backend.py` - C++ backend integration
- `tests/test_conditionals.py` - If/else operations
- `tests/test_loops.py` - For/while loops
- `tests/test_recursion.py` - Recursive functions

### C++ Tests
**TODO:** Add C++ unit tests for:
- MLIRBuilder operations
- Protobuf deserialization
- LLVM lowering
- JIT execution

## Future Enhancements

### Short Term
- [ ] Finalize protobuf integration
- [ ] Define multi-function compilation strategy
- [ ] Improve error messages
- [ ] Add type inference improvements

### Medium Term
- [ ] Tensor operations
- [ ] Custom types/structs
- [ ] Memory operations (alloc/load/store)
- [ ] Optimization passes

### Long Term
- [ ] GPU backend
- [ ] Automatic differentiation
- [ ] Distributed execution
- [ ] Advanced ML operations

## Resolved Design Decisions

1. ✅ **Multi-function compilation:** Option C (Hybrid - Multiple ASTs with Managed Program State)
2. ✅ **Protobuf vs pybind11:** Use protobuf for complete function AST serialization
3. ✅ **Module persistence:** Maintain persistent global module (no reset between functions)
4. ✅ **Program structure:** Multiple ASTs (one per function) → One MLIR Module

## Open Questions

1. **Function pointer management:** How to handle multiple compiled functions in JIT executor?
2. **Dependency detection:** Automatic vs manual (how to detect `CallOp` dependencies)?
3. **Error recovery:** Can we recover from compilation failures without losing module?
4. **Type system:** Should we support more MLIR types (i64, f64, tensors)?
5. **Optimization:** When to run MLIR optimization passes?
6. **Function redefinition:** Error, warning, or allow overwrite?

## Next Steps

See TODO list for implementation tasks:
1. Define backend state management API (`has_function`, `clear_module`, etc.)
2. Remove `backend.reset()` from `functions.py`
3. Implement C++ module persistence and function tracking
4. Update protobuf integration to work with persistent module
5. Implement dependency detection and lazy compilation

---

**Document Status:** Architecture Defined - Ready for Implementation
**Last Updated:** 2025-10-01
