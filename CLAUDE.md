# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLIR-based Embedded Domain-Specific Language (EDSL) for Machine Learning. The project uses a **Python frontend with C++ MLIR backend** architecture, implemented step-by-step starting with string-based MLIR generation and progressing toward a full AST → MLIR → LLVM pipeline.

📋 **For detailed roadmap, implementation status, and technical specifications, see [ROADMAP.md](ROADMAP.md)**

## Development Approach

This project follows **user-driven implementation**. Claude should guide and advise rather than implement directly. The user will write the code with Claude providing:
- Architecture guidance and design recommendations
- Code review and suggestions
- Debugging help and problem diagnosis
- Next step recommendations and planning
- Reference to ROADMAP.md for implementation details

## Coding Standards

### Python Code
- **Style**: PEP 8 compliant
- **Formatting**: Use consistent spacing and naming conventions
- **Type Hints**: Required for all public API functions and class methods
  ```python
  def process_value(x: int, dtype: Type) -> Value:
      """Process a value with specified dtype."""
      ...
  ```
- **Imports**: Organize as stdlib, third-party, local (use blank lines between groups)
- **Docstrings**: Required for all public functions/classes
  - Use triple quotes even for one-liners
  - Format: Brief description, then Args/Returns sections for complex functions
- **Naming Conventions**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### C++ Code
- **Style**: Follow LLVM coding standards (for MLIR consistency)
- **Naming**:
  - Classes/Types: `PascalCase` (e.g., `MLIRBuilder`)
  - Functions/Methods: `camelCase` (e.g., `buildFunction`)
  - Variables: `camelCase` (e.g., `moduleOp`)
  - Constants: `kPascalCase` (e.g., `kDefaultOptLevel`)
- **Headers**: Use `#pragma once` for header guards
- **Documentation**: Doxygen-style comments for public APIs
- **Error Handling**: Return `mlir::LogicalResult` where appropriate

### General Principles
- **Clarity over cleverness** - Write self-documenting code
- **DRY principle** - Don't repeat yourself; extract common patterns
- **Small, focused functions** - Aim for <50 lines where practical
- **Consistent error handling** - Use exceptions in Python, LogicalResult in C++
- **Comment the "why", not the "what"** - Code should be self-explanatory

## Architecture

### Core Components
- **C++ Backend**: `MLIRCompiler` (unified facade) → `MLIRBuilder` → `MLIRLowering` → `MLIRExecutor`
- **Python Frontend**: `mlir_edsl/` - AST-based frontend with strict typing
- **Execution**: `MLIRExecutor` - JIT compilation with optimization levels (O0/O2/O3)
- **Testing**: `tests/` - Comprehensive pytest suite with conftest fixtures and FileCheck IR tests
- **Build System**: CMake with LLVM/MLIR 18+ dependencies

### Code Organization
- `cpp/include/mlir_edsl/` - C++ headers
  - `MLIRCompiler.h` - Unified facade (only class exposed to Python)
  - `MLIRBuilder.h` - IR generation
  - `MLIRExecutor.h` - JIT execution engine
  - `MLIRLowering.h` - MLIR → LLVM lowering
  - `ArithBuilder.h`, `MemRefBuilder.h`, `SCFBuilder.h`, `TensorBuilder.h` - Dialect builders
  - `proto_fwd.h` - Forward declarations for protobuf types
- `cpp/src/` - C++ MLIR backend implementation
  - `MLIRCompiler.cpp` - Unified facade orchestrating Builder → Lowering → Executor
  - `MLIRBuilder.cpp` - Low-level IR generation
  - `MLIRExecutor.cpp` - JIT execution engine
  - `MLIRLowering.cpp` - MLIR → LLVM lowering
  - `python_bindings.cpp` - pybind11 bindings (protobuf deserialization happens here)
  - `builders/` - Dialect builder implementations
    - `ArithBuilder.cpp` - Arithmetic dialect operations
    - `MemRefBuilder.cpp` - MemRef dialect (arrays)
    - `SCFBuilder.cpp` - Structured control flow (if/for/while)
    - `TensorBuilder.cpp` - Tensor dialect operations
- `cpp/schemas/` - Protobuf schema definitions
  - `ast.proto` - AST node protobuf schema
- `cmake/` - CMake helper modules
  - `CompilerFlags.cmake`, `Development.cmake`, `FindMLIR.cmake`, `MLIRDialects.cmake`, `ProtobufSchemas.cmake`
- `mlir_edsl/` - Python frontend and API
  - `ast/` - AST node implementations
    - `base.py` - Value base class with core AST methods
    - `operators.py` - OperatorMixin with all operator overloads
    - `serialization.py` - SerializationContext and protobuf helpers
    - `helpers.py` - JAX-style .at[] array indexing
    - `dump.py` - AST dump utility (DUMP_AST=1 env var)
    - `nodes/` - Concrete AST node implementations
      - `scalars.py` - Constant, BinaryOp, CompareOp, CastOp
      - `arrays.py` - ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp
      - `tensors.py` - TensorLiteral, TensorInsert, TensorBinaryOp
      - `control_flow.py` - IfOp, ForLoopOp, WhileLoopOp
      - `functions.py` - Parameter, CallOp
  - `ops/` - User-facing operation builders
    - `arithmetic.py` - add, sub, mul, div with scalar/array dispatch
    - `comparison.py` - lt, le, gt, ge, eq, ne with predicate inference
    - `control_flow.py` - If, For, While wrappers
    - `conversion.py` - cast, call utilities
  - `functions/` - ml_function decorator and compilation
    - `decorator.py` - ml_function decorator implementation
    - `compilation.py` - ML function compilation logic
    - `context.py` - Function execution context management
    - `signature.py` - Function signature caching/validation
    - `validation.py` - Type and parameter validation
  - `backend.py` - Python/C++ interface (pybind11)
  - `types.py` - Type system (ScalarType, ArrayType, TypeSystem)
  - `ast_pb2.py` - Generated protobuf Python module
- `tests/` - Test suite
  - `conftest.py` - Pytest fixtures (`backend`, `clean_module`, `check_ir`)
  - `core/` - Core feature tests (parameters, control flow, recursion, typing, backend, JIT, optimization, IR)
  - `memref/` - MemRef/array tests (2D, 3D, AST, elementwise, execution, types, IR)
  - `tensor/` - Tensor dialect tests (AST, execution, insert, types, IR)
- `examples/` - Usage examples and demos
- `build/` - CMake build directory (gitignored)

### Working with the Codebase
1. **Always reference ROADMAP.md** for current implementation status and phase details
2. **Maintain user-driven approach** - guide rather than implement directly
3. **Focus on architecture and design** when advising
4. **Use existing test patterns** - Follow patterns in existing test files
5. **Follow established C++/Python integration patterns** via pybind11
6. **Test incrementally** - Run tests frequently during development

## Testing

**For test structure, categories, conventions, and guidelines, see [`tests/CLAUDE.md`](tests/CLAUDE.md).**

### Running Tests
```bash
python3 -m pytest tests/ -v                    # All tests
python3 -m pytest tests/core/ -v               # Core tests only
SAVE_IR=1 python3 -m pytest tests/ -v          # Save IR to ir_output/ + ir_html/
DUMP_AST=1 python3 -m pytest tests/ -v         # Dump AST to ast_output/
```

## Build and Development

### Building the Project
```bash
# Build the project
./build.sh

# Clean build
./build.sh clean
```

### Development Workflow
1. Make changes to Python or C++ code
2. If C++ changes: rebuild with `./build.sh`
3. Run relevant tests to verify changes
4. Check ROADMAP.md to ensure alignment with implementation plan
5. Update tests if adding new functionality