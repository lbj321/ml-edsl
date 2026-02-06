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
- **C++ Backend**: `cpp/src/MLIRBuilder.cpp` - MLIR IR generation with dialect support
- **Python Frontend**: `mlir_edsl/` - AST-based frontend with strict typing
- **Execution**: `MLIRExecutor` - JIT compilation with optimization levels (O0/O2/O3)
- **Testing**: `tests/` - Comprehensive pytest suite with class-based organization
- **Build System**: CMake with LLVM/MLIR 18+ dependencies

### Code Organization
- `cpp/src/` - C++ MLIR backend implementation
  - `MLIRBuilder.cpp/h` - IR generation
  - `MLIRExecutor.cpp/h` - JIT execution engine
  - `MLIRLowering.cpp/h` - MLIR → LLVM lowering
- `mlir_edsl/` - Python frontend and API
  - `ast/` - AST node implementations (refactored module)
    - `base.py` - Value base class with core AST methods
    - `operators.py` - OperatorMixin with all operator overloads
    - `serialization.py` - SerializationContext and protobuf helpers
    - `helpers.py` - JAX-style .at[] array indexing
    - `nodes/` - Concrete AST node implementations
      - `scalars.py` - Constant, BinaryOp, CompareOp, CastOp
      - `arrays.py` - ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp
      - `control_flow.py` - IfOp, ForLoopOp, WhileLoopOp
      - `functions.py` - Parameter, CallOp
  - `ops/` - User-facing operation builders (refactored module)
    - `arithmetic.py` - add, sub, mul, div with scalar/array dispatch
    - `comparison.py` - lt, le, gt, ge, eq, ne with predicate inference
    - `control_flow.py` - If, For, While wrappers
    - `conversion.py` - cast, call utilities
  - `backend.py` - Python/C++ interface (pybind11)
  - `functions.py` - ml_function decorator implementation
  - `types.py` - Type system (ScalarType, ArrayType, TypeSystem)
- `tests/` - Test suite (see `tests/CLAUDE.md` for test-specific guidelines)
- `examples/` - Usage examples and demos
- `build/` - CMake build directory (gitignored)

### Working with the Codebase
1. **Always reference ROADMAP.md** for current implementation status and phase details
2. **Maintain user-driven approach** - guide rather than implement directly
3. **Focus on architecture and design** when advising
4. **Use existing test patterns** - Follow patterns in `tests/` (see `tests/CLAUDE.md`)
5. **Follow established C++/Python integration patterns** via pybind11
6. **Test incrementally** - Run tests frequently during development

## Testing Guidelines

### Quick Reference
- **Test Organization**: Class-based with `MLIRTestBase` inheritance
- **Detailed Guidelines**: See `tests/CLAUDE.md` for comprehensive test standards
- **Test Files**: Follow `test_<feature>.py` naming convention

### Running Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_parameters.py -v

# Run single test
python3 -m pytest tests/test_parameters.py::TestParameterFunctionality::test_basic_two_parameters -v

# Save IR to files in ir_output/ (for debugging)
SAVE_IR=1 python3 -m pytest tests/test_parameters.py -v
```

### Test Coverage Requirements
When adding new features:
1. Create corresponding test file: `tests/test_<feature>.py`
2. Test both success and error cases
3. Test edge cases and boundary conditions
4. Follow patterns in `tests/CLAUDE.md`

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