# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MLIR-based Embedded Domain-Specific Language (EDSL) for Machine Learning. The project uses a **Python frontend with C++ MLIR backend** architecture, implemented step-by-step starting with string-based MLIR generation and progressing toward a full AST → MLIR → LLVM pipeline.

**Current Status**: Phase 3 Complete - Ready for Phase 4 (MLIR → LLVM Lowering)

📋 **For detailed roadmap, implementation status, and technical specifications, see [ROADMAP.md](ROADMAP.md)**

## Development Approach

This project follows **user-driven implementation**. Claude should guide and advise rather than implement directly. The user will write the code with Claude providing:
- Architecture guidance and design recommendations
- Code review and suggestions  
- Debugging help and problem diagnosis
- Next step recommendations and planning
- Reference to ROADMAP.md for implementation details

## Key Implementation Guidelines

### Current Architecture (Phase 3)
- **C++ Backend**: `cpp/src/MLIRBuilder.cpp` - Real MLIR IR generation
- **Python Frontend**: `mlir_edsl/backend.py` - Python wrapper with pybind11
- **Testing**: `tests/test_cpp_backend.py` - Comprehensive backend tests
- **Build System**: CMake with LLVM/MLIR dependencies

### Code Organization
- `cpp/` - C++ MLIR backend implementation
- `mlir_edsl/` - Python frontend and API
- `tests/` - Test suite for all components
- `build/` - CMake build directory

### Working with the Codebase
1. **Always reference ROADMAP.md** for current implementation status
2. **Maintain user-driven approach** - guide rather than implement
3. **Focus on architecture and design** when advising
4. **Use existing test patterns** when adding new functionality
5. **Follow established C++/Python integration patterns**

## Testing Guidelines

### Test Strategy
- **Comprehensive Coverage**: All operations, types, and edge cases
- **Integration Testing**: Full Python API → C++ → MLIR pipeline
- **Backend Validation**: C++ backend matches expected MLIR output
- **Performance Testing**: When applicable for optimization verification

### Test Organization
- `tests/test_basic_ops.py` - Basic operation tests (Phase 1-2)
- `tests/test_cpp_backend.py` - C++ backend integration tests (Phase 3+)
- Future: `tests/test_llvm_lowering.py` - LLVM IR generation tests (Phase 4)

### Running Tests
```bash
# Run all tests
python3 -m pytest tests/ -v

# Run specific backend tests  
python3 -m pytest tests/test_cpp_backend.py -v

# Run single test
python3 -m pytest tests/test_cpp_backend.py::test_cpp_function_generation -v
```