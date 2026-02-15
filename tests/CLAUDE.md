# Tests — Structure and Guidelines

## Test Categories

Every test should fall into exactly one of these categories:

### 1. Runtime Execution Tests (`test_*_execution.py`, `test_*.py`)
Verify that compiled functions produce correct values.

```python
class TestFeatureExecution:
    def test_something(self, backend):
        @ml_function
        def my_func(x: int) -> int:
            return add(x, 5)
        assert my_func(10) == 15
```

- **Requires**: `backend` fixture
- **Asserts**: Return values, side effects
- **This is the primary correctness layer** — if the function produces the right answer, the IR was good enough

### 2. IR Structure Tests (`test_ir.py`)
Verify MLIR properties that **cannot be caught by runtime tests**.

```python
class TestFeatureIR:
    def test_no_duplicate_comparison(self, check_ir):
        @ml_function
        def clamp(x: int, lo: int, hi: int) -> int:
            return If(lt(x, lo), lo, If(gt(x, hi), hi, x))
        clamp(1, 2, 3)
        check_ir("""
        // CHECK: arith.cmpi slt
        // CHECK-NOT: arith.cmpi slt
        """)
```

- **Requires**: `check_ir` fixture (depends on FileCheck binary)
- **Asserts**: IR structure via FileCheck patterns

**Good IR tests** check things invisible to runtime:
- Dialect boundary enforcement (no `tensor` ops in memref code, no `memref` ops in tensor code)
- Optimization properties (no duplicate comparisons, no redundant allocations)
- Allocation strategy (`alloca` vs `alloc`)
- Correct type representation (`memref<2x3xf32>` not `memref<6xf32>`)
- Structural invariants (function signatures, block structure)

**Bad IR tests** (don't write these — they duplicate runtime tests):
- "Does `add(x, y)` emit `arith.addi`?" — if it didn't, the runtime test would fail too
- "Does array access emit `memref.load`?" — same, runtime catches this
- Basic "does this op exist?" checks with no structural insight

### 3. AST Construction Tests (`test_*_ast.py`)
Verify that Python AST nodes are built correctly, without any backend.

```python
class TestArrayAST:
    def test_access_type_inference(self):
        arr = ArrayLiteral([1, 2, 3], Array[i32, 3])
        access = arr[1]
        assert access.infer_type() == i32
```

- **Requires**: Nothing (pure Python)
- **Asserts**: Node properties, type inference, `get_children()`, protobuf serialization
- Tests here should never need the `backend` fixture

### 4. Type System Tests (`test_*_types.py`)
Verify type creation, validation, equality, and error handling.

```python
class TestArrayTypeValidation:
    def test_negative_size_rejected(self):
        with pytest.raises(TypeError):
            Array[i32, -1]

    def test_type_equality(self):
        assert Array[i32, 4] == Array[i32, 4]
        assert Array[i32, 4] != Array[f32, 4]
```

- **Requires**: Nothing (pure Python)
- **Asserts**: Type properties, equality, hashing, MLIR string output, error messages
- Includes validation of subscript syntax (`Array[dtype, dims]`, `Tensor[dtype, dims]`)

### 5. Error and Validation Tests
Verify that bad input produces clear error messages. These live **within** the appropriate category file rather than in a separate file.

```python
class TestStrictTyping:
    def test_missing_return_type_raises(self):
        with pytest.raises(TypeError, match="return type"):
            @ml_function
            def bad(x: int):
                return x
```

- Use `pytest.raises(ExceptionType, match="substring")` to verify both exception type and message
- Test error paths alongside the happy paths in the same file

## Directory Layout

```
tests/
├── conftest.py              # Shared fixtures (backend, clean_module, check_ir)
├── CLAUDE.md                # This file
├── core/                    # Scalar ops, control flow, recursion, typing
│   ├── test_parameter.py        # Runtime: parameter passing
│   ├── test_control_flow.py     # Runtime: if/else, comparisons
│   ├── test_recursion.py        # Runtime: recursive functions
│   ├── test_strict_typing.py    # Type system + errors: type hints, validation
│   ├── test_jit_integration.py  # Runtime: operator overloading, JIT
│   ├── test_cpp_backend.py      # Backend: low-level backend API
│   ├── test_optimization_benchmark.py  # Backend: optimization levels
│   └── test_ir.py               # IR: structural checks for core ops
├── memref/                  # Array operations (memref dialect)
│   ├── test_array_ast.py        # AST: ArrayLiteral, ArrayAccess, ArrayStore
│   ├── test_array_types.py      # Type system: ArrayType creation, validation
│   ├── test_array_execution.py  # Runtime: array ops end-to-end
│   ├── test_array_elementwise.py # Runtime: element-wise ops, broadcasting
│   ├── test_array_2d.py         # Mixed: 2D array AST + runtime
│   ├── test_array_3d.py         # Mixed: 3D array AST + runtime
│   └── test_ir.py               # IR: memref structural checks
└── tensor/                  # Tensor operations (tensor dialect)
    ├── test_tensor_ast.py       # AST: TensorFromElements, TensorExtract
    ├── test_tensor_types.py     # Type system: TensorType creation, validation
    ├── test_tensor_execution.py # Runtime: tensor ops end-to-end
    ├── test_tensor_insert.py    # Mixed: TensorInsert AST + runtime
    └── test_ir.py               # IR: tensor structural checks
```

## Fixtures (defined in `conftest.py`)

| Fixture | Scope | Purpose |
|---|---|---|
| `backend` | session | C++ backend instance; auto-skips if unavailable |
| `clean_module` | function (autouse) | Clears module before each test; saves IR after if `SAVE_IR=1` |
| `check_ir` | function | FileCheck-based IR assertions; skips if FileCheck not found |

## IR Inspection Workflow

`SAVE_IR=1` generates HTML reports showing the full compilation pipeline (Python source, AST, MLIR, lowering passes, LLVM IR). Output mirrors the test directory structure:

```
ir_html/
├── core/
│   ├── test_parameter/
│   │   ├── test_basic_two_parameters.html
│   │   └── test_explicit_type_casting.html
│   └── test_control_flow/
│       └── test_if_greater_than.html
├── memref/
│   ├── test_array_execution/
│   │   ├── test_array_access_execution.html
│   │   └── test_array_store_execution.html
│   └── test_array_2d/
│       └── ...
└── tensor/
    └── ...
```

**Target specific tests** to avoid generating hundreds of files:

```bash
# Inspect IR for a single test file
SAVE_IR=1 python3 -m pytest tests/memref/test_array_execution.py -v

# Inspect IR for one specific test
SAVE_IR=1 python3 -m pytest tests/memref/test_array_execution.py::TestArrayExecution::test_array_access_execution -v

# Inspect IR for a whole dialect
SAVE_IR=1 python3 -m pytest tests/tensor/ -v
```

The `ir_html/` directory is cleared at the start of each test session, so only the tests you just ran will be present.

## Conventions

- **No base classes** — tests are plain classes, no inheritance
- **One assert concept per test** — test one behavior, not a whole workflow
- **Test names describe the expectation**: `test_negative_size_rejected`, not `test_size`
- **Backend-dependent tests** take `backend` as a fixture parameter; pure-Python tests omit it
- **Subscript syntax**: `Array[dtype, dims]` and `Tensor[dtype, dims]` (type first, shape after)

## Running Tests

```bash
python3 -m pytest tests/ -v                    # All tests
python3 -m pytest tests/core/ -v               # Core tests only
python3 -m pytest tests/memref/test_ir.py -v   # Specific file
SAVE_IR=1 python3 -m pytest tests/ -v          # Save IR to ir_output/ + ir_html/
DUMP_AST=1 python3 -m pytest tests/ -v         # Dump AST to ast_output/
```

## When Adding a New Feature

1. Add **type system tests** if introducing new types or validation rules
2. Add **AST tests** if introducing new AST nodes (construction, type inference, serialization)
3. Add **runtime execution tests** to verify correctness end-to-end
4. Add **IR tests** only for structural properties invisible to runtime
