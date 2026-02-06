# Test Guidelines for MLIR EDSL

This file provides guidance to Claude Code when writing or updating tests in this test suite.

## Overview

All tests in this project use a common base class (`MLIRTestBase`) that provides:
- Automatic backend initialization and cleanup
- Optional IR file output via `SAVE_IR=1` environment variable (saved to `ir_output/`)
- Proper test isolation with `setup_method()` and `teardown_method()`

## File Organization Patterns

### Pattern 1: Single Test Class (Simple Features)
Use for focused, single-feature test files:

```python
"""Test <feature name> with <brief description>"""

import pytest
from mlir_edsl import <imports>
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


class Test<FeatureName>(MLIRTestBase):
    """Test <feature description>"""

    def test_basic_scenario(self):
        """Test <specific scenario being validated>"""
        # Test implementation
        pass
```

**Example**: `test_parameters.py`

### Pattern 2: Multiple Test Classes (Complex Features)
Use for comprehensive feature testing with multiple aspects:

```python
"""Tests for <feature name>

Multi-line description explaining what this test suite validates:
- Key aspect 1
- Key aspect 2
- Key aspect 3
"""

import pytest
from mlir_edsl import <imports>
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


# ==================== SECTION NAME ====================

class Test<AspectName>(MLIRTestBase):
    """Test <specific aspect of feature>"""

    def test_scenario_one(self):
        """Test <what this validates>"""
        pass


# ==================== ANOTHER SECTION ====================

class Test<AnotherAspect>(MLIRTestBase):
    """Test <another aspect of feature>"""

    def test_scenario_two(self):
        """Test <what this validates>"""
        pass
```

**Example**: `test_strict_typing.py`

**Section Comment Format**: Use `# ====` style with ALL CAPS section names for visual separation.

## Naming Conventions

### File Names
- Pattern: `test_<feature_name>.py`
- Use underscores for multi-word features
- Examples: `test_parameters.py`, `test_strict_typing.py`, `test_control_flow.py`

### Class Names
- Pattern: `Test<FeatureName>` or `Test<FeatureAspect>`
- Use PascalCase
- Be specific about what aspect is being tested
- Examples: `TestParameterFunctionality`, `TestTypeHintValidation`, `TestExplicitCastOperations`

### Test Method Names
- Pattern: `test_<specific_scenario_being_tested>`
- Use underscores to separate words
- Be descriptive - the name should explain what's being validated
- Examples:
  - `test_basic_two_parameters`
  - `test_missing_parameter_type_hint`
  - `test_cast_execution_int_to_float`

## Docstring Requirements

### File-Level Docstring
**Required** - First line of every test file:
```python
"""Test <feature name> [with additional context]"""
```

For complex test files, use multi-line format:
```python
"""Tests for <feature name>

This test suite validates:
- Aspect 1
- Aspect 2
- Aspect 3
"""
```

### Class-Level Docstring
**Required** - Every test class needs a docstring:
```python
class TestFeatureName(MLIRTestBase):
    """Test <what this class validates>"""
```

### Method-Level Docstring
**Required** - Every test method needs a docstring:
```python
def test_something(self):
    """Test <specific scenario being validated>"""
```

The docstring should clearly explain:
- What specific behavior is being tested
- What edge case is being validated
- What error condition is being checked

## Test Implementation Patterns

### Basic Test Structure
```python
def test_feature(self):
    """Test <what is being validated>"""
    @ml_function
    def my_func(x: int) -> int:
        return add(x, 5)

    result = my_func(10)
    assert result == 15
```

### Error Testing Pattern
```python
def test_error_condition(self):
    """Test that <invalid operation> raises <ErrorType>"""
    with pytest.raises(TypeError, match="expected error message pattern"):
        @ml_function
        def bad_func(x: int) -> float:
            return x  # Type mismatch
```

### Conditional Tests (Backend-Dependent)
For tests requiring execution (not just compilation):
```python
@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_runtime_behavior(self):
    """Test runtime execution with backend"""
    @ml_function
    def my_func(x: int) -> int:
        return add(x, 5)

    result = my_func(10)
    assert result == 15
```

**When to use**: Any test that calls the generated function and checks return values needs the C++ backend.

## Import Patterns

### Standard Imports
```python
import pytest
from mlir_edsl import ml_function, add, sub, mul, div, cast
from mlir_edsl import i32, f32, i1
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase
```

### Feature-Specific Imports
Import only what you need for the specific tests:
```python
from mlir_edsl import For, While, If  # Control flow
from mlir_edsl import lt, gt, eq      # Comparison ops
```

## Assertion Patterns

### Exact Value Assertions
```python
assert result == 15
assert result == expected_value
```

### Floating-Point Assertions
```python
assert abs(result - 15.5) < 0.001
```

### Type Assertions
```python
assert isinstance(result, int)
assert isinstance(result, float)
```

### Error Message Validation
```python
try:
    # Code that should raise
    assert False, "Should have raised TypeError"
except TypeError as e:
    error_msg = str(e)
    assert "expected substring" in error_msg
```

## Environment Variables

### SAVE_IR=1
Save MLIR and LLVM IR to files in `ir_output/`:
```bash
SAVE_IR=1 pytest tests/test_parameters.py -v
```

Output files:
- `ir_output/<test_name>.mlir` - MLIR dialect
- `ir_output/<test_name>.ll` - LLVM IR
- `ir_output/<test_name>.html` - HTML with collapsible sections

## Test Organization Guidelines

### When to Create a New Test File
- Testing a distinct feature area (parameters, typing, control flow, etc.)
- File size exceeds ~500 lines
- Tests are logically separate from existing test files

### When to Add to Existing File
- Testing an aspect of an existing feature
- Adds to existing test class coverage
- Extends error cases for existing functionality

### Class Organization Within File
- Group related test classes using section comments
- Order classes from simple to complex scenarios
- Order classes logically (validation → operations → execution)

## Example Test Template

```python
"""Test <feature name>"""

import pytest
from mlir_edsl import ml_function, add, mul
from mlir_edsl import i32, f32, i1
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


class Test<FeatureName>(MLIRTestBase):
    """Test <feature description>"""

    def test_basic_scenario(self):
        """Test basic <feature> functionality"""
        @ml_function
        def my_func(x: int) -> int:
            return add(x, 5)

        result = my_func(10)
        assert result == 15

    def test_edge_case(self):
        """Test edge case: <specific edge case>"""
        @ml_function
        def edge_func(x: int) -> int:
            return mul(x, 0)

        result = edge_func(42)
        assert result == 0

    def test_error_condition(self):
        """Test that <invalid operation> raises TypeError"""
        with pytest.raises(TypeError, match="error pattern"):
            @ml_function
            def bad_func(x):  # Missing type hint
                return x
```

## Migrating Legacy Test Files

Some older test files may not follow the current standard pattern. When updating these files:

### Legacy Pattern (Do Not Use)
```python
# Standalone test functions without classes
def test_something():
    """Test something"""
    result = do_something()
    assert result == expected
```

### Current Pattern (Use This)
```python
class TestFeatureName(MLIRTestBase):
    """Test feature description"""

    def test_something(self):
        """Test something"""
        result = do_something()
        assert result == expected
```

### Migration Checklist

When migrating a legacy test file:

1. ✅ **Convert to class-based tests** - Wrap all test functions in a class inheriting from MLIRTestBase
2. ✅ **Import MLIRTestBase** - Add `from tests.test_base import MLIRTestBase`
3. ✅ **Add file/class/method docstrings** - Ensure all three levels are documented
4. ✅ **Update test method signatures** - Add `self` parameter to all test methods
5. ✅ **Verify backend setup** - Remove manual backend initialization (MLIRTestBase handles this)
6. ✅ **Test the migration** - Run `pytest` to verify

**Files needing migration** (as of current status):
- `test_loops.py` - Uses standalone functions
- `test_conditionals.py` - Uses standalone functions
- Any other files with standalone `def test_*()` functions outside of classes

## Guidelines When Writing New Tests

1. **Always inherit from MLIRTestBase** - Don't create standalone test functions or classes
2. **Include all required docstrings** - File, class, and method level
3. **Use descriptive names** - Test names should explain what's being validated
4. **Test both success and failure cases** - Include error condition tests
5. **Use appropriate pytest markers** - Mark backend-dependent tests with `@pytest.mark.skipif`
6. **Follow existing patterns** - Match the style of `test_parameters.py` and `test_strict_typing.py`
7. **Keep tests focused** - One test should validate one specific behavior
8. **Add helpful comments** - Explain complex test logic or expected values

## Running Tests

### Run all tests
```bash
python3 -m pytest tests/ -v
```

### Run specific test file
```bash
python3 -m pytest tests/test_parameters.py -v
```

### Run specific test
```bash
python3 -m pytest tests/test_parameters.py::TestParameterFunctionality::test_basic_two_parameters -v
```

### Run with IR output
```bash
SAVE_IR=1 python3 -m pytest tests/test_parameters.py -v
```
