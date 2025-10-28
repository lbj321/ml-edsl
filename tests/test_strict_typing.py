"""Tests for strict type checking system

This test suite validates the strict typing enforcement:
- Type hints required on all parameters and returns
- No automatic type promotion in operations
- Explicit cast() required for type conversions
- Return type validation
"""

import pytest
from mlir_edsl import ml_function, add, sub, mul, div, cast
from mlir_edsl import I32, F32, I1, i32, f32, i1
from mlir_edsl.backend import HAS_CPP_BACKEND


# ==================== TYPE HINT VALIDATION ====================

def test_missing_parameter_type_hint():
    """Test that missing parameter type hints raise TypeError"""
    with pytest.raises(TypeError, match="parameter 'x' missing type hint"):
        @ml_function
        def bad_func(x):  # Missing type hint
            return x


def test_missing_return_type_hint():
    """Test that missing return type hints raise TypeError"""
    with pytest.raises(TypeError, match="missing return type"):
        @ml_function
        def bad_func(x: int):  # Missing return type
            return x


def test_all_parameters_need_type_hints():
    """Test that all parameters need type hints"""
    with pytest.raises(TypeError, match="parameter 'y' missing type hint"):
        @ml_function
        def bad_func(x: int, y):  # y missing type hint
            return add(x, y)


# ==================== PYTHON TYPE HINTS (int, float, bool) ====================

def test_python_int_type_hint():
    """Test using Python's int type hint"""
    @ml_function
    def use_int(x: int, y: int) -> int:
        return add(x, y)

    # Validate function compiles (actual execution requires C++ backend)
    assert use_int is not None


def test_python_float_type_hint():
    """Test using Python's float type hint"""
    @ml_function
    def use_float(x: float, y: float) -> float:
        return add(x, y)

    assert use_float is not None


def test_python_bool_type_hint():
    """Test using Python's bool type hint (for I1)"""
    @ml_function
    def use_bool(condition: bool, a: int, b: int) -> int:
        from mlir_edsl import If
        return If(condition, a, b)

    assert use_bool is not None


# ==================== MLIR TYPE HINTS (i32, f32, i1) ====================

def test_mlir_i32_type_hint():
    """Test using MLIR i32 type hint"""
    @ml_function
    def use_i32(x: i32, y: i32) -> i32:
        return add(x, y)

    assert use_i32 is not None


def test_mlir_f32_type_hint():
    """Test using MLIR f32 type hint"""
    @ml_function
    def use_f32(x: f32, y: f32) -> f32:
        return add(x, y)

    assert use_f32 is not None


def test_mlir_i1_type_hint():
    """Test using MLIR i1 type hint (boolean)"""
    @ml_function
    def use_i1(condition: i1, a: i32, b: i32) -> i32:
        from mlir_edsl import If
        return If(condition, a, b)

    assert use_i1 is not None


def test_mixed_python_and_mlir_hints():
    """Test mixing Python and MLIR type hints"""
    @ml_function
    def mixed_hints(a: int, b: i32, c: float) -> f32:
        # a and b are both i32, need cast to add with c (f32)
        return add(cast(add(a, b), f32), c)

    assert mixed_hints is not None


# ==================== STRICT TYPE MATCHING ====================

def test_strict_type_mismatch_add():
    """Test that adding int and float without cast fails"""
    with pytest.raises(TypeError, match="requires matching types"):
        @ml_function
        def bad_add(x: int, y: float) -> float:
            return add(x, y)  # Should fail - no auto promotion


def test_strict_type_mismatch_sub():
    """Test that subtracting different types without cast fails"""
    with pytest.raises(TypeError, match="requires matching types"):
        @ml_function
        def bad_sub(x: i32, y: f32) -> f32:
            return sub(x, y)


def test_strict_type_mismatch_mul():
    """Test that multiplying different types without cast fails"""
    with pytest.raises(TypeError, match="requires matching types"):
        @ml_function
        def bad_mul(x: int, y: float) -> float:
            return mul(x, y)


def test_strict_type_mismatch_div():
    """Test that dividing different types without cast fails"""
    with pytest.raises(TypeError, match="requires matching types"):
        @ml_function
        def bad_div(x: i32, y: f32) -> f32:
            return div(x, y)


# ==================== EXPLICIT CAST OPERATIONS ====================

def test_cast_int_to_float():
    """Test explicit cast from int to float"""
    @ml_function
    def cast_to_float(x: int, y: float) -> float:
        return add(cast(x, f32), y)

    assert cast_to_float is not None


def test_cast_float_to_int():
    """Test explicit cast from float to int"""
    @ml_function
    def cast_to_int(x: float, y: int) -> int:
        return add(cast(x, i32), y)

    assert cast_to_int is not None


def test_cast_with_constants():
    """Test casting constants"""
    @ml_function
    def cast_constants(x: int) -> float:
        # Cast integer constant to float
        return add(cast(5, f32), cast(x, f32))

    assert cast_constants is not None


def test_nested_casts():
    """Test nested cast operations"""
    @ml_function
    def nested_casts(x: int, y: float) -> float:
        # x: i32, cast to f32, add with y (f32)
        x_float = cast(x, f32)
        result = add(x_float, y)
        return result

    assert nested_casts is not None


def test_cast_in_expression():
    """Test cast used inline in expression"""
    @ml_function
    def inline_cast(a: int, b: int, c: float) -> float:
        # (a + b) casted to float, then added to c
        return add(cast(add(a, b), f32), c)

    assert inline_cast is not None


# ==================== RETURN TYPE VALIDATION ====================

def test_return_type_mismatch_int_declared_float_returned():
    """Test that declaring i32 but returning f32 fails"""
    with pytest.raises(TypeError, match="Return type mismatch"):
        @ml_function
        def bad_return(x: float, y: float) -> int:  # Declares i32
            return add(x, y)  # Returns f32


def test_return_type_mismatch_float_declared_int_returned():
    """Test that declaring f32 but returning i32 fails"""
    with pytest.raises(TypeError, match="Return type mismatch"):
        @ml_function
        def bad_return(x: int, y: int) -> float:  # Declares f32
            return add(x, y)  # Returns i32


def test_return_type_matches():
    """Test that matching return types succeed"""
    @ml_function
    def correct_return_int(x: int, y: int) -> int:
        return add(x, y)

    @ml_function
    def correct_return_float(x: float, y: float) -> float:
        return add(x, y)

    assert correct_return_int is not None
    assert correct_return_float is not None


# ==================== COMPLEX TYPE SCENARIOS ====================

def test_complex_expression_with_casts():
    """Test complex expression requiring multiple casts"""
    @ml_function
    def complex_expr(a: int, b: int, c: float, d: float) -> float:
        # (a + b) -> i32
        int_sum = add(a, b)
        # cast to f32
        int_sum_float = cast(int_sum, f32)
        # (c * d) -> f32
        float_product = mul(c, d)
        # add them
        return add(int_sum_float, float_product)

    assert complex_expr is not None


def test_conditional_with_strict_types():
    """Test conditional (If) with strict type checking"""
    from mlir_edsl import lt, If

    @ml_function
    def conditional_strict(x: int, y: int, threshold: int) -> int:
        condition = lt(x, threshold)
        return If(condition, x, y)

    assert conditional_strict is not None


def test_loop_with_strict_types():
    """Test loops with strict type checking"""
    from mlir_edsl import For

    @ml_function
    def loop_strict(n: int) -> int:
        # For loop with all i32 values
        return For(start=0, end=n, init=0, operation="add", step=1)

    assert loop_strict is not None


# ==================== RUNTIME VALUE VALIDATION ====================

@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_runtime_value_type_mismatch():
    """Test that passing wrong runtime type raises error"""
    @ml_function
    def expects_int(x: int) -> int:
        return add(x, 5)

    # Passing float when int expected should fail validation
    with pytest.raises(TypeError, match="expects int/i32"):
        expects_int(3.14)


@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_runtime_value_type_match():
    """Test that correct runtime types work"""
    @ml_function
    def expects_int(x: int) -> int:
        return add(x, 5)

    # Passing int should work
    result = expects_int(10)
    assert result == 15


# ==================== CAST WITH BACKEND EXECUTION ====================

@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_cast_execution_int_to_float():
    """Test cast execution with C++ backend - int to float"""
    @ml_function
    def cast_exec(x: int) -> float:
        return cast(x, f32)

    result = cast_exec(42)
    assert isinstance(result, float)
    assert result == 42.0


@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_cast_execution_float_to_int():
    """Test cast execution with C++ backend - float to int"""
    @ml_function
    def cast_exec(x: float) -> int:
        return cast(x, i32)

    result = cast_exec(3.7)
    assert isinstance(result, int)
    assert result == 3  # Truncates


@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend for execution")
def test_cast_in_arithmetic_execution():
    """Test cast used in arithmetic operations"""
    @ml_function
    def mixed_arithmetic(x: int, y: float) -> float:
        # Cast x to float, then add
        return add(cast(x, f32), y)

    result = mixed_arithmetic(10, 5.5)
    assert isinstance(result, float)
    assert abs(result - 15.5) < 0.001


# ==================== ERROR MESSAGE QUALITY ====================

def test_error_message_shows_hint():
    """Test that error messages show helpful hints"""
    try:
        @ml_function
        def bad_op(x: int, y: float) -> float:
            return add(x, y)
        assert False, "Should have raised TypeError"
    except TypeError as e:
        error_msg = str(e)
        assert "Use cast()" in error_msg or "matching types" in error_msg


def test_error_message_shows_types():
    """Test that error messages show the conflicting types"""
    try:
        @ml_function
        def bad_return(x: int) -> float:
            return x  # Returns i32 but declared f32
        assert False, "Should have raised TypeError"
    except TypeError as e:
        error_msg = str(e)
        # Should mention both i32 and f32
        assert "i32" in error_msg or "int" in error_msg
        assert "f32" in error_msg or "float" in error_msg
