"""Shared helpers for shaped (array/tensor) AST nodes.

ArrayLiteral/TensorFromElements, ArrayAccess/TensorExtract, and
ArrayStore/TensorInsert share validation logic (index/element-type
checking). Each class stays independent — own attributes, own
`_serialize_node` writing to its own proto message (the proto schema
intentionally stays unmerged) — and just calls these functions instead of
duplicating their bodies. ArrayBinaryOp and TensorEmpty have no counterpart
on the other side and don't use any of this.
"""

from ..base import Value
from ...types import ScalarType, ShapedType, i32, f32, i1


def _normalize_indices(index):
    """Convert single index or tuple to list of AST nodes"""
    from .scalars import IndexConstant

    if not isinstance(index, tuple):
        indices = (index,)
    else:
        indices = index

    result = []
    for idx in indices:
        if isinstance(idx, int):
            result.append(IndexConstant(idx))
        elif isinstance(idx, Value):
            result.append(idx)
        else:
            raise TypeError(f"Array index must be int or Value, got {type(idx)}")
    return result


def _to_scalar_node(value):
    """Convert Python literal to Constant node if needed"""
    if isinstance(value, Value):
        return value

    from .scalars import Constant

    if isinstance(value, bool):
        return Constant(value, i1)
    elif isinstance(value, int):
        return Constant(value, i32)
    elif isinstance(value, float):
        return Constant(value, f32)
    else:
        raise TypeError(f"Invalid value: {value}")


def _validate_and_flatten(elements, shape, path=""):
    """Recursively validate nested list structure and flatten to row-major order.

    Works for any number of dimensions (1D, 2D, 3D, ...).
    """
    if len(shape) == 0:
        return [elements]

    expected = shape[0]
    if not isinstance(elements, list) or len(elements) != expected:
        actual = len(elements) if isinstance(elements, list) else "non-list"
        raise TypeError(
            f"{path or 'Array'}: expected {expected} elements, got {actual}"
        )

    flat = []
    for i, elem in enumerate(elements):
        flat.extend(_validate_and_flatten(elem, shape[1:], f"{path}[{i}]"))
    return flat


def _validate_index_count(indices, container_type, is_store=False):
    """Ensure the number of indices matches the container's dimensionality.

    `container_type` is the resolved ArrayType/TensorType instance; its
    `_noun`/`_var_name` class attributes (set in mlir_edsl/types.py) drive
    the message text instead of the caller repeating them as strings.

    Used by ArrayAccess/TensorExtract (is_store=False) and ArrayStore/
    TensorInsert (is_store=True, different usage-hint style).
    """
    if len(indices) != container_type.ndim:
        noun, var_name = container_type._noun, container_type._var_name
        if is_store:
            usage = (
                f"{var_name}.at[i].set(v) for 1D, {var_name}.at[i,j].set(v) for 2D"
            )
        else:
            usage = (
                f"{var_name}[i] for 1D, {var_name}[i,j] for 2D, "
                f"{var_name}[i,j,k] for 3D"
            )
        raise TypeError(
            f"{noun} dimension mismatch: {container_type.ndim}D {noun.lower()} requires "
            f"{container_type.ndim} indices, got {len(indices)}. Usage: {usage}"
        )


def _validate_indices_are_int(indices, container_type):
    """Ensure every index is an integer scalar. Used by all 4 index-taking classes."""
    noun = container_type._noun
    for i, idx in enumerate(indices):
        idx_type = idx.infer_type()
        if not (isinstance(idx_type, ScalarType) and idx_type.is_integer()):
            raise TypeError(
                f"{noun} index {i} must be i32, got {idx_type}. "
                f"Use cast() to convert to i32."
            )


def _require_container_type(value_type, container_cls):
    """Ensure a container value has the expected shaped type. `container_cls`
    is the ArrayType/TensorType class itself (not yet known to be the right
    one — that's what we're checking), so its `_noun` is read as a class
    attribute. Used by ArrayAccess/TensorExtract only — ArrayStore/
    TensorInsert keep their own inline check since those messages already
    diverged in wording."""
    if not isinstance(value_type, container_cls):
        noun = container_cls._noun
        raise TypeError(
            f"Cannot index into non-{noun.lower()} type. "
            f"Expected {noun.lower()}, got {value_type}"
        )


def _validate_element_type(elem_type, expected_type, container_type, index):
    """Ensure a literal element matches the container's element type. Used by
    ArrayLiteral/TensorFromElements per element."""
    noun, article = container_type._noun, container_type._article
    if isinstance(elem_type, ShapedType):
        raise TypeError(
            f"{noun} element at index {index} cannot be {article} {noun.lower()}. "
            f"Nested {noun.lower()}s not supported yet."
        )

    if elem_type != expected_type:
        raise TypeError(
            f"{noun} element type mismatch at index {index}: "
            f"expected {expected_type}, "
            f"got {elem_type}. "
            f"Use cast() for explicit type conversion."
        )


def _validate_store_value_type(actual_type, expected_type, container_type):
    """Ensure a value being stored matches the container's element type. Used
    by ArrayStore/TensorInsert. `container_type._store_verb` picks "store"
    vs "insert" — TensorInsert's existing message uses "insert", not
    "store"; tests assert on that exact wording, so the verb stays
    per-type rather than unified to one word."""
    noun, verb = container_type._noun, container_type._store_verb
    if isinstance(actual_type, ShapedType):
        raise TypeError(
            f"Cannot {verb} {noun.lower()} into {noun.lower()} element. "
            f"Expected {expected_type}, got {actual_type}"
        )

    if actual_type != expected_type:
        raise TypeError(
            f"Cannot {verb} {actual_type} into "
            f"{noun}[..., {expected_type}]. "
            f"Use cast() for explicit conversion."
        )
