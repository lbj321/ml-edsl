"""MLIR type system - definitions, operations, and validation"""

import ctypes
from typing import Any, Tuple, TYPE_CHECKING
from mlir_edsl import ast_pb2

# ==================== TYPE ENUMS (Internal) ====================
# These are protobuf enums - internal use only
I32 = ast_pb2.I32
F32 = ast_pb2.F32
I1 = ast_pb2.I1

# ==================== TYPE HINTS (Public API) ====================

class MLIRType:
    """Type hint object for MLIR types"""
    def __init__(self, name: str, enum_value: int):
        self.name = name
        self.enum_value = enum_value

    def __repr__(self):
        return f"mlir_edsl.{self.name}"

    # Add this to make it work with isinstance checks in type system
    def __class_getitem__(cls, params):
        return cls

# Create type hint instances (lowercase to match MLIR syntax)
if TYPE_CHECKING:
    # For type checkers, these are type aliases
    from typing import TypeAlias
    i32: TypeAlias = MLIRType
    f32: TypeAlias = MLIRType
    i1: TypeAlias = MLIRType
else:
    # At runtime, these are MLIRType instances
    i32 = MLIRType("i32", I32)
    f32 = MLIRType("f32", F32)
    i1 = MLIRType("i1", I1)

# ==================== TYPE SYSTEM ====================

class TypeSystem:
    """Strict type system - operations require exact type matches"""

    # Python type -> MLIR type mapping
    PYTHON_TYPE_MAP = {
        int: I32,
        float: F32,
        bool: I1,
    }

    @classmethod
    def parse_type_hint(cls, hint, context: str = "parameter") -> int:
        """Parse type hint to MLIR type enum

        Supports: int, float, bool, i32, f32, i1
        """
        if isinstance(hint, MLIRType):
            return hint.enum_value

        if hint in cls.PYTHON_TYPE_MAP:
            return cls.PYTHON_TYPE_MAP[hint]

        raise TypeError(f"Invalid type hint for {context}: {hint}")

    @classmethod
    def validate_value_matches_type(cls, value: Any, type_enum: int, param_name: str):
        """Validate runtime value matches type"""
        if type_enum == I1:
            if not isinstance(value, bool):
                raise TypeError(f"Parameter '{param_name}' expects bool/i1 but got {type(value).__name__}")
        elif type_enum == I32:
            if not isinstance(value, (int, bool)):
                raise TypeError(f"Parameter '{param_name}' expects int/i32 but got {type(value).__name__}")
        elif type_enum == F32:
            if not isinstance(value, (int, float, bool)):
                raise TypeError(f"Parameter '{param_name}' expects float/f32 but got {type(value).__name__}")

    @classmethod
    def types_match(cls, inferred: int, declared: int) -> Tuple[bool, str]:
        """Check if inferred type exactly matches declared type"""
        if inferred == declared:
            return True, ""

        return False, (
            f"Type mismatch:\n"
            f"  Declared: {cls.type_name(declared)}\n"
            f"  Inferred: {cls.type_name(inferred)}\n"
            f"  Hint: Change return type to {cls.type_name(inferred)} or add explicit cast"
        )

    @classmethod
    def type_name(cls, type_enum: int) -> str:
        """Get MLIR type name from enum"""
        return {I32: "i32", F32: "f32", I1: "i1"}.get(type_enum, f"unknown({type_enum})")

# ==================== UTILITIES ====================

def is_numeric_type(ty: int) -> bool:
    """Check if type is numeric (integer or float)

    Args:
        ty: Type enum value to check

    Returns:
        True if type is I32 or F32

    Example:
        >>> is_numeric_type(I32)
        True
        >>> is_numeric_type(F32)
        True
        >>> is_numeric_type(I1)
        False
    """
    return ty in (I32, F32)

def is_integer_type(ty: int) -> bool:
    """Check if type is integer

    Args:
        ty: Type enum value to check

    Returns:
        True if type is I32
    """
    return ty == I32

def is_float_type(ty: int) -> bool:
    """Check if type is float

    Args:
        ty: Type enum value to check

    Returns:
        True if type is F32
    """
    return ty == F32

def type_to_string(value_type):
    """Convert type to string (for C++ boundary only)
    """

    # Handle enum values
    if value_type == I32:
        return "i32"
    elif value_type == F32:
        return "f32"
    elif value_type == I1:
        return "i1"
    else:
        raise ValueError(f"Unknown type: {value_type}")

def type_from_string(type_str):
    """Convert string to protobuf enum (for C++ boundary/legacy)"""
    if type_str == "i32":
        return I32
    elif type_str == "f32":
        return F32
    elif type_str == "i1":
        return I1
    else:
        raise ValueError(f"Unknown type string: {type_str}")

# ==================== BACKEND MAPPINGS ====================

# Type mapping for ctypes (JIT execution)
TYPE_TO_CTYPES = {
    I32: ctypes.c_int32,
    F32: ctypes.c_float,
    I1: ctypes.c_bool,
}
