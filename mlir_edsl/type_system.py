"""Centralized type system for MLIR EDSL - strict type matching"""

from typing import Any, Tuple
from .types import I32, F32, I1


class TypeSystem:
    """Strict type system - operations require exact type matches"""

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
        from .types import MLIRType

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
