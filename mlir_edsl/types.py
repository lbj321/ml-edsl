"""Type system utilities using protobuf enums"""
from typing import TYPE_CHECKING
from mlir_edsl import ast_pb2

# Type aliases - single source of truth for type enums
I32 = ast_pb2.I32
F32 = ast_pb2.F32
I1 = ast_pb2.I1

def type_to_string(value_type):
    """Convert protobuf enum to string (for C++ boundary only)"""
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

# Python type -> MLIR type mapping (used by TypeSystem)
PYTHON_TYPE_MAP = {
    int: I32,
    float: F32,
    bool: I1,
}