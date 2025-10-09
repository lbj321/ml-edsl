"""Type system utilities using protobuf enums"""
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
