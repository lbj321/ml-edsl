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

class ScalarType:
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
    i32: TypeAlias = ScalarType
    f32: TypeAlias = ScalarType
    i1: TypeAlias = ScalarType
else:
    # At runtime, these are MLIRType instances
    i32 = ScalarType("i32", I32)
    f32 = ScalarType("f32", F32)
    i1 = ScalarType("i1", I1)

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
    def parse_type_hint(cls, hint, context: str = "parameter"):
        """Parse type hint to MLIR type enum OR ArrayType

        Supports: int, float, bool, i32, f32, i1, Array[N, dtype]

        Returns:
            - For scalars: int (protobuf enum)
            - For arrays: ArrayType instance
        """
        # Handle ScalarType instances (i32, f32, i1)
        if isinstance(hint, ScalarType):
            return hint.enum_value

        # Handle ArrayType instances (Array[10, i32])
        if isinstance(hint, ArrayType):
            return hint  # Return the ArrayType directly, not an enum

        # Handle Python built-in types (int, float, bool)
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

# ==================== ARRAY TYPES ====================

class ArrayType:
    """
    Represents a fixed-size array type: memref<NxT> (1D), memref<MxNxT> (2D), memref<MxNxPxT> (3D)

    Used for type hints in function signatures.

    Examples:
        def foo(arr: Array[10, i32]) -> Array[10, i32]:  # 1D array
            ...

        def bar(matrix: Array[2, 3, i32]) -> i32:  # 2D array
            ...
    """

    def __init__(self, shape, element_type: ScalarType):
        # Normalize shape to tuple
        if isinstance(shape, int):
            self.shape = (shape,)  # 1D
        elif isinstance(shape, tuple):
            self.shape = tuple(shape)  # 2D/3D
        else:
            raise TypeError(f"Array shape must be int or tuple of ints, got {type(shape).__name__}")

        # Validate all dimensions
        if not all(isinstance(d, int) and d > 0 for d in self.shape):
            raise TypeError(f"All dimensions must be positive integers, got {self.shape}")

        # Validate dimensionality (only 1D, 2D, 3D)
        if len(self.shape) == 0 or len(self.shape) > 3:
            raise TypeError(f"Only 1D, 2D, and 3D arrays supported, got {len(self.shape)}D")

        if not isinstance(element_type, ScalarType):
            raise TypeError(
                f"Array element type must be ScalarType (i32, f32, i1), got {element_type}"
            )

        self.element_type = element_type  # This is a ScalarType instance (i32, f32, i1)
        self.element_enum = element_type.enum_value  # Store the protobuf enum too

    @property
    def size(self) -> int:
        """
        Backward compatibility: Return size for 1D arrays only.

        Raises AttributeError for multi-dimensional arrays.
        Use .shape for multi-dimensional arrays.
        """
        if len(self.shape) != 1:
            raise AttributeError(
                f"'.size' only available for 1D arrays. "
                f"This is a {len(self.shape)}D array with shape {self.shape}. "
                f"Use '.shape' instead."
            )
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions (1, 2, or 3)"""
        return len(self.shape)

    @property
    def total_elements(self) -> int:
        """Total number of elements (product of all dimensions)"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def __call__(self, elements: list):
        """
        Enable Array[4, i32]([1, 2, 3, 4]) construction syntax.

        This creates an ArrayLiteral AST node (will be implemented in Step 2).
        For now, we'll just validate and prepare for later.
        """
        # Import here to avoid circular dependency
        # (We'll implement ArrayLiteral in Step 2)
        try:
            from .ast import ArrayLiteral
            return ArrayLiteral(elements, self)
        except (ImportError, AttributeError):
            # During initial implementation, ast.ArrayLiteral doesn't exist yet
            raise NotImplementedError(
                "ArrayLiteral not yet implemented. "
                "This will be added in Step 2 (AST extensions)."
            )

    def to_mlir_string(self) -> str:
        """Convert to MLIR type string: memref<10xi32>, memref<2x3xi32>, etc."""
        elem_name = self.element_type.name  # "i32", "f32", or "i1"
        # 1D: memref<10xi32>
        # 2D: memref<2x3xi32>
        # 3D: memref<2x3x4xi32>
        dims = 'x'.join(str(d) for d in self.shape)
        return f"memref<{dims}x{elem_name}>"

    def __repr__(self):
        if len(self.shape) == 1:
            return f"Array[{self.shape[0]}, {self.element_type.name}]"
        else:
            # 2D: Array[2, 3, i32]
            # 3D: Array[2, 3, 4, i32]
            dims = ', '.join(str(d) for d in self.shape)
            return f"Array[{dims}, {self.element_type.name}]"

    def __eq__(self, other):
        """Enable type equality checking for strict type system"""
        if not isinstance(other, ArrayType):
            return False
        return (self.shape == other.shape and  # Tuple comparison
                self.element_enum == other.element_enum)

    def __hash__(self):
        """Enable use as dict key"""
        return hash((self.shape, self.element_enum))  # Hash tuple


class ArrayMeta(type):
    """
    Metaclass to enable Array[size, dtype] or Array[M, N, dtype] subscript syntax.

    This makes Array[10, i32] and Array[2, 3, i32] work even though Array is a class.
    """

    def __getitem__(cls, params):
        """
        Handle Array[size, dtype] or Array[M, N, dtype] or Array[M, N, P, dtype] syntax.

        Args:
            params: Tuple where last element is dtype, preceding are dimensions

        Returns:
            ArrayType instance

        Examples:
            Array[10, i32]        -> 1D array
            Array[2, 3, i32]      -> 2D array
            Array[2, 3, 4, f32]   -> 3D array
        """
        if not isinstance(params, tuple):
            raise TypeError(
                f"Array requires parameters: Array[size, dtype] or Array[M, N, dtype]. "
                f"Example: Array[10, i32] or Array[2, 3, i32]"
            )

        if len(params) < 2:
            raise TypeError(
                f"Array requires at least 2 parameters (dimensions + dtype), got {len(params)}"
            )

        # Last parameter must be dtype
        dtype = params[-1]
        dims = params[:-1]

        # Validate dtype is ScalarType
        if not isinstance(dtype, ScalarType):
            raise TypeError(
                f"Last parameter must be element type (i32, f32, i1), got {dtype!r}"
            )

        # Validate dimensionality (only 1D, 2D, 3D)
        if len(dims) > 3:
            raise TypeError(
                f"Only 1D, 2D, and 3D arrays supported, got {len(dims)}D. "
                f"Usage: Array[N, dtype] or Array[M, N, dtype] or Array[M, N, P, dtype]"
            )

        # Validate all dimensions are positive integers
        for i, dim in enumerate(dims):
            if not isinstance(dim, int) or dim <= 0:
                raise TypeError(
                    f"Dimension {i} must be positive integer, got {dim!r}"
                )

        # Create ArrayType with shape tuple
        # For 1D: dims=(10,) -> ArrayType receives int 10
        # For 2D/3D: dims=(2,3) or (2,3,4) -> ArrayType receives tuple
        if len(dims) == 1:
            return ArrayType(dims[0], dtype)  # Backward compat: pass int for 1D
        else:
            return ArrayType(dims, dtype)     # Pass tuple for 2D/3D


class Array(metaclass=ArrayMeta):
    """
    Fixed-size array type for memref dialect.

    Usage as type hint:
        def foo(arr: Array[10, i32]) -> Array[10, i32]:
            ...

    Usage for construction (inside @ml_function):
        arr = Array[4, i32]([1, 2, 3, 4])

    The Array class itself is never instantiated - it's just a namespace
    for the subscript syntax enabled by ArrayMeta.
    """
    pass
