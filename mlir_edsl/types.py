"""MLIR type system - algebraic type hierarchy with protobuf serialization"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, TYPE_CHECKING
from mlir_edsl import ast_pb2

# ============================================================================
# TYPE BASE CLASS (Algebraic Type System)
# ============================================================================

class Type(ABC):
    """Base class for algebraic type system.

    All types support:
    - Category predicates: is_scalar(), is_aggregate()
    - Property predicates: is_numeric(), is_integer(), is_float(), is_boolean()
    - Cast checking: can_cast_to()
    - Serialization: to_proto() -> TypeSpec
    """

    @abstractmethod
    def is_scalar(self) -> bool:
        """Returns True if this is a scalar type (i32, f32, i1)"""
        pass

    @abstractmethod
    def is_aggregate(self) -> bool:
        """Returns True if this is an aggregate type (memref, tensor)"""
        pass

    @abstractmethod
    def is_numeric(self) -> bool:
        """Returns True if supports arithmetic operations"""
        pass

    @abstractmethod
    def is_integer(self) -> bool:
        """Returns True if this is an integer type"""
        pass

    @abstractmethod
    def is_float(self) -> bool:
        """Returns True if this is a floating-point type"""
        pass

    @abstractmethod
    def is_boolean(self) -> bool:
        """Returns True if this is a boolean type"""
        pass

    def can_cast_to(self, target: 'Type') -> bool:
        """Returns True if this type can be cast to target type.

        Default: only scalar-to-scalar casts allowed.
        """
        return self.is_scalar() and target.is_scalar()

    @abstractmethod
    def to_proto(self) -> ast_pb2.TypeSpec: # pyright: ignore[reportInvalidTypeForm]
        """Convert to protobuf TypeSpec"""
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass


# ============================================================================
# SCALAR TYPE
# ============================================================================

class ScalarType(Type):
    """Scalar type backed by protobuf ScalarTypeSpec.Kind.

    The protobuf schema is the single source of truth for type definitions.
    """

    # Constants from schema (single source of truth)
    I32 = ast_pb2.ScalarTypeSpec.I32
    F32 = ast_pb2.ScalarTypeSpec.F32
    I1 = ast_pb2.ScalarTypeSpec.I1

    # Semantic groupings derived from schema
    _NUMERIC_KINDS = frozenset({I32, F32})
    _INTEGER_KINDS = frozenset({I32})
    _FLOAT_KINDS = frozenset({F32})
    _BOOLEAN_KINDS = frozenset({I1})

    _KIND_TO_NAME = {I32: 'i32', F32: 'f32', I1: 'i1'}
    _ALL_KINDS = frozenset(_KIND_TO_NAME.keys())

    def __init__(self, kind: int):
        """Initialize from ScalarTypeSpec.Kind enum value.

        Args:
            kind: One of ScalarType.I32, ScalarType.F32, ScalarType.I1
        """
        if kind not in self._ALL_KINDS:
            raise ValueError(f"Unknown scalar kind: {kind}")
        self.kind = kind

    @property
    def name(self) -> str:
        """MLIR type name (i32, f32, i1)"""
        return self._KIND_TO_NAME[self.kind]

    @property
    def enum_value(self) -> int:
        """Backward compatibility: returns the protobuf enum value"""
        return self.kind

    # Category predicates
    def is_scalar(self) -> bool:
        return True

    def is_aggregate(self) -> bool:
        return False

    # Property predicates
    def is_numeric(self) -> bool:
        return self.kind in self._NUMERIC_KINDS

    def is_integer(self) -> bool:
        return self.kind in self._INTEGER_KINDS

    def is_float(self) -> bool:
        return self.kind in self._FLOAT_KINDS

    def is_boolean(self) -> bool:
        return self.kind in self._BOOLEAN_KINDS

    # Serialization
    def to_proto(self) -> ast_pb2.TypeSpec: # pyright: ignore[reportInvalidTypeForm]
        ts = ast_pb2.TypeSpec()
        ts.scalar.kind = self.kind
        return ts

    # Equality and hashing
    def __eq__(self, other) -> bool:
        if isinstance(other, ScalarType):
            return self.kind == other.kind
        return False

    def __hash__(self) -> int:
        return hash(self.kind)

    def __repr__(self) -> str:
        return self.name

    # Support Array[size, dtype] syntax
    def __class_getitem__(cls, params):
        return cls


# ============================================================================
# SINGLETON SCALAR TYPE INSTANCES
# ============================================================================

i32 = ScalarType(ScalarType.I32)
f32 = ScalarType(ScalarType.F32)
i1 = ScalarType(ScalarType.I1)

# Dynamic dimension sentinel (matches MLIR's ShapedType::kDynamic)
DYN = -1

# All scalar types - add new types here
SCALAR_TYPES = (i32, f32, i1)

# Derived: string name -> instance (for type hint resolution)
TYPE_HINT_NAMESPACE = {t.name: t for t in SCALAR_TYPES}

# Python type -> MLIR type (semantic mapping)
PYTHON_TO_MLIR = {
    int: i32,
    float: f32,
    bool: i1,
}


# ============================================================================
# ARRAY TYPE (MemRef)
# ============================================================================

class ArrayType(Type):
    """Fixed-size array type: memref<NxT> (1D), memref<MxNxT> (2D), memref<MxNxPxT> (3D)

    Used for type hints in function signatures and array construction.

    Examples:
        def foo(arr: Array[i32, 10]) -> i32:  # 1D array parameter
            ...

        arr = Array[f32, 2, 3]([...])  # 2D array literal
    """

    def __init__(self, shape, element_type: ScalarType):
        """Initialize array type.

        Args:
            shape: int for 1D, tuple for 2D/3D
            element_type: ScalarType instance (i32, f32, i1)
        """
        # Normalize shape to tuple
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = tuple(shape)
        else:
            raise TypeError(f"Array shape must be int or tuple, got {type(shape).__name__}")

        # Validate dimensions
        if not all(isinstance(d, int) and d > 0 for d in self.shape):
            raise TypeError(f"All dimensions must be positive integers, got {self.shape}")

        # Validate dimensionality (1D, 2D, 3D only)
        if len(self.shape) == 0 or len(self.shape) > 3:
            raise TypeError(f"Only 1D, 2D, and 3D arrays supported, got {len(self.shape)}D")

        # Validate element type
        if not isinstance(element_type, ScalarType):
            raise TypeError(f"element_type must be ScalarType (i32, f32, i1), got {element_type}")

        self.element_type = element_type

    @property
    def size(self) -> int:
        """For 1D arrays only: returns the size.

        Raises AttributeError for multi-dimensional arrays.
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

    # Category predicates
    def is_scalar(self) -> bool:
        return False

    def is_aggregate(self) -> bool:
        return True

    # Property predicates (delegate to element type)
    def is_numeric(self) -> bool:
        return self.element_type.is_numeric()

    def is_integer(self) -> bool:
        return self.element_type.is_integer()

    def is_float(self) -> bool:
        return self.element_type.is_float()

    def is_boolean(self) -> bool:
        return self.element_type.is_boolean()

    def can_cast_to(self, target: Type) -> bool:
        """Arrays cannot be cast"""
        return False

    # Serialization
    def to_proto(self) -> ast_pb2.TypeSpec: # pyright: ignore[reportInvalidTypeForm]
        ts = ast_pb2.TypeSpec()
        ts.memref.shape.extend(self.shape)
        ts.memref.element_type.CopyFrom(self.element_type.to_proto())
        return ts

    def to_mlir_string(self) -> str:
        """Convert to MLIR type string: memref<10xi32>, memref<2x3xf32>, etc."""
        dims = 'x'.join(str(d) for d in self.shape)
        return f"memref<{dims}x{self.element_type.name}>"

    # Equality and hashing
    def __eq__(self, other) -> bool:
        if not isinstance(other, ArrayType):
            return False
        return self.shape == other.shape and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash((self.shape, self.element_type))

    def __repr__(self) -> str:
        dims = ', '.join(str(d) for d in self.shape)
        return f"Array[{self.element_type.name}, {dims}]"

    def __call__(self, elements: list):
        """Enable Array[i32, 4]([1, 2, 3, 4]) construction syntax."""
        from .ast import ArrayLiteral
        return ArrayLiteral(elements, self)


# ============================================================================
# TENSOR TYPE (Value-semantic)
# ============================================================================

class TensorType(Type):
    """Value-semantic tensor type: tensor<NxT> (1D), tensor<MxNxT> (2D), tensor<MxNxPxT> (3D)

    Unlike ArrayType (memref), tensors are immutable. Operations produce
    new tensors rather than mutating in place.

    Examples:
        t = Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])
        val = t[2]  # Extract element
    """

    def __init__(self, shape, element_type: ScalarType):
        """Initialize tensor type.

        Args:
            shape: int for 1D, tuple for 2D/3D
            element_type: ScalarType instance (i32, f32, i1)
        """
        # Normalize shape to tuple
        if isinstance(shape, int):
            self.shape = (shape,)
        elif isinstance(shape, tuple):
            self.shape = tuple(shape)
        else:
            raise TypeError(f"Tensor shape must be int or tuple, got {type(shape).__name__}")

        # Validate dimensions
        if not all(isinstance(d, int) and (d > 0 or d == DYN) for d in self.shape):
            raise TypeError(f"All dimensions must be positive integers or DYN, got {self.shape}")

        # Validate dimensionality (1D, 2D, 3D only)
        if len(self.shape) == 0 or len(self.shape) > 3:
            raise TypeError(f"Only 1D, 2D, and 3D tensors supported, got {len(self.shape)}D")

        # Validate element type
        if not isinstance(element_type, ScalarType):
            raise TypeError(f"element_type must be ScalarType (i32, f32, i1), got {element_type}")

        self.element_type = element_type

    @property
    def size(self) -> int:
        """For 1D tensors only: returns the size."""
        if len(self.shape) != 1:
            raise AttributeError(
                f"'.size' only available for 1D tensors. "
                f"This is a {len(self.shape)}D tensor with shape {self.shape}. "
                f"Use '.shape' instead."
            )
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions (1, 2, or 3)."""
        return len(self.shape)

    @property
    def is_dynamic(self) -> bool:
        """True if any dimension is dynamic."""
        return DYN in self.shape

    @property
    def total_elements(self) -> int:
        """Total number of elements (product of all dimensions)."""
        if self.is_dynamic:
            raise ValueError("Cannot compute total_elements for dynamic tensor")
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    # Category predicates
    def is_scalar(self) -> bool:
        return False

    def is_aggregate(self) -> bool:
        return True

    # Property predicates (delegate to element type)
    def is_numeric(self) -> bool:
        return self.element_type.is_numeric()

    def is_integer(self) -> bool:
        return self.element_type.is_integer()

    def is_float(self) -> bool:
        return self.element_type.is_float()

    def is_boolean(self) -> bool:
        return self.element_type.is_boolean()

    def can_cast_to(self, target: Type) -> bool:
        """Tensors cannot be cast."""
        return False

    # Serialization
    def to_proto(self) -> ast_pb2.TypeSpec: # pyright: ignore[reportInvalidTypeForm]
        ts = ast_pb2.TypeSpec()
        ts.tensor.shape.extend(self.shape)
        ts.tensor.element_type.CopyFrom(self.element_type.to_proto())
        return ts

    def to_mlir_string(self) -> str:
        """Convert to MLIR type string: tensor<4xf32>, tensor<2x3xi32>, etc."""
        dims = 'x'.join('?' if d == DYN else str(d) for d in self.shape)
        return f"tensor<{dims}x{self.element_type.name}>"

    # Equality and hashing
    def __eq__(self, other) -> bool:
        if not isinstance(other, TensorType):
            return False
        return self.shape == other.shape and self.element_type == other.element_type

    def __hash__(self) -> int:
        return hash(('tensor', self.shape, self.element_type))

    def __repr__(self) -> str:
        dims = ', '.join('DYN' if d == DYN else str(d) for d in self.shape)
        return f"Tensor[{self.element_type.name}, {dims}]"

    def __call__(self, elements: list):
        """Enable Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0]) construction syntax."""
        from .ast import TensorFromElements
        return TensorFromElements(elements, self)


# ============================================================================
# ARRAY SUBSCRIPT SYNTAX (Array[dtype, N] or Array[dtype, M, N])
# ============================================================================

class ArrayMeta(type):
    """Metaclass to enable Array[dtype, size] subscript syntax."""

    def __getitem__(cls, params):
        """Handle Array[dtype, size] or Array[dtype, M, N] syntax.

        Args:
            params: Tuple where first element is dtype, rest are dimensions

        Returns:
            ArrayType instance
        """
        if not isinstance(params, tuple):
            raise TypeError(
                f"Array requires parameters: Array[dtype, size] or Array[dtype, M, N]. "
                f"Example: Array[i32, 10]"
            )

        if len(params) < 2:
            raise TypeError(
                f"Array requires at least 2 parameters (dtype + dimensions), got {len(params)}"
            )

        # First parameter is dtype, rest are dimensions
        dtype = params[0]
        dims = params[1:]

        # Validate dtype
        if not isinstance(dtype, ScalarType):
            raise TypeError(
                f"First parameter must be element type (i32, f32, i1), got {dtype!r}"
            )

        # Validate dimensionality
        if len(dims) > 3:
            raise TypeError(
                f"Only 1D, 2D, and 3D arrays supported, got {len(dims)}D"
            )

        # Validate dimensions are positive integers
        for i, dim in enumerate(dims):
            if not isinstance(dim, int) or dim <= 0:
                raise TypeError(f"Dimension {i} must be positive integer, got {dim!r}")

        # Create ArrayType
        if len(dims) == 1:
            return ArrayType(dims[0], dtype)
        else:
            return ArrayType(dims, dtype)


class Array(metaclass=ArrayMeta):
    """Fixed-size array type for memref dialect.

    Usage as type hint:
        def foo(arr: Array[i32, 10]) -> i32:
            ...

    Usage for construction (inside @ml_function):
        arr = Array[i32, 4]([1, 2, 3, 4])
    """
    pass


# ============================================================================
# TENSOR SUBSCRIPT SYNTAX (Tensor[dtype, N] or Tensor[dtype, M, N])
# ============================================================================

class TensorMeta(type):
    """Metaclass to enable Tensor[dtype, size] subscript syntax."""

    def __getitem__(cls, params):
        """Handle Tensor[dtype, size] or Tensor[dtype, M, N] syntax.

        Args:
            params: Tuple where first element is dtype, rest are dimensions

        Returns:
            TensorType instance
        """
        if not isinstance(params, tuple):
            raise TypeError(
                f"Tensor requires parameters: Tensor[dtype, size] or Tensor[dtype, M, N]. "
                f"Example: Tensor[f32, 4]"
            )

        if len(params) < 2:
            raise TypeError(
                f"Tensor requires at least 2 parameters (dtype + dimensions), got {len(params)}"
            )

        # First parameter is dtype, rest are dimensions
        dtype = params[0]
        dims = params[1:]

        # Validate dtype
        if not isinstance(dtype, ScalarType):
            raise TypeError(
                f"First parameter must be element type (i32, f32, i1), got {dtype!r}"
            )

        # Validate dimensionality
        if len(dims) > 3:
            raise TypeError(
                f"Only 1D, 2D, and 3D tensors supported, got {len(dims)}D"
            )

        # Validate dimensions are positive integers or DYN
        for i, dim in enumerate(dims):
            if not isinstance(dim, int) or (dim <= 0 and dim != DYN):
                raise TypeError(f"Dimension {i} must be positive integer or DYN, got {dim!r}")

        # Create TensorType
        if len(dims) == 1:
            return TensorType(dims[0], dtype)
        else:
            return TensorType(dims, dtype)


class Tensor(metaclass=TensorMeta):
    """Value-semantic tensor type for tensor dialect.

    Usage for construction (inside @ml_function):
        t = Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])
        val = t[2]  # Extract element

    Usage for empty tensor:
        t = Tensor.empty(f32, 4)
        t = Tensor.empty(i32, 2, 3)
    """

    @staticmethod
    def empty(dtype, *shape):
        """Create an uninitialized tensor of the given shape and element type.

        Args:
            dtype: Element type (i32, f32, etc.)
            *shape: Dimension sizes — integers for static, Value nodes for dynamic.
                    e.g., (4,) for static 1D, (n,) where n is a Value for dynamic 1D.

        Returns:
            TensorEmpty AST node
        """
        from .ast import TensorEmpty, Value

        # Separate static shape (with DYN markers) from dynamic Value operands
        static_shape = []
        dynamic_dims = []
        for dim in shape:
            if isinstance(dim, int):
                static_shape.append(dim)
            elif isinstance(dim, Value):
                static_shape.append(DYN)
                dynamic_dims.append(dim)
            else:
                raise TypeError(
                    f"Tensor.empty() dimensions must be int or Value, got {type(dim).__name__}"
                )

        if len(static_shape) == 1:
            tensor_type = TensorType(static_shape[0], dtype)
        else:
            tensor_type = TensorType(tuple(static_shape), dtype)
        return TensorEmpty(tensor_type, dynamic_dims)


# ============================================================================
# TYPE SYSTEM UTILITIES
# ============================================================================

class TypeSystem:
    """Type validation and parsing utilities"""

    @classmethod
    def parse_type_hint(cls, hint, context: str = "parameter") -> Type:
        """Parse type hint to Type object.

        Supports: int, float, bool, i32, f32, i1, Array[dtype, N], Tensor[dtype, N]

        Returns:
            Type instance (ScalarType or ArrayType)
        """
        # Handle Type instances directly
        if isinstance(hint, Type):
            return hint

        # Handle Python built-in types
        if hint in PYTHON_TO_MLIR:
            return PYTHON_TO_MLIR[hint]

        raise TypeError(f"Invalid type hint for {context}: {hint}")

    @classmethod
    def validate_value_matches_type(cls, value: Any, type_spec: Type, param_name: str):
        """Validate runtime value matches type.

        Args:
            value: Runtime value to validate
            type_spec: Type instance
            param_name: Parameter name for error messages
        """
        if isinstance(type_spec, ArrayType):
            _is_ndarray = hasattr(value, 'shape') and hasattr(value, 'dtype')
            if _is_ndarray:
                if tuple(value.shape) != tuple(type_spec.shape):
                    raise ValueError(
                        f"Parameter '{param_name}': expected shape {type_spec.shape}, "
                        f"got {tuple(value.shape)}"
                    )
            elif isinstance(value, list):
                cls._check_nested_shape(value, type_spec.shape, param_name)
            else:
                raise TypeError(
                    f"Parameter '{param_name}': expected list or ndarray for {type_spec}, "
                    f"got {type(value).__name__}"
                )
            return

        if isinstance(type_spec, TensorType):
            _is_ndarray = hasattr(value, 'shape') and hasattr(value, 'dtype')
            if _is_ndarray:
                if not type_spec.is_dynamic and tuple(value.shape) != tuple(type_spec.shape):
                    raise ValueError(
                        f"Parameter '{param_name}': expected shape {type_spec.shape}, "
                        f"got {tuple(value.shape)}"
                    )
            elif isinstance(value, list):
                if not type_spec.is_dynamic:
                    cls._check_nested_shape(value, type_spec.shape, param_name)
                else:
                    cls._check_nested_ndim(value, type_spec.ndim, param_name)
            else:
                raise TypeError(
                    f"Parameter '{param_name}': expected list or ndarray for {type_spec}, "
                    f"got {type(value).__name__}"
                )
            return

        # Scalar validation
        if type_spec.is_boolean():
            if not isinstance(value, bool):
                raise TypeError(f"Parameter '{param_name}' expects bool/i1 but got {type(value).__name__}")
        elif type_spec.is_integer():
            if not isinstance(value, (int, bool)):
                raise TypeError(f"Parameter '{param_name}' expects int/i32 but got {type(value).__name__}")
        elif type_spec.is_float():
            if not isinstance(value, (int, float, bool)):
                raise TypeError(f"Parameter '{param_name}' expects float/f32 but got {type(value).__name__}")
        else:
            raise TypeError(f"Parameter '{param_name}': unknown scalar type {type_spec}")

    @classmethod
    def _check_nested_shape(cls, data: list, shape: tuple, param_name: str):
        """Recursively verify data has the expected shape."""
        if not isinstance(data, list) or len(data) != shape[0]:
            got = len(data) if isinstance(data, list) else f"non-list ({type(data).__name__})"
            raise ValueError(
                f"Parameter '{param_name}': expected {shape[0]} elements, got {got}"
            )
        if len(shape) > 1:
            for i, item in enumerate(data):
                cls._check_nested_shape(item, shape[1:], f"{param_name}[{i}]")

    @classmethod
    def _check_nested_ndim(cls, data: list, ndim: int, param_name: str):
        """Verify data has the right number of dimensions (for dynamic tensors)."""
        if not isinstance(data, list):
            raise ValueError(
                f"Parameter '{param_name}': expected list, got {type(data).__name__}"
            )
        if ndim > 1:
            for i, item in enumerate(data):
                cls._check_nested_ndim(item, ndim - 1, f"{param_name}[{i}]")
        
    @classmethod
    def types_match(cls, inferred: Type, declared: Type) -> Tuple[bool, str]:
        """Check if inferred type matches declared type.

        Returns:
            (matches: bool, error_message: str)
        """
        if inferred == declared:
            return True, ""

        # Different type categories
        if inferred.is_scalar() != declared.is_scalar():
            return False, (
                f"Type category mismatch:\n"
                f"  Declared: {declared}\n"
                f"  Inferred: {inferred}\n"
                f"  Hint: Cannot mix scalar and array types"
            )

        # Same category but different types
        if inferred.is_scalar():
            return False, (
                f"Type mismatch:\n"
                f"  Declared: {declared}\n"
                f"  Inferred: {inferred}\n"
                f"  Hint: Change return type or add explicit cast"
            )
        else:
            return False, (
                f"Array type mismatch:\n"
                f"  Declared: {declared}\n"
                f"  Inferred: {inferred}\n"
                f"  Hint: Ensure array shapes and element types match"
            )


