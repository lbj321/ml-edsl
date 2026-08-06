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
        """Returns True for shaped types (memref/tensor).

        Aggregate types are lowered to struct descriptors and returned via
        caller-allocated output parameter at the JIT ABI boundary.
        """
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
# SHAPED TYPE (shared base for ArrayType/TensorType)
# ============================================================================

class ShapedType(Type):
    """Shared base for shaped (memref/tensor) types.

    Hoists everything ArrayType and TensorType have in common: shape
    normalization/validation, the numeric/integer/float/boolean delegation
    to element_type, serialization, and equality/hashing/repr. Subclasses
    customize behavior purely via class attributes:
        _noun          - display name for error messages ("Array"/"Tensor")
        _kind          - ast_pb2.ShapedTypeSpec.MEMREF / .TENSOR
        _mlir_prefix   - MLIR type string prefix ("memref"/"tensor")
        _hash_tag      - extra discriminator mixed into __hash__, or None
        _dyn_in_repr   - if True, DYN dims render as "DYN" in __repr__
        _article       - indefinite article for _noun ("an"/"a"), used by AST
                         node error messages (see mlir_edsl/ast/nodes/shaped.py)
        _var_name      - example variable name for AST usage hints ("arr"/"t")
        _store_verb    - verb used in AST store/insert error messages
                         ("store"/"insert")
    """

    _noun = ""
    _kind = None
    _mlir_prefix = ""
    _hash_tag = None
    _dyn_in_repr = False
    _article = ""
    _var_name = ""
    _store_verb = ""

    def __init__(self, shape, element_type: ScalarType):
        """Initialize shaped type.

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
            raise TypeError(f"{self._noun} shape must be int or tuple, got {type(shape).__name__}")

        # Validate dimensions
        if not all(isinstance(d, int) and (d > 0 or d == DYN) for d in self.shape):
            raise TypeError(f"All dimensions must be positive integers or DYN, got {self.shape}")

        # Validate dimensionality (1D, 2D, 3D only)
        if len(self.shape) == 0 or len(self.shape) > 3:
            raise TypeError(f"Only 1D, 2D, and 3D {self._noun.lower()}s supported, got {len(self.shape)}D")

        # Validate element type
        if not isinstance(element_type, ScalarType):
            raise TypeError(f"element_type must be ScalarType (i32, f32, i1), got {element_type}")

        self.element_type = element_type

    @property
    def size(self) -> int:
        """For 1D shapes only: returns the size.

        Raises AttributeError for multi-dimensional shapes.
        """
        if len(self.shape) != 1:
            raise AttributeError(
                f"'.size' only available for 1D {self._noun.lower()}s. "
                f"This is a {len(self.shape)}D {self._noun.lower()} with shape {self.shape}. "
                f"Use '.shape' instead."
            )
        return self.shape[0]

    @property
    def ndim(self) -> int:
        """Number of dimensions (1, 2, or 3)"""
        return len(self.shape)

    @property
    def is_dynamic(self) -> bool:
        """True if any dimension is dynamic."""
        return DYN in self.shape

    @property
    def total_elements(self) -> int:
        """Total number of elements (product of all dimensions)"""
        if self.is_dynamic:
            raise ValueError(f"Cannot compute total_elements for dynamic {self._noun.lower()}")
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
        """Shaped types cannot be cast"""
        return False

    # Serialization
    def to_proto(self) -> ast_pb2.TypeSpec: # pyright: ignore[reportInvalidTypeForm]
        ts = ast_pb2.TypeSpec()
        ts.shaped.kind = self._kind
        ts.shaped.shape.extend(self.shape)
        ts.shaped.element_type.CopyFrom(self.element_type.to_proto())
        return ts

    def to_mlir_string(self) -> str:
        """Convert to MLIR type string: memref<10xi32>, tensor<4xf32>, etc."""
        dims = 'x'.join('?' if d == DYN else str(d) for d in self.shape)
        return f"{self._mlir_prefix}<{dims}x{self.element_type.name}>"

    # Equality and hashing
    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.shape == other.shape and self.element_type == other.element_type

    def __hash__(self) -> int:
        if self._hash_tag is not None:
            return hash((self._hash_tag, self.shape, self.element_type))
        return hash((self.shape, self.element_type))

    def __repr__(self) -> str:
        if self._dyn_in_repr:
            dims = ', '.join('DYN' if d == DYN else str(d) for d in self.shape)
        else:
            dims = ', '.join(str(d) for d in self.shape)
        return f"{self._noun}[{self.element_type.name}, {dims}]"


# ============================================================================
# ARRAY TYPE (MemRef)
# ============================================================================

class ArrayType(ShapedType):
    """Fixed-size array type: memref<NxT> (1D), memref<MxNxT> (2D), memref<MxNxPxT> (3D)

    Used for type hints in function signatures and array construction.

    Examples:
        def foo(arr: Array[i32, 10]) -> i32:  # 1D array parameter
            ...

        arr = Array[f32, 2, 3]([...])  # 2D array literal
    """

    _noun = "Array"
    _kind = ast_pb2.ShapedTypeSpec.MEMREF
    _mlir_prefix = "memref"
    _article = "an"
    _var_name = "arr"
    _store_verb = "store"

    def __call__(self, elements: list):
        """Enable Array[i32, 4]([1, 2, 3, 4]) construction syntax."""
        from .ast import ArrayLiteral
        return ArrayLiteral(elements, self)


# ============================================================================
# TENSOR TYPE (Value-semantic)
# ============================================================================

class TensorType(ShapedType):
    """Value-semantic tensor type: tensor<NxT> (1D), tensor<MxNxT> (2D), tensor<MxNxPxT> (3D)

    Unlike ArrayType (memref), tensors are immutable. Operations produce
    new tensors rather than mutating in place.

    Examples:
        t = Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])
        val = t[2]  # Extract element
    """

    _noun = "Tensor"
    _kind = ast_pb2.ShapedTypeSpec.TENSOR
    _mlir_prefix = "tensor"
    _hash_tag = "tensor"
    _dyn_in_repr = True
    _article = "a"
    _var_name = "t"
    _store_verb = "insert"

    def __call__(self, elements: list):
        """Enable Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0]) construction syntax."""
        from .ast import TensorFromElements
        return TensorFromElements(elements, self)


# ============================================================================
# SHAPED SUBSCRIPT SYNTAX (Array[dtype, N] / Tensor[dtype, N] / [dtype, M, N])
# ============================================================================

class _ShapedMeta(type):
    """Shared metaclass for Array[...]/Tensor[...] subscript syntax.

    Subclasses set _type_cls (ArrayType/TensorType), _noun ("Array"/"Tensor"),
    and _example (the example shown in the parameter-count error message).
    """

    _type_cls = None
    _noun = ""
    _example = ""

    def __getitem__(cls, params):
        """Handle Array[dtype, size] / Array[dtype, M, N] (and Tensor equivalents).

        Args:
            params: Tuple where first element is dtype, rest are dimensions

        Returns:
            ArrayType or TensorType instance
        """
        if not isinstance(params, tuple):
            raise TypeError(
                f"{cls._noun} requires parameters: {cls._noun}[dtype, size] or {cls._noun}[dtype, M, N]. "
                f"Example: {cls._example}"
            )

        if len(params) < 2:
            raise TypeError(
                f"{cls._noun} requires at least 2 parameters (dtype + dimensions), got {len(params)}"
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
                f"Only 1D, 2D, and 3D {cls._noun.lower()}s supported, got {len(dims)}D"
            )

        # Validate dimensions are positive integers or DYN
        for i, dim in enumerate(dims):
            if not isinstance(dim, int) or (dim <= 0 and dim != DYN):
                raise TypeError(f"Dimension {i} must be positive integer or DYN, got {dim!r}")

        # Create the shaped type
        if len(dims) == 1:
            return cls._type_cls(dims[0], dtype)
        else:
            return cls._type_cls(dims, dtype)


class ArrayMeta(_ShapedMeta):
    """Metaclass to enable Array[dtype, size] subscript syntax."""

    _type_cls = ArrayType
    _noun = "Array"
    _example = "Array[i32, 10]"


class Array(metaclass=ArrayMeta):
    """Fixed-size array type for memref dialect.

    Usage as type hint:
        def foo(arr: Array[i32, 10]) -> i32:
            ...

    Usage for construction (inside @ml_function):
        arr = Array[i32, 4]([1, 2, 3, 4])
    """
    pass


class TensorMeta(_ShapedMeta):
    """Metaclass to enable Tensor[dtype, size] subscript syntax."""

    _type_cls = TensorType
    _noun = "Tensor"
    _example = "Tensor[f32, 4]"


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
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                if len(value.shape) != len(type_spec.shape):
                    raise ValueError(
                        f"Parameter '{param_name}': expected {len(type_spec.shape)}D array, "
                        f"got {len(value.shape)}D"
                    )
                for i, (got, expected) in enumerate(zip(value.shape, type_spec.shape)):
                    if expected != DYN and got != expected:
                        raise ValueError(
                            f"Parameter '{param_name}': expected shape {type_spec.shape}, "
                            f"got {tuple(value.shape)}"
                        )
            else:
                raise TypeError(
                    f"Parameter '{param_name}': expected ndarray for {type_spec}, "
                    f"got {type(value).__name__}"
                )
            return

        if isinstance(type_spec, TensorType):
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                if len(value.shape) != len(type_spec.shape):
                    raise ValueError(
                        f"Parameter '{param_name}': expected {len(type_spec.shape)}D tensor, "
                        f"got {len(value.shape)}D"
                    )
                if not type_spec.is_dynamic and tuple(value.shape) != tuple(type_spec.shape):
                    raise ValueError(
                        f"Parameter '{param_name}': expected shape {type_spec.shape}, "
                        f"got {tuple(value.shape)}"
                    )
            else:
                raise TypeError(
                    f"Parameter '{param_name}': expected ndarray for {type_spec}, "
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
            # Concrete inferred type satisfies a DYN declared type
            if (type(inferred) == type(declared)
                    and inferred.element_type == declared.element_type
                    and len(inferred.shape) == len(declared.shape)
                    and all(d == -1 or d == i
                            for d, i in zip(declared.shape, inferred.shape))):
                return True, ""
            return False, (
                f"Array type mismatch:\n"
                f"  Declared: {declared}\n"
                f"  Inferred: {inferred}\n"
                f"  Hint: Ensure array shapes and element types match"
            )


