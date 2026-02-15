"""Array AST nodes: ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp"""

from ..base import Value
from ...types import Type, ScalarType, ArrayType, i32, f32, i1

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, OP_NAMES


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


class ArrayLiteral(Value):
    """
    Array creation: Array[i32, 4]([1, 2, 3, 4])

    Compile-time type checking:
    - Validates size matches number of elements
    - Validates all elements match declared element type (strict!)
    """

    def __init__(self, elements: list, array_type: ArrayType):
        super().__init__()
        self.elements = elements
        self.array_type = array_type

        # COMPILE-TIME TYPE CHECKING
        self._validate_size()
        self._validate_element_types()

    def _validate_size(self):
        """Ensure number of elements matches declared shape and flatten nested lists"""
        if self.array_type.ndim == 1:
            # 1D: elements should be flat list
            if len(self.elements) != self.array_type.shape[0]:
                raise TypeError(
                    f"Array size mismatch: declared Array[{self.array_type.shape[0]}, ...] "
                    f"but got {len(self.elements)} elements"
                )
        elif self.array_type.ndim == 2:
            # 2D: elements should be nested list [[...], [...]]
            self.elements = self._validate_and_flatten_2d(self.elements)
        elif self.array_type.ndim == 3:
            # 3D: elements should be triply nested list [[[...]], [[...]]]
            self.elements = self._validate_and_flatten_3d(self.elements)

    def _validate_and_flatten_2d(self, elements):
        """Validate 2D structure and flatten to row-major order"""
        rows, cols = self.array_type.shape

        if not isinstance(elements, list) or len(elements) != rows:
            raise TypeError(
                f"2D array expects {rows} rows, got {len(elements) if isinstance(elements, list) else 'non-list'}"
            )

        flat = []
        for i, row in enumerate(elements):
            if not isinstance(row, list) or len(row) != cols:
                raise TypeError(
                    f"Row {i}: expected {cols} elements, got {len(row) if isinstance(row, list) else 'non-list'}"
                )
            flat.extend(row)

        return flat

    def _validate_and_flatten_3d(self, elements):
        """Validate 3D structure and flatten to row-major order"""
        d0, d1, d2 = self.array_type.shape

        if not isinstance(elements, list) or len(elements) != d0:
            raise TypeError(
                f"3D array expects {d0} matrices, got {len(elements) if isinstance(elements, list) else 'non-list'}"
            )

        flat = []
        for i, matrix in enumerate(elements):
            if not isinstance(matrix, list) or len(matrix) != d1:
                raise TypeError(
                    f"Matrix {i}: expected {d1} rows, got {len(matrix) if isinstance(matrix, list) else 'non-list'}"
                )
            for j, row in enumerate(matrix):
                if not isinstance(row, list) or len(row) != d2:
                    raise TypeError(
                        f"Matrix {i}, row {j}: expected {d2} elements, got {len(row) if isinstance(row, list) else 'non-list'}"
                    )
                flat.extend(row)

        return flat

    def _validate_element_types(self):
        """Ensure all elements match the declared element type (strict!)"""
        expected_type = self.array_type.element_type

        for i, elem in enumerate(self.elements):
            # Convert Python literals to AST nodes if needed
            elem_node = _to_scalar_node(elem)
            self.elements[i] = elem_node  # Update with AST node

            # Infer element type
            elem_type = elem_node.infer_type()

            # Strict type checking - must be scalar and match exactly
            if isinstance(elem_type, ArrayType):
                raise TypeError(
                    f"Array element at index {i} cannot be an array. "
                    f"Nested arrays not supported yet."
                )

            if elem_type != expected_type:
                raise TypeError(
                    f"Array element type mismatch at index {i}: "
                    f"expected {expected_type}, "
                    f"got {elem_type}. "
                    f"Use cast() for explicit type conversion."
                )

    def infer_type(self) -> Type:
        """ArrayLiteral returns its full ArrayType"""
        return self.array_type

    def get_children(self) -> list['Value']:
        """Return element nodes for serialization traversal"""
        return self.elements

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.array.literal.type.CopyFrom(self.array_type.to_proto())
        for elem in self.elements:
            pb_node.array.literal.elements.append(elem.to_proto(context))
        return pb_node


class ArrayAccess(Value):
    """
    Array element read: arr[index]

    Compile-time type checking:
    - Array must be ArrayType
    - Index must be i32
    - Result type is the array's element type
    """

    def __init__(self, array: Value, index):
        super().__init__()
        self.array = array
        self.indices = _normalize_indices(index)

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array access is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot index into non-array type. "
                f"Expected array, got {array_type}"
            )

        # Check that number of indices matches array dimensions
        if len(self.indices) != array_type.ndim:
            raise TypeError(
                f"Array dimension mismatch: {array_type.ndim}D array requires "
                f"{array_type.ndim} indices, got {len(self.indices)}. "
                f"Usage: arr[i] for 1D, arr[i,j] for 2D, arr[i,j,k] for 3D"
            )

        # Check that all indices are i32
        for i, idx in enumerate(self.indices):
            idx_type = idx.infer_type()
            if not (isinstance(idx_type, ScalarType) and idx_type.is_integer()):
                raise TypeError(
                    f"Array index {i} must be i32, got {idx_type}. "
                    f"Use cast() to convert to i32."
                )

        # Store the array type for infer_type()
        self._array_type = array_type

    def infer_type(self) -> Type:
        """Array access returns the element type (ScalarType)"""
        return self._array_type.element_type

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array] + self.indices

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.array.access.array.CopyFrom(self.array.to_proto(context))
        for idx in self.indices:
            pb_node.array.access.indices.append(idx.to_proto(context))
        pb_node.array.access.result_type.CopyFrom(self._array_type.element_type.to_proto())
        return pb_node


class ArrayStore(Value):
    """
    Array element write: arr[index] = value

    Compile-time type checking:
    - Array must be ArrayType
    - Index must be i32
    - Value type must match array element type exactly
    """

    def __init__(self, array: Value, index, value):
        super().__init__()
        self.array = array
        self.indices = _normalize_indices(index)
        self.value = _to_scalar_node(value)

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array store is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot use []= on non-array type: {array_type}"
            )

        # Check that number of indices matches array dimensions
        if len(self.indices) != array_type.ndim:
            raise TypeError(
                f"Array dimension mismatch: {array_type.ndim}D array requires "
                f"{array_type.ndim} indices, got {len(self.indices)}. "
                f"Usage: arr.at[i].set(v) for 1D, arr.at[i,j].set(v) for 2D"
            )

        # Check that all indices are i32
        for i, idx in enumerate(self.indices):
            idx_type = idx.infer_type()
            if not (isinstance(idx_type, ScalarType) and idx_type.is_integer()):
                raise TypeError(
                    f"Array index {i} must be i32, got {idx_type}"
                )

        # Check value type matches array element type (STRICT!)
        expected_type = array_type.element_type
        actual_type = self.value.infer_type()

        if isinstance(actual_type, ArrayType):
            raise TypeError(
                f"Cannot store array into array element. "
                f"Expected {expected_type}, got {actual_type}"
            )

        if actual_type != expected_type:
            raise TypeError(
                f"Cannot store {actual_type} into "
                f"Array[..., {expected_type}]. "
                f"Use cast() for explicit conversion."
            )

        # Store array type for later
        self._array_type = array_type

    def infer_type(self) -> Type:
        """Store doesn't produce a value, but return array type for consistency"""
        return self._array_type

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array] + self.indices + [self.value]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.array.store.array.CopyFrom(self.array.to_proto(context))
        pb_node.array.store.value.CopyFrom(self.value.to_proto(context))
        for idx in self.indices:
            pb_node.array.store.indices.append(idx.to_proto(context))
        pb_node.array.store.result_type.CopyFrom(self._array_type.to_proto())
        return pb_node


class ArrayBinaryOp(Value):
    """Element-wise binary operation on arrays with broadcasting support

    Supports three modes:
    - Array + Array: Element-wise with matching shapes
    - Array + Scalar: Broadcasting scalar to all elements
    - Scalar + Array: Broadcasting scalar to all elements
    """

    def __init__(self, op: int, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

        # Type inference and validation
        left_type = left.infer_type()
        right_type = right.infer_type()

        # Determine operation mode and result type
        self._infer_broadcast_mode(left_type, right_type)

    def _op_name(self) -> str:
        return OP_NAMES.get(self.op, str(self.op))

    def _infer_broadcast_mode(self, left_type, right_type):
        """Determine broadcasting mode and validate types"""
        left_is_array = isinstance(left_type, ArrayType)
        right_is_array = isinstance(right_type, ArrayType)

        if left_is_array and right_is_array:
            # ARRAY + ARRAY: Shapes AND element types must match exactly
            if left_type.shape != right_type.shape:
                raise TypeError(
                    f"Array shapes must match for element-wise {self._op_name()}.\n"
                    f"  Left:  {left_type} (shape {left_type.shape})\n"
                    f"  Right: {right_type} (shape {right_type.shape})"
                )
            if left_type.element_type != right_type.element_type:
                raise TypeError(
                    f"Array element types must match for element-wise {self._op_name()}.\n"
                    f"  Left:  {left_type.element_type}\n"
                    f"  Right: {right_type.element_type}\n"
                    f"  Hint: Use cast() for explicit type conversion"
                )
            self._result_type = left_type
            self._broadcast_mode = ast_pb2.NONE

        elif left_is_array and not right_is_array:
            # ARRAY + SCALAR: Validate scalar type matches array element type
            if right_type != left_type.element_type:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Array element type: {left_type.element_type}\n"
                    f"  Scalar type: {right_type}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = left_type
            self._broadcast_mode = ast_pb2.SCALAR_RIGHT

        elif not left_is_array and right_is_array:
            # SCALAR + ARRAY: Validate scalar type matches array element type
            if left_type != right_type.element_type:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Scalar type: {left_type}\n"
                    f"  Array element type: {right_type.element_type}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = right_type
            self._broadcast_mode = ast_pb2.SCALAR_LEFT

        else:
            # SCALAR + SCALAR: This should use BinaryOp, not ArrayBinaryOp
            raise TypeError(
                f"ArrayBinaryOp requires at least one array operand.\n"
                f"For scalar operations, use BinaryOp instead."
            )

    def infer_type(self) -> Type:
        """Element-wise operations preserve array type"""
        return self._result_type

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.array.binary_op.op_type = self.op
        pb_node.array.binary_op.result_type.CopyFrom(self._result_type.to_proto())
        pb_node.array.binary_op.broadcast = self._broadcast_mode
        pb_node.array.binary_op.left.CopyFrom(self.left.to_proto(context))
        pb_node.array.binary_op.right.CopyFrom(self.right.to_proto(context))
        return pb_node
