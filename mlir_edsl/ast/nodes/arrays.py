"""Array AST nodes: ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp"""

from ..base import Value
from ...types import Type, ArrayType

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, OP_NAMES
from .shaped import (
    _normalize_indices,
    _to_scalar_node,
    _validate_and_flatten,
    _validate_index_count,
    _validate_indices_are_int,
    _require_container_type,
    _validate_element_type,
    _validate_store_value_type,
)


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
        """Ensure number of elements matches declared shape and flatten nested lists."""
        self.elements = _validate_and_flatten(self.elements, self.array_type.shape, self.array_type._noun)

    def _validate_element_types(self):
        """Ensure all elements match the declared element type (strict!)"""
        expected_type = self.array_type.element_type

        for i, elem in enumerate(self.elements):
            # Convert Python literals to AST nodes if needed
            elem_node = _to_scalar_node(elem)
            self.elements[i] = elem_node  # Update with AST node

            # Infer element type
            elem_type = elem_node.infer_type()

            _validate_element_type(elem_type, expected_type, self.array_type, i)

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
        _require_container_type(array_type, ArrayType)

        # Check that number of indices matches array dimensions
        _validate_index_count(self.indices, array_type)

        # Check that all indices are i32
        _validate_indices_are_int(self.indices, array_type)

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
        _validate_index_count(self.indices, array_type, is_store=True)

        # Check that all indices are i32
        _validate_indices_are_int(self.indices, array_type)

        # Check value type matches array element type (STRICT!)
        expected_type = array_type.element_type
        actual_type = self.value.infer_type()

        _validate_store_value_type(actual_type, expected_type, array_type)

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
