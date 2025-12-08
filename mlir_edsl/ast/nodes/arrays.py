"""Array AST nodes: ArrayLiteral, ArrayAccess, ArrayStore, ArrayBinaryOp"""

from typing import Union
from ..base import Value
from ...types import I32, F32, I1, ArrayType

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, _binary_op_to_proto


class ArrayLiteral(Value):
    """
    Array creation: Array[4, i32]([1, 2, 3, 4])

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
        """Ensure number of elements matches declared size"""
        if len(self.elements) != self.array_type.size:
            raise TypeError(
                f"Array size mismatch: declared Array[{self.array_type.size}, ...] "
                f"but got {len(self.elements)} elements"
            )

    def _validate_element_types(self):
        """Ensure all elements match the declared element type (strict!)"""
        expected_enum = self.array_type.element_enum

        for i, elem in enumerate(self.elements):
            # Convert Python literals to AST nodes if needed
            elem_node = self._ensure_ast_node(elem)
            self.elements[i] = elem_node  # Update with AST node

            # Infer element type
            elem_type = elem_node.infer_type()

            # Strict type checking - must be scalar and match exactly
            if isinstance(elem_type, ArrayType):
                raise TypeError(
                    f"Array element at index {i} cannot be an array. "
                    f"Nested arrays not supported yet."
                )

            if elem_type != expected_enum:
                raise TypeError(
                    f"Array element type mismatch at index {i}: "
                    f"expected {self.array_type.element_type.name}, "
                    f"got {self._enum_to_name(elem_type)}. "
                    f"Use cast() for explicit type conversion."
                )

    def _ensure_ast_node(self, elem):
        """Convert Python literal to AST node if needed"""
        if isinstance(elem, Value):
            return elem

        # Import here to avoid circular dependency
        from .scalars import Constant

        # Convert Python literals
        if isinstance(elem, bool):
            # Must check bool before int (bool is subclass of int)
            return Constant(elem, I1)
        elif isinstance(elem, int):
            return Constant(elem, I32)
        elif isinstance(elem, float):
            return Constant(elem, F32)
        else:
            raise TypeError(f"Invalid array element: {elem}")

    def _enum_to_name(self, enum_val):
        """Helper: convert enum to readable name"""
        return {I32: "i32", F32: "f32", I1: "i1"}.get(enum_val, f"unknown({enum_val})")

    def infer_type(self) -> Union[int, ArrayType]:
        """ArrayLiteral returns its full ArrayType"""
        return self.array_type  # Returns ArrayType instance, not int!

    def get_children(self) -> list['Value']:
        """Return element nodes for serialization traversal"""
        return self.elements

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayLiteral to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Set array type specification
        pb_node.array_literal.array_type.size = self.array_type.size
        pb_node.array_literal.array_type.element_type = self.array_type.element_enum

        # Serialize each element (with context-aware serialization)
        for elem in self.elements:
            if context:
                pb_node.array_literal.elements.append(elem._to_proto_impl(context))
            else:
                pb_node.array_literal.elements.append(elem.to_proto())

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

        # Import here to avoid circular dependency
        from .scalars import Constant

        # Convert index to AST node if it's a Python int
        if isinstance(index, int):
            self.index = Constant(index, I32)
        elif isinstance(index, Value):
            self.index = index
        else:
            raise TypeError(f"Array index must be int or Value, got {type(index)}")

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array access is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot index into non-array type. "
                f"Expected array, got {self._type_to_str(array_type)}"
            )

        # Check that index is i32
        index_type = self.index.infer_type()
        if isinstance(index_type, ArrayType) or index_type != I32:
            raise TypeError(
                f"Array index must be i32, got {self._type_to_str(index_type)}. "
                f"Use cast() to convert to i32."
            )

        # Store the array type for infer_type()
        self._array_type = array_type

    def _type_to_str(self, typ):
        """Helper: convert type to readable string"""
        if isinstance(typ, ArrayType):
            return repr(typ)
        return {I32: "i32", F32: "f32", I1: "i1"}.get(typ, f"unknown({typ})")

    def infer_type(self) -> Union[int, ArrayType]:
        """Array access returns the element type (scalar enum)"""
        # If we index into Array[10, i32], we get back i32
        return self._array_type.element_enum  # Returns I32/F32/I1 (int)

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array, self.index]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayAccess to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Serialize array and index (context-aware)
        if context:
            pb_node.array_access.array.CopyFrom(self.array._to_proto_impl(context))
            pb_node.array_access.index.CopyFrom(self.index._to_proto_impl(context))
        else:
            pb_node.array_access.array.CopyFrom(self.array.to_proto())
            pb_node.array_access.index.CopyFrom(self.index.to_proto())

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

        # Import here to avoid circular dependency
        from .scalars import Constant

        # Convert index to AST node
        if isinstance(index, int):
            self.index = Constant(index, I32)
        elif isinstance(index, Value):
            self.index = index
        else:
            raise TypeError(f"Array index must be int or Value")

        # Convert value to AST node if needed
        if isinstance(value, bool):
            # Must check bool before int (bool is subclass of int)
            self.value = Constant(value, I1)
        elif isinstance(value, int):
            self.value = Constant(value, I32)
        elif isinstance(value, float):
            self.value = Constant(value, F32)
        elif isinstance(value, Value):
            self.value = value
        else:
            raise TypeError(f"Array value must be int/float/bool or Value")

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array store is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot use []= on non-array type: {self._type_to_str(array_type)}"
            )

        # Check index is i32
        index_type = self.index.infer_type()
        if isinstance(index_type, ArrayType) or index_type != I32:
            raise TypeError(f"Array index must be i32, got {self._type_to_str(index_type)}")

        # Check value type matches array element type (STRICT!)
        expected_enum = array_type.element_enum
        actual_type = self.value.infer_type()

        if isinstance(actual_type, ArrayType):
            raise TypeError(
                f"Cannot store array into array element. "
                f"Expected {array_type.element_type.name}, got {actual_type}"
            )

        if actual_type != expected_enum:
            raise TypeError(
                f"Cannot store {self._type_to_str(actual_type)} into "
                f"Array[..., {array_type.element_type.name}]. "
                f"Use cast() for explicit conversion."
            )

        # Store array type for later
        self._array_type = array_type

    def _type_to_str(self, typ):
        """Helper: convert type to readable string"""
        if isinstance(typ, ArrayType):
            return repr(typ)
        return {I32: "i32", F32: "f32", I1: "i1"}.get(typ, f"unknown({typ})")

    def infer_type(self) -> Union[int, ArrayType]:
        """Store doesn't produce a value, but return array type for consistency"""
        return self._array_type

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array, self.index, self.value]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayStore to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Serialize array, index, and value (context-aware)
        if context:
            pb_node.array_store.array.CopyFrom(self.array._to_proto_impl(context))
            pb_node.array_store.index.CopyFrom(self.index._to_proto_impl(context))
            pb_node.array_store.value.CopyFrom(self.value._to_proto_impl(context))
        else:
            pb_node.array_store.array.CopyFrom(self.array.to_proto())
            pb_node.array_store.index.CopyFrom(self.index.to_proto())
            pb_node.array_store.value.CopyFrom(self.value.to_proto())

        return pb_node


class ArrayBinaryOp(Value):
    """Element-wise binary operation on arrays with broadcasting support

    Supports three modes:
    - Array + Array: Element-wise with matching shapes
    - Array + Scalar: Broadcasting scalar to all elements
    - Scalar + Array: Broadcasting scalar to all elements

    Examples:
        arr1 + arr2          # Array[4,i32] + Array[4,i32] -> Array[4,i32]
        arr * 2              # Array[4,i32] * Constant(2) -> Array[4,i32]
        3.0 + arr            # Constant(3.0) + Array[4,f32] -> Array[4,f32]
    """

    def __init__(self, op: str, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

        # Type inference and validation
        left_type = left.infer_type()
        right_type = right.infer_type()

        # Determine operation mode and result type
        self._infer_broadcast_mode(left_type, right_type)

    def _infer_broadcast_mode(self, left_type, right_type):
        """Determine broadcasting mode and validate types"""
        left_is_array = isinstance(left_type, ArrayType)
        right_is_array = isinstance(right_type, ArrayType)

        if left_is_array and right_is_array:
            # ARRAY + ARRAY: Shapes must match exactly
            if left_type != right_type:
                raise TypeError(
                    f"Array shapes must match for element-wise {self.op}.\n"
                    f"  Left:  {left_type}\n"
                    f"  Right: {right_type}"
                )
            self._result_type = left_type
            self._broadcast_mode = "NONE"

        elif left_is_array and not right_is_array:
            # ARRAY + SCALAR: Validate scalar type matches array element type
            if right_type != left_type.element_enum:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Array element type: {left_type.element_type.name}\n"
                    f"  Scalar type: {self._enum_to_name(right_type)}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = left_type
            self._broadcast_mode = "SCALAR_RIGHT"

        elif not left_is_array and right_is_array:
            # SCALAR + ARRAY: Validate scalar type matches array element type
            if left_type != right_type.element_enum:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Scalar type: {self._enum_to_name(left_type)}\n"
                    f"  Array element type: {right_type.element_type.name}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = right_type
            self._broadcast_mode = "SCALAR_LEFT"

        else:
            # SCALAR + SCALAR: This should use BinaryOp, not ArrayBinaryOp
            raise TypeError(
                f"ArrayBinaryOp requires at least one array operand.\n"
                f"For scalar operations, use BinaryOp instead."
            )

    def _enum_to_name(self, enum_val):
        """Helper: convert enum to readable name"""
        return {I32: "i32", F32: "f32", I1: "i1"}.get(enum_val, f"unknown({enum_val})")

    def infer_type(self) -> ArrayType:
        """Element-wise operations preserve array type"""
        return self._result_type

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.array_binary_op.op_type = _binary_op_to_proto(self.op)

        # Set result type (array shape)
        pb_node.array_binary_op.result_type.size = self._result_type.size
        pb_node.array_binary_op.result_type.element_type = self._result_type.element_enum

        # Set broadcast mode
        broadcast_map = {
            "NONE": ast_pb2.NONE,
            "SCALAR_LEFT": ast_pb2.SCALAR_LEFT,
            "SCALAR_RIGHT": ast_pb2.SCALAR_RIGHT,
        }
        pb_node.array_binary_op.broadcast = broadcast_map[self._broadcast_mode]

        # Context-aware child serialization
        if context:
            pb_node.array_binary_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.array_binary_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            pb_node.array_binary_op.left.CopyFrom(self.left.to_proto())
            pb_node.array_binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node
