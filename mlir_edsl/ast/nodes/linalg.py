"""Linalg AST nodes: LinalgDot, LinalgMatmul, LinalgMapElement, LinalgMap"""

from ..base import Value
from ...types import Type, ScalarType, ArrayType

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext


class LinalgDot(Value):
    """Dot product of two 1D arrays.

    Maps to MLIR linalg.dot op. Returns a scalar of the operand element type.

    Compile-time type checking:
    - Both operands must be 1D ArrayType
    - Element types of lhs and rhs must match

    Example:
        result = dot(a, b)  # a, b: Array[f32, 4] → f32
    """

    def __init__(self, lhs: Value, rhs: Value):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self._validate()

    def _validate(self):
        """Validate operand types for linalg.dot."""
        lhs_type = self.lhs.infer_type()
        rhs_type = self.rhs.infer_type()

        if not isinstance(lhs_type, ArrayType) or lhs_type.ndim != 1:
            raise TypeError(
                f"linalg.dot: lhs must be a 1D array, got {lhs_type}"
            )
        if not isinstance(rhs_type, ArrayType) or rhs_type.ndim != 1:
            raise TypeError(
                f"linalg.dot: rhs must be a 1D array, got {rhs_type}"
            )
        if lhs_type.element_type != rhs_type.element_type:
            raise TypeError(
                f"linalg.dot: operand element types must match, "
                f"got {lhs_type.element_type} and {rhs_type.element_type}"
            )
        self._element_type = lhs_type.element_type

    def infer_type(self) -> Type:
        """Dot product returns a scalar (element type of the inputs)."""
        return self._element_type

    def get_children(self) -> list['Value']:
        """Return operand nodes."""
        return [self.lhs, self.rhs]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.dot.lhs.CopyFrom(self.lhs.to_proto(context))
        pb_node.linalg.dot.rhs.CopyFrom(self.rhs.to_proto(context))
        return pb_node


class LinalgMatmul(Value):
    """Matrix multiplication of two 2D arrays.

    Maps to MLIR linalg.matmul op. Returns a 2D array of shape [M, N].

    Compile-time type checking:
    - Both operands must be 2D ArrayType
    - Element types must match
    - Inner dimensions must match: lhs[M, K] x rhs[K, N]

    Example:
        C = matmul(A, B)  # A: Array[f32, 4, 4], B: Array[f32, 4, 4] → Array[f32, 4, 4]
    """

    def __init__(self, lhs: Value, rhs: Value):
        super().__init__()
        self.lhs = lhs
        self.rhs = rhs
        self._validate()

    def _validate(self):
        """Validate operand types for linalg.matmul."""
        lhs_type = self.lhs.infer_type()
        rhs_type = self.rhs.infer_type()

        if not isinstance(lhs_type, ArrayType) or lhs_type.ndim != 2:
            raise TypeError(
                f"linalg.matmul: lhs must be a 2D array, got {lhs_type}"
            )
        if not isinstance(rhs_type, ArrayType) or rhs_type.ndim != 2:
            raise TypeError(
                f"linalg.matmul: rhs must be a 2D array, got {rhs_type}"
            )
        if lhs_type.element_type != rhs_type.element_type:
            raise TypeError(
                f"linalg.matmul: operand element types must match, "
                f"got {lhs_type.element_type} and {rhs_type.element_type}"
            )

        M, K_lhs = lhs_type.shape
        K_rhs, N = rhs_type.shape

        if K_lhs != K_rhs:
            raise TypeError(
                f"linalg.matmul: inner dimensions must match, "
                f"got lhs shape {lhs_type.shape} and rhs shape {rhs_type.shape}"
            )

        self._out_type = ArrayType((M, N), lhs_type.element_type)

    def infer_type(self) -> Type:
        """Matmul returns a 2D array of shape [M, N]."""
        return self._out_type

    def get_children(self) -> list['Value']:
        """Return operand nodes."""
        return [self.lhs, self.rhs]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.matmul.lhs.CopyFrom(self.lhs.to_proto(context))
        pb_node.linalg.matmul.rhs.CopyFrom(self.rhs.to_proto(context))
        pb_node.linalg.matmul.out_type.CopyFrom(self._out_type.to_proto())
        return pb_node


class LinalgMapElement(Value):
    """Placeholder for the linalg.map body block argument.

    Leaf node — resolved at IR build time via valueCache injection.
    The element_type matches the input array's element type.
    """

    def __init__(self, element_type: Type):
        super().__init__()
        self.node_id = self.id
        self._element_type = element_type

    def infer_type(self) -> Type:
        """Returns the element type of the mapped array."""
        return self._element_type

    def get_children(self) -> list['Value']:
        return []

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.map_element.node_id = self.node_id
        return pb_node


class LinalgMap(Value):
    """Element-wise map over a 1D array using linalg.map (MapOp).

    The fn lambda is called at AST construction time with a LinalgMapElement
    placeholder. At C++ build time the placeholder is resolved to the
    linalg.map body block argument via valueCache injection.

    Example:
        result = tensor_map(arr, lambda v: v * 2.0)
    """

    def __init__(self, input: Value, fn):
        super().__init__()
        self.input = input
        input_type = input.infer_type()
        if not isinstance(input_type, ArrayType) or input_type.ndim != 1:
            raise TypeError(
                f"tensor_map: input must be a 1D array, got {input_type}"
            )
        self._out_type = input_type
        self._element_placeholder = LinalgMapElement(input_type.element_type)
        self.body = fn(self._element_placeholder)
        body_type = self.body.infer_type()
        if body_type != input_type.element_type:
            raise TypeError(
                f"tensor_map body must return element type {input_type.element_type}, "
                f"got {body_type}"
            )

    def infer_type(self) -> Type:
        """Returns the same array type as the input."""
        return self._out_type

    def get_children(self) -> list['Value']:
        # LinalgMapElement appears inside body's subtree — do not list it here
        return [self.input, self.body]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.map.input.CopyFrom(self.input.to_proto(context))
        pb_node.linalg.map.body.CopyFrom(self.body.to_proto(context))
        pb_node.linalg.map.element_node_id = self._element_placeholder.node_id
        pb_node.linalg.map.out_type.CopyFrom(self._out_type.to_proto())
        return pb_node
