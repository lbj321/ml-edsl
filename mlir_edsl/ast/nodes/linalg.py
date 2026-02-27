"""Linalg AST nodes: LinalgDot, LinalgMatmul"""

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
