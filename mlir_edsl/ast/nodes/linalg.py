"""Linalg AST nodes: LinalgDot, LinalgMatmul, LinalgMapElement, LinalgMap, LinalgReduce"""

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
        # Only check length compatibility when both dimensions are concrete.
        # DYN (-1) dims are deferred to specialization time when abstract eval
        # runs with concrete shapes. Restricted to 1D arrays only (no batched dot).
        lhs_len, rhs_len = lhs_type.shape[0], rhs_type.shape[0]
        if lhs_len != -1 and rhs_len != -1 and lhs_len != rhs_len:
            raise TypeError(
                f"linalg.dot: operand lengths must match, "
                f"got {lhs_type.shape} and {rhs_type.shape}"
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

        # Only check inner-dimension compatibility when both K values are concrete.
        # DYN (-1) dims are deferred to specialization time when abstract eval
        # runs with concrete shapes. Restricted to 2D arrays only (no batched matmul).
        if K_lhs != -1 and K_rhs != -1 and K_lhs != K_rhs:
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


class LinalgReduceElement(Value):
    """Placeholder for the input element argument in a linalg.reduce body.

    Leaf node — resolved at IR build time via valueCache injection.
    The element_type matches the input array's element type.
    """

    def __init__(self, element_type: Type):
        super().__init__()
        self.node_id = self.id
        self._element_type = element_type

    def infer_type(self) -> Type:
        """Returns the element type of the reduced array."""
        return self._element_type

    def get_children(self) -> list['Value']:
        return []

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.reduce_element.node_id = self.node_id
        return pb_node


class LinalgReduceAccumulator(Value):
    """Placeholder for the accumulator argument in a linalg.reduce body.

    Leaf node — resolved at IR build time via valueCache injection.
    The element_type matches the input array's element type.
    """

    def __init__(self, element_type: Type):
        super().__init__()
        self.node_id = self.id
        self._element_type = element_type

    def infer_type(self) -> Type:
        """Returns the element type (same as the accumulated scalar)."""
        return self._element_type

    def get_children(self) -> list['Value']:
        return []

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.reduce_accum.node_id = self.node_id
        return pb_node


class LinalgReduce(Value):
    """Reduction over a 1D array using linalg.reduce.

    The fn lambda is called at AST construction time with two placeholders:
    - element: LinalgReduceElement (current input element)
    - accumulator: LinalgReduceAccumulator (running accumulator)

    At C++ build time both placeholders are resolved to the linalg.reduce
    body block arguments via valueCache injection.

    Example:
        result = reduce(arr, to_value(0.0), lambda elem, acc: acc + elem)
    """

    def __init__(self, input: Value, init: Value, fn):
        super().__init__()
        self.input = input
        self.init = init

        input_type = input.infer_type()
        if not isinstance(input_type, ArrayType) or input_type.ndim != 1:
            raise TypeError(
                f"linalg.reduce: input must be a 1D array, got {input_type}"
            )
        self._element_type = input_type.element_type

        init_type = init.infer_type()
        if not isinstance(init_type, ScalarType) or init_type != self._element_type:
            raise TypeError(
                f"linalg.reduce: init must be a scalar of element type "
                f"{self._element_type}, got {init_type}"
            )

        self._elem_placeholder = LinalgReduceElement(self._element_type)
        self._accum_placeholder = LinalgReduceAccumulator(self._element_type)
        self.body = fn(self._elem_placeholder, self._accum_placeholder)

        body_type = self.body.infer_type()
        if body_type != self._element_type:
            raise TypeError(
                f"linalg.reduce body must return element type {self._element_type}, "
                f"got {body_type}"
            )

    def infer_type(self) -> Type:
        """Reduction returns a scalar of the element type."""
        return self._element_type

    def get_children(self) -> list['Value']:
        return [self.input, self.init, self.body]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.linalg.reduce.input.CopyFrom(self.input.to_proto(context))
        pb_node.linalg.reduce.init.CopyFrom(self.init.to_proto(context))
        pb_node.linalg.reduce.body.CopyFrom(self.body.to_proto(context))
        pb_node.linalg.reduce.element_node_id = self._elem_placeholder.node_id
        pb_node.linalg.reduce.accum_node_id = self._accum_placeholder.node_id
        pb_node.linalg.reduce.elem_type.CopyFrom(self._element_type.to_proto())
        return pb_node
