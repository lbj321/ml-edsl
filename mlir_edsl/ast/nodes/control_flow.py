"""Control flow AST nodes: IfOp, ForLoopOp, WhileLoopOp"""

from ..base import Value
from ...types import I32, F32, I1, is_numeric_type, is_integer_type

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, _binary_op_to_proto, _predicate_to_proto


class IfOp(Value):
    """Represents a conditional if-then-else operation"""

    def __init__(self, condition: Value, then_value: Value, else_value: Value):
        super().__init__()
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value

        # Type checking at construction
        if condition.infer_type() != I1:
            raise TypeError(f"If condition must be bool (I1), got {condition.infer_type()}")

        then_type = then_value.infer_type()
        else_type = else_value.infer_type()

        if then_type != else_type:
            raise TypeError(
                f"If branches must have same type: then={then_type}, else={else_type}"
            )

        self._inferred_type = then_type

    def infer_type(self) -> int:
        """Return the type of both branches (guaranteed same)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.condition, self.then_value, self.else_value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        pb_node.if_op.condition.CopyFrom(self.condition._to_proto_impl(context))
        pb_node.if_op.then_value.CopyFrom(self.then_value._to_proto_impl(context))
        pb_node.if_op.else_value.CopyFrom(self.else_value._to_proto_impl(context))

        pb_node.if_op.result_type = self._inferred_type
        return pb_node


class ForLoopOp(Value):
    """Represents a for loop operation (scf.for) - STRICT TYPE ENFORCEMENT

    Loop bounds (start, end, step) must all be integers (I32).
    Accumulator (init_value) must all be integers (I32).

    Represents: for(i = start; i < end; i += step) { accumulator = accumulator op i }

    Examples:
        - For(start=0, end=10, step=1, init=0, op="add")      # int loop, int accumulator
    """

    def __init__(self, start: Value, end: Value, step: Value,
                 init_value: Value, operation: str):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.init_value = init_value
        self.operation = operation

        # Get all types
        start_type = start.infer_type()
        end_type = end.infer_type()
        step_type = step.infer_type()
        init_type = init_value.infer_type()

        # STRICT: Loop bounds (start, end, step) must all be the same type
        if not (start_type == end_type == step_type):
            raise TypeError(
                f"ForLoopOp requires loop bounds to have the same type. "
                f"Got: start={start_type}, end={end_type}, step={step_type}"
            )

        # STRICT: Loop bounds must be integers (scf.for requires this)
        if not is_integer_type(start_type):
            raise TypeError(
                f"ForLoopOp requires integer loop bounds (I32). "
                f"Got: {start_type}. Use integer indices with float accumulator if needed."
            )

        # STRICT: Accumulator must be numeric
        if not is_integer_type(init_type):
            raise TypeError(
                f"ForLoopOp accumulator must be numeric (I32). Got: {init_type}"
            )

        # Result type matches the accumulator type
        self._inferred_type = init_type

    def infer_type(self) -> int:
        """Return the loop result type (same as accumulator)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.start, self.end, self.step, self.init_value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        pb_node.for_loop_op.start.CopyFrom(self.start._to_proto_impl(context))
        pb_node.for_loop_op.end.CopyFrom(self.end._to_proto_impl(context))
        pb_node.for_loop_op.step.CopyFrom(self.step._to_proto_impl(context))
        pb_node.for_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))

        pb_node.for_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.for_loop_op.result_type = self._inferred_type

        return pb_node


class WhileLoopOp(Value):
    """Represents a while loop operation (scf.while) - STRICT TYPE ENFORCEMENT

    init_value and target must be the same type.
    Supports I32 or F32 (not I1).

    Represents: while(current predicate target) { current = current op constant }
    """

    def __init__(self, init_value: Value, target: Value,
                 operation: str, predicate: str):
        super().__init__()
        self.init_value = init_value
        self.target = target
        self.operation = operation
        self.predicate = predicate

        # Get types
        init_type = init_value.infer_type()
        target_type = target.infer_type()

        # STRICT: Both must be the same type
        if init_type != target_type:
            raise TypeError(
                f"WhileLoopOp requires init_value and target to have the same type. "
                f"Got: init_value={init_type}, target={target_type}"
            )

        # Only allow I32 or F32
        if init_type not in (I32, F32):
            raise TypeError(
                f"WhileLoopOp only supports I32 or F32, got {init_type}"
            )

        # Result type is trivial - same as inputs
        self._inferred_type = init_type

    def infer_type(self) -> int:
        """Return the loop result type (same as inputs)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.init_value, self.target]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        pb_node.while_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))
        pb_node.while_loop_op.target.CopyFrom(self.target._to_proto_impl(context))

        pb_node.while_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.while_loop_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.while_loop_op.result_type = self._inferred_type

        return pb_node
