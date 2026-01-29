"""Control flow AST nodes: IfOp, ForLoopOp"""

from typing import TYPE_CHECKING
from ..base import Value
from ...types import Type, i32, f32, i1

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext

if TYPE_CHECKING:
    from ...types import Type


class IfOp(Value):
    """Represents a conditional if-then-else operation"""

    def __init__(self, condition: Value, then_value: Value, else_value: Value):
        super().__init__()
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value

        # Type checking at construction
        cond_type = condition.infer_type()
        if not cond_type.is_boolean():
            raise TypeError(f"If condition must be bool (i1), got {cond_type}")

        then_type = then_value.infer_type()
        else_type = else_value.infer_type()

        if then_type != else_type:
            raise TypeError(
                f"If branches must have same type: then={then_type}, else={else_type}"
            )

        self._inferred_type = then_type

    def infer_type(self) -> Type:
        """Return the type of both branches (guaranteed same)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.condition, self.then_value, self.else_value]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.control_flow.if_op.condition.CopyFrom(self.condition.to_proto(context))
        pb_node.control_flow.if_op.then_value.CopyFrom(self.then_value.to_proto(context))
        pb_node.control_flow.if_op.else_value.CopyFrom(self.else_value.to_proto(context))
        pb_node.control_flow.if_op.result_type.CopyFrom(self._inferred_type.to_proto())
        return pb_node


class ForLoopOp(Value):
    """Represents a for loop operation (scf.for) - STRICT TYPE ENFORCEMENT

    Loop bounds (start, end, step) must all be integers (i32).
    Accumulator (init_value) must be integer (i32).

    Represents: for(i = start; i < end; i += step) { accumulator = accumulator op i }

    Examples:
        - For(start=0, end=10, step=1, init=0, op=ast_pb2.ADD)
    """

    def __init__(self, start: Value, end: Value, step: Value,
                 init_value: Value, operation: int):
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
        if not start_type.is_integer():
            raise TypeError(
                f"ForLoopOp requires integer loop bounds (i32). "
                f"Got: {start_type}. Use integer indices with float accumulator if needed."
            )

        # STRICT: Accumulator must be integer
        if not init_type.is_integer():
            raise TypeError(
                f"ForLoopOp accumulator must be integer (i32). Got: {init_type}"
            )

        # Result type matches the accumulator type
        self._inferred_type = init_type

    def infer_type(self) -> Type:
        """Return the loop result type (same as accumulator)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.start, self.end, self.step, self.init_value]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.control_flow.for_loop.start.CopyFrom(self.start.to_proto(context))
        pb_node.control_flow.for_loop.end.CopyFrom(self.end.to_proto(context))
        pb_node.control_flow.for_loop.step.CopyFrom(self.step.to_proto(context))
        pb_node.control_flow.for_loop.init_value.CopyFrom(self.init_value.to_proto(context))
        pb_node.control_flow.for_loop.operation = self.operation
        pb_node.control_flow.for_loop.result_type.CopyFrom(self._inferred_type.to_proto())
        return pb_node
