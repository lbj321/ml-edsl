"""Control flow AST nodes: IfOp, ForLoopOp, ForIndex, ForIterArg"""

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


class ForIndex(Value):
    """Placeholder for the induction variable of a for loop.

    Leaf node — resolved at IR build time by injecting its node_id
    into the valueCache, mapping to the scf.for block argument %iv.
    """

    def __init__(self):
        super().__init__()
        self.node_id = self.id

    def infer_type(self) -> Type:
        """Induction variable is always i32."""
        return i32

    def get_children(self) -> list['Value']:
        return []

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.control_flow.for_index.node_id = self.node_id
        return pb_node


class ForIterArg(Value):
    """Placeholder for a loop-carried accumulator (iter_arg).

    Leaf node — resolved at IR build time by injecting its node_id
    into the valueCache, mapping to the scf.for block argument for the iter_arg.
    """

    def __init__(self, value_type: Type, arg_index: int = 0):
        super().__init__()
        self.node_id = self.id
        self.value_type = value_type
        self.arg_index = arg_index

    def infer_type(self) -> Type:
        """Returns the type of the accumulator."""
        return self.value_type

    def get_children(self) -> list['Value']:
        return []

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.control_flow.for_iter_arg.node_id = self.node_id
        pb_node.control_flow.for_iter_arg.arg_index = self.arg_index
        return pb_node


class ForLoopOp(Value):
    """Represents a for loop with lambda body (scf.for with iter_args).

    Loop bounds (start, end, step) must all be integers (i32).
    Accumulator (init_value) can be any type (scalar or tensor).
    Body is an AST subtree built by the lambda, referencing ForIndex/ForIterArg.

    Examples:
        t = Tensor.empty(i32, 4)
        t = For(0, 4, init=t, body=lambda i, acc: acc.at[i].set(i * 2))
    """

    def __init__(self, start: Value, end: Value, step: Value,
                 init_value: Value, body: Value,
                 index_placeholder: 'ForIndex', iter_arg_placeholder: 'ForIterArg'):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.init_value = init_value
        self.body = body
        self.index_node_id = index_placeholder.node_id
        self.iter_arg_node_id = iter_arg_placeholder.node_id

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

        # Body result type must match init type
        body_type = body.infer_type()
        if body_type != init_type:
            raise TypeError(
                f"For loop body result type {body_type} does not match "
                f"init type {init_type}. The body must return the same type "
                f"as the accumulator."
            )

        # Result type matches the accumulator type
        self._inferred_type = init_type

    def infer_type(self) -> Type:
        """Return the loop result type (same as accumulator)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.start, self.end, self.step, self.init_value, self.body]

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.control_flow.for_loop.start.CopyFrom(self.start.to_proto(context))
        pb_node.control_flow.for_loop.end.CopyFrom(self.end.to_proto(context))
        pb_node.control_flow.for_loop.step.CopyFrom(self.step.to_proto(context))
        pb_node.control_flow.for_loop.init_value.CopyFrom(self.init_value.to_proto(context))
        pb_node.control_flow.for_loop.result_type.CopyFrom(self._inferred_type.to_proto())
        pb_node.control_flow.for_loop.body.CopyFrom(self.body.to_proto(context))
        pb_node.control_flow.for_loop.index_node_id = self.index_node_id
        pb_node.control_flow.for_loop.iter_arg_node_id = self.iter_arg_node_id
        return pb_node
