"""Abstract Syntax Tree nodes for the EDSL with protobuf serialization"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

# Import generated protobuf code
try:
    from . import ast_pb2
except ImportError:
    # If protobuf hasn't been generated yet, define a placeholder
    ast_pb2 = None

# Import type system
from .types import I32, F32, I1, is_numeric_type, is_integer_type

if TYPE_CHECKING:
    from ._mlir_backend import MLIRBuilder


# ==================== SERIALIZATION CONTEXT ====================
class SerializationContext:
    """Tracks Value reuse during AST serialization for SSA value reuse"""

    def __init__(self):
        self.use_counts = {}     # value.id -> int (how many times referenced)
        self.serialized = set()  # set of value.id that have been serialized

    def count_uses(self, value: 'Value'):
        """Recursively count how many times each Value appears in the tree"""
        if value.id in self.use_counts:
            # Already seen - increment count but don't traverse again
            self.use_counts[value.id] += 1
            return

        # First encounter
        self.use_counts[value.id] = 1

        # Generic traversal - works for ALL node types!
        for child in value.get_children():
            self.count_uses(child)

    def is_reused(self, value: 'Value') -> bool:
        """Check if a value is used more than once"""
        return self.use_counts.get(value.id, 0) > 1

    def mark_serialized(self, value: 'Value'):
        """Mark a value as already serialized"""
        self.serialized.add(value.id)

    def is_serialized(self, value: 'Value') -> bool:
        """Check if a value has already been serialized"""
        return value.id in self.serialized


# ==================== PROTOBUF ENUM MAPPINGS ====================
# These mappings convert string operations/predicates to protobuf enums
# Shared across multiple AST node classes during serialization

def _binary_op_to_proto(operation: str):
    """Convert operation string to protobuf BinaryOpType enum"""
    op_map = {
        "add": ast_pb2.ADD,
        "sub": ast_pb2.SUB,
        "mul": ast_pb2.MUL,
        "div": ast_pb2.DIV,
    }
    return op_map[operation]


def _predicate_to_proto(predicate: str):
    """Convert predicate string to protobuf ComparisonPredicate enum"""
    pred_map = {
        "gt": ast_pb2.GT, "lt": ast_pb2.LT,
        "eq": ast_pb2.EQ, "ne": ast_pb2.NE,
        "ge": ast_pb2.GE, "le": ast_pb2.LE,
        "slt": ast_pb2.SLT, "sle": ast_pb2.SLE,
        "sgt": ast_pb2.SGT, "sge": ast_pb2.SGE,
        "ult": ast_pb2.ULT, "ule": ast_pb2.ULE,
        "ugt": ast_pb2.UGT, "uge": ast_pb2.UGE,
        "olt": ast_pb2.OLT, "ole": ast_pb2.OLE,
        "ogt": ast_pb2.OGT, "oge": ast_pb2.OGE,
        "oeq": ast_pb2.OEQ, "one": ast_pb2.ONE,
        "ueq": ast_pb2.UEQ, "une": ast_pb2.UNE,
    }
    if predicate not in pred_map:
        raise ValueError(f"Unknown predicate: {predicate}")
    return pred_map[predicate]


class Value(ABC):
    """Base class for all values in the EDSL"""
    _next_id = 0

    def __init__(self):
        """Every Value gets a unique ID for reference tracking"""
        self.id = Value._next_id
        Value._next_id += 1

    @abstractmethod
    def infer_type(self) -> int:
        """Infer the type of this value (returns I32, F32, or I1 enum)

        Returns:
            int: Protobuf ValueType enum (I32=0, F32=1, I1=2)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement infer_type()")

    @abstractmethod
    def to_proto(self, context: 'SerializationContext' = None):
        """Convert this AST node to protobuf Value message

        Args:
            context: Optional serialization context for SSA value reuse

        Raises:
            NotImplementedError: If subclass doesn't implement serialization
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_proto()")

    def get_children(self) -> list['Value']:
        """Return list of child Value nodes. Override in subclasses with children."""
        return []  # Default: no children (for leaf nodes like Constant, Parameter)

    def to_proto_with_reuse(self):
        """Serialize with SSA value reuse detection (two-pass approach)"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        # Pass 1: Count how many times each Value is referenced
        context = SerializationContext()
        context.count_uses(self)

        # Pass 2: Serialize with let bindings for reused values
        return self._to_proto_impl(context)

    def _to_proto_impl(self, context: 'SerializationContext'):
        """Serialize this value, emitting let bindings or references as needed"""
        # If this value was already serialized, emit a reference
        if context.is_serialized(self):
            pb_node = ast_pb2.ASTNode()
            pb_node.value_ref.node_id = self.id
            return pb_node

        # If this value will be reused, wrap it in a let binding
        if context.is_reused(self):
            context.mark_serialized(self)
            pb_node = ast_pb2.ASTNode()
            pb_node.let_binding.node_id = self.id
            pb_node.let_binding.value.CopyFrom(self.to_proto(context))
            return pb_node

        # Otherwise, inline it normally
        return self.to_proto(context)

    # Arithmetic operators
    def __add__(self, other):
        """Overload + operator: x + y"""
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        """Reverse add: 5 + x"""
        from .ops import add
        return add(other, self)

    def __sub__(self, other):
        """Overload - operator: x - y"""
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        """Reverse sub: 5 - x"""
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other):
        """Overload * operator: x * y"""
        from .ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        """Reverse mul: 5 * x"""
        from .ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        """Overload / operator: x / y"""
        from .ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        """Reverse div: 5 / x"""
        from .ops import div
        return div(other, self)

    # Comparison operators
    def __lt__(self, other):
        """Overload < operator: x < y"""
        from .ops import lt
        return lt(self, other)

    def __le__(self, other):
        """Overload <= operator: x <= y"""
        from .ops import le
        return le(self, other)

    def __gt__(self, other):
        """Overload > operator: x > y"""
        from .ops import gt
        return gt(self, other)

    def __ge__(self, other):
        """Overload >= operator: x >= y"""
        from .ops import ge
        return ge(self, other)

    def __eq__(self, other):
        """Overload == operator: x == y"""
        from .ops import eq
        return eq(self, other)

    def __ne__(self, other):
        """Overload != operator: x != y"""
        from .ops import ne
        return ne(self, other)


class Constant(Value):
    """Represents a constant integer or float value"""

    def __init__(self, value: Union[int, float], value_type=None):
        super().__init__()
        self.value = value
        # Allow explicit type or infer from value
        if value_type is not None:
            self.value_type = value_type
        else:
            self.value_type = I32 if isinstance(value, int) else F32

    def infer_type(self) -> int:
        """Constants know their own type"""
        return self.value_type

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.constant.value_type = self.value_type
        if self.value_type == I32:
            pb_node.constant.int_value = self.value
        else:
            pb_node.constant.float_value = self.value
        return pb_node


class BinaryOp(Value):
    """Represents a binary operation - STRICT TYPE MATCHING ENFORCED"""

    def __init__(self, op: str, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def infer_type(self) -> int:
        """STRICT: Both operands must have the same type"""
        left_type = self.left.infer_type()
        right_type = self.right.infer_type()

        if left_type != right_type:
            from .types import type_to_string
            raise TypeError(
                f"Binary operation '{self.op}' requires matching types.\n"
                f"  Left operand type:  {type_to_string(left_type)}\n"
                f"  Right operand type: {type_to_string(right_type)}\n"
                f"  Hint: Use cast() to convert types explicitly"
            )

        return left_type

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.binary_op.op_type = _binary_op_to_proto(self.op)
        pb_node.binary_op.result_type = self.infer_type()

        # Context-aware child serialization
        if context:
            pb_node.binary_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.binary_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            # Backward compatibility: no context = old behavior
            pb_node.binary_op.left.CopyFrom(self.left.to_proto())
            pb_node.binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class Parameter(Value):
    """Represents a named parameter"""

    def __init__(self, name: str, value_type):
        super().__init__()
        self.name = name
        self.value_type = value_type

    def infer_type(self) -> int:
        """Parameters have declared types"""
        return self.value_type

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.parameter.name = self.name
        pb_node.parameter.value_type = self.value_type
        # Value fields are unused by C++ backend (uses parameterMap instead)
        return pb_node


class CompareOp(Value):
    """Represents a comparison operation"""

    def __init__(self, predicate: str, left: Value, right: Value):
        super().__init__()
        self.predicate = predicate
        self.left = left
        self.right = right

        # Validate and infer operand types
        left_type = left.infer_type()
        right_type = right.infer_type()
        if left_type == I1 or right_type == I1:
            raise TypeError("Cannot compare boolean values")

        # Compute promoted operand type (same rule as BinaryOp)
        # F32 + anything = F32, otherwise I32
        # TODO: Fix
        if left_type == F32 or right_type == F32:
            self._operand_type = F32
        else:
            self._operand_type = I32

    def infer_type(self) -> int:
        """Comparisons always return bool"""
        return I1

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.compare_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.compare_op.operand_type = self._operand_type

        # Context-aware child serialization
        if context:
            pb_node.compare_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.compare_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            pb_node.compare_op.left.CopyFrom(self.left.to_proto())
            pb_node.compare_op.right.CopyFrom(self.right.to_proto())

        return pb_node


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
        if context:
            pb_node.if_op.condition.CopyFrom(self.condition._to_proto_impl(context))
            pb_node.if_op.then_value.CopyFrom(self.then_value._to_proto_impl(context))
            pb_node.if_op.else_value.CopyFrom(self.else_value._to_proto_impl(context))
        else:
            pb_node.if_op.condition.CopyFrom(self.condition.to_proto())
            pb_node.if_op.then_value.CopyFrom(self.then_value.to_proto())
            pb_node.if_op.else_value.CopyFrom(self.else_value.to_proto())

        pb_node.if_op.result_type = self._inferred_type
        return pb_node


class CallOp(Value):
    """Represents a function call operation"""

    def __init__(self, func_name: str, args: list[Value], return_type):
        super().__init__()
        self.func_name = func_name
        self.args = args
        self.return_type = return_type

    def infer_type(self) -> int:
        """Return type is explicitly declared"""
        return self.return_type

    def get_children(self) -> list['Value']:
        return self.args

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.call_op.func_name = self.func_name
        pb_node.call_op.return_type = self.return_type

        # Context-aware child serialization
        if context:
            for arg in self.args:
                pb_node.call_op.args.append(arg._to_proto_impl(context))
        else:
            for arg in self.args:
                pb_node.call_op.args.append(arg.to_proto())

        return pb_node


class ForLoopOp(Value):
    """Represents a for loop operation (scf.for) - STRICT TYPE ENFORCEMENT

    Loop bounds (start, end, step) must all be integers (I32).
    Accumulator (init_value) can be any numeric type (I32 or F32).

    Represents: for(i = start; i < end; i += step) { accumulator = accumulator op i }

    Examples:
        - For(start=0, end=10, step=1, init=0, op="add")      # int loop, int accumulator
        - For(start=0, end=10, step=1, init=0.5, op="add")    # int loop, float accumulator
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
        if context:
            pb_node.for_loop_op.start.CopyFrom(self.start._to_proto_impl(context))
            pb_node.for_loop_op.end.CopyFrom(self.end._to_proto_impl(context))
            pb_node.for_loop_op.step.CopyFrom(self.step._to_proto_impl(context))
            pb_node.for_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))
        else:
            pb_node.for_loop_op.start.CopyFrom(self.start.to_proto())
            pb_node.for_loop_op.end.CopyFrom(self.end.to_proto())
            pb_node.for_loop_op.step.CopyFrom(self.step.to_proto())
            pb_node.for_loop_op.init_value.CopyFrom(self.init_value.to_proto())

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
        if context:
            pb_node.while_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))
            pb_node.while_loop_op.target.CopyFrom(self.target._to_proto_impl(context))
        else:
            pb_node.while_loop_op.init_value.CopyFrom(self.init_value.to_proto())
            pb_node.while_loop_op.target.CopyFrom(self.target.to_proto())

        pb_node.while_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.while_loop_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.while_loop_op.result_type = self._inferred_type

        return pb_node


class CastOp(Value):
    """Explicit type cast operation"""

    def __init__(self, value: Value, target_type: int):
        """Create a cast operation

        Args:
            value: Value to cast
            target_type: Target MLIR type enum (I32, F32, I1)
        """
        super().__init__()
        self.value = value
        self.target_type = target_type

    def infer_type(self) -> int:
        """Cast always produces the target type"""
        return self.target_type

    def get_children(self) -> list['Value']:
        return [self.value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        if context:
            pb_node.cast_op.value.CopyFrom(self.value._to_proto_impl(context))
        else:
            pb_node.cast_op.value.CopyFrom(self.value.to_proto())

        pb_node.cast_op.target_type = self.target_type
        return pb_node
