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
from .types import I32, F32, I1

if TYPE_CHECKING:
    from ._mlir_backend import MLIRBuilder


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

    @abstractmethod
    def infer_type(self) -> int:
        """Infer the type of this value (returns I32, F32, or I1 enum)

        Returns:
            int: Protobuf ValueType enum (I32=0, F32=1, I1=2)
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement infer_type()")

    @abstractmethod
    def to_proto(self):
        """Convert this AST node to protobuf Value message

        Raises:
            NotImplementedError: If subclass doesn't implement serialization
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_proto()")

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
        self.value = value
        # Allow explicit type or infer from value
        if value_type is not None:
            self.value_type = value_type
        else:
            self.value_type = I32 if isinstance(value, int) else F32

    def infer_type(self) -> int:
        """Constants know their own type"""
        return self.value_type

    def to_proto(self):
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
    """Represents a binary operation like addition, subtraction, etc."""

    def __init__(self, op: str, left: Value, right: Value):
        self.op = op
        self.left = left
        self.right = right
        # Infer and cache type at construction
        self._inferred_type = self._compute_type()

    def _compute_type(self) -> int:
        """Type promotion rules: F32 > I32"""
        left_type = self.left.infer_type()
        right_type = self.right.infer_type()

        # Float promotion: if either operand is float, result is float
        if left_type == F32 or right_type == F32:
            return F32
        return I32

    def infer_type(self) -> int:
        """Return cached inferred type"""
        return self._inferred_type

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.binary_op.op_type = _binary_op_to_proto(self.op)
        pb_node.binary_op.result_type = self._inferred_type

        # Recursively serialize children
        pb_node.binary_op.left.CopyFrom(self.left.to_proto())
        pb_node.binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class Parameter(Value):
    """Represents a named parameter"""

    def __init__(self, name: str, value_type):
        self.name = name
        self.value_type = value_type

    def infer_type(self) -> int:
        """Parameters have declared types"""
        return self.value_type

    def to_proto(self):
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
        if left_type == F32 or right_type == F32:
            self._operand_type = F32
        else:
            self._operand_type = I32

    def infer_type(self) -> int:
        """Comparisons always return bool"""
        return I1

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.compare_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.compare_op.operand_type = self._operand_type
        pb_node.compare_op.left.CopyFrom(self.left.to_proto())
        pb_node.compare_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class IfOp(Value):
    """Represents a conditional if-then-else operation"""

    def __init__(self, condition: Value, then_value: Value, else_value: Value):
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

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.if_op.condition.CopyFrom(self.condition.to_proto())
        pb_node.if_op.then_value.CopyFrom(self.then_value.to_proto())
        pb_node.if_op.else_value.CopyFrom(self.else_value.to_proto())
        pb_node.if_op.result_type = self._inferred_type
        return pb_node


class CallOp(Value):
    """Represents a function call operation"""

    def __init__(self, func_name: str, args: list[Value], return_type):
        self.func_name = func_name
        self.args = args
        self.return_type = return_type

    def infer_type(self) -> int:
        """Return type is explicitly declared"""
        return self.return_type

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.call_op.func_name = self.func_name
        pb_node.call_op.return_type = self.return_type
        for arg in self.args:
            pb_node.call_op.args.append(arg.to_proto())
        return pb_node


class ForLoopOp(Value):
    """Represents a for loop operation (scf.for) - STRICT TYPE ENFORCEMENT

    All inputs (start, end, step, init_value) must be the same type.
    Supports I32 or F32 (not I1).

    Represents: for(i = start; i < end; i += step) { accumulator = accumulator op i }
    """

    def __init__(self, start: Value, end: Value, step: Value,
                 init_value: Value, operation: str):
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

        # STRICT: All must be the same type
        if not (start_type == end_type == step_type == init_type):
            raise TypeError(
                f"ForLoopOp requires all inputs to have the same type. "
                f"Got: start={start_type}, end={end_type}, "
                f"step={step_type}, init_value={init_type}"
            )

        # Only allow I32 or F32
        if start_type not in (I32, F32):
            raise TypeError(
                f"ForLoopOp only supports I32 or F32, got {start_type}"
            )

        # Result type is trivial - same as all inputs
        self._inferred_type = start_type

    def infer_type(self) -> int:
        """Return the loop result type (same as all inputs)"""
        return self._inferred_type

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
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

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.while_loop_op.init_value.CopyFrom(self.init_value.to_proto())
        pb_node.while_loop_op.target.CopyFrom(self.target.to_proto())
        pb_node.while_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.while_loop_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.while_loop_op.result_type = self._inferred_type

        return pb_node
