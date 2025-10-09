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


class Value(ABC):
    """Base class for all values in the EDSL"""

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

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Map string op to protobuf enum
        op_map = {
            "add": ast_pb2.ADD,
            "sub": ast_pb2.SUB,
            "mul": ast_pb2.MUL,
            "div": ast_pb2.DIV,
        }
        pb_node.binary_op.op_type = op_map[self.op]

        # Recursively serialize children
        pb_node.binary_op.left.CopyFrom(self.left.to_proto())
        pb_node.binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class Parameter(Value):
    """Represents a named parameter"""

    def __init__(self, name: str, value_type):
        self.name = name
        self.value_type = value_type

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
        self.value_type = I1  # Comparisons always return bool

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Map string predicate to protobuf enum
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

        if self.predicate not in pred_map:
            raise ValueError(f"Unknown predicate: {self.predicate}")

        pb_node.compare_op.predicate = pred_map[self.predicate]
        pb_node.compare_op.left.CopyFrom(self.left.to_proto())
        pb_node.compare_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class IfOp(Value):
    """Represents a conditional if-then-else operation"""

    def __init__(self, condition: CompareOp, then_value: Value, else_value: Value):
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value
        self.value_type = then_value.value_type

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.if_op.condition.CopyFrom(self.condition.to_proto())
        pb_node.if_op.then_value.CopyFrom(self.then_value.to_proto())
        pb_node.if_op.else_value.CopyFrom(self.else_value.to_proto())
        pb_node.if_op.result_type = self.value_type
        return pb_node


class CallOp(Value):
    """Represents a function call operation"""

    def __init__(self, func_name: str, args: list[Value], return_type):
        self.func_name = func_name
        self.args = args
        self.value_type = return_type

    def to_proto(self):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.call_op.func_name = self.func_name
        pb_node.call_op.return_type = self.value_type
        for arg in self.args:
            pb_node.call_op.args.append(arg.to_proto())
        return pb_node
