"""Scalar AST nodes: Constant, BinaryOp, CompareOp, CastOp"""

from typing import Union, TYPE_CHECKING
from ..base import Value
from ...types import Type, ScalarType, i32, f32, i1

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, _binary_op_to_proto, _predicate_to_proto

if TYPE_CHECKING:
    from ...types import Type


class Constant(Value):
    """Represents a constant integer or float value"""

    def __init__(self, value: Union[int, float, bool], value_type: Type = None):
        super().__init__()
        self.value = value
        # Allow explicit type or infer from value
        if value_type is not None:
            self.value_type = value_type
        elif isinstance(value, bool):
            self.value_type = i1
        elif isinstance(value, int):
            self.value_type = i32
        else:
            self.value_type = f32

    def infer_type(self) -> Type:
        """Constants know their own type"""
        return self.value_type

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.constant.type.CopyFrom(self.value_type.to_proto())
        if self.value_type.is_integer():
            pb_node.constant.int_value = int(self.value)
        elif self.value_type.is_float():
            pb_node.constant.float_value = float(self.value)
        elif self.value_type.is_boolean():
            pb_node.constant.bool_value = bool(self.value)
        return pb_node


class BinaryOp(Value):
    """Represents a binary operation - STRICT TYPE MATCHING ENFORCED"""

    def __init__(self, op: str, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def infer_type(self) -> Type:
        """STRICT: Both operands must have the same type"""
        left_type = self.left.infer_type()
        right_type = self.right.infer_type()

        if left_type != right_type:
            raise TypeError(
                f"Binary operation '{self.op}' requires matching types.\n"
                f"  Left operand type:  {left_type}\n"
                f"  Right operand type: {right_type}\n"
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
        pb_node.binary_op.result_type.CopyFrom(self.infer_type().to_proto())

        # Context-aware child serialization
        pb_node.binary_op.left.CopyFrom(self.left._to_proto_impl(context))
        pb_node.binary_op.right.CopyFrom(self.right._to_proto_impl(context))

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
        if left_type.is_boolean() or right_type.is_boolean():
            raise TypeError("Cannot compare boolean values")

        # Compute promoted operand type (same rule as BinaryOp)
        # F32 + anything = F32, otherwise I32
        if left_type.is_float() or right_type.is_float():
            self._operand_type = f32
        else:
            self._operand_type = i32

    def infer_type(self) -> Type:
        """Comparisons always return bool"""
        return i1

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.compare_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.compare_op.operand_type.CopyFrom(self._operand_type.to_proto())

        # Context-aware child serialization
        pb_node.compare_op.left.CopyFrom(self.left._to_proto_impl(context))
        pb_node.compare_op.right.CopyFrom(self.right._to_proto_impl(context))

        return pb_node


class CastOp(Value):
    """Explicit type cast operation"""

    def __init__(self, value: Value, target_type: Type):
        """Create a cast operation

        Args:
            value: Value to cast
            target_type: Target type (i32, f32, i1)
        """
        super().__init__()
        self.value = value
        self.target_type = target_type

    def infer_type(self) -> Type:
        """Cast always produces the target type"""
        return self.target_type

    def get_children(self) -> list['Value']:
        return [self.value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        pb_node.cast_op.value.CopyFrom(self.value._to_proto_impl(context))

        # Schema requires both source and target types
        pb_node.cast_op.source_type.CopyFrom(self.value.infer_type().to_proto())
        pb_node.cast_op.target_type.CopyFrom(self.target_type.to_proto())
        return pb_node
