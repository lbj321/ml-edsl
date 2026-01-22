"""Scalar AST nodes: Constant, BinaryOp, CompareOp, CastOp"""

from typing import Union
from ..base import Value
from ...types import I32, F32, I1, is_integer_type, type_to_proto

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext, _binary_op_to_proto, _predicate_to_proto


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
        pb_node.constant.type.CopyFrom(type_to_proto(self.value_type))
        if self.value_type == I32:
            pb_node.constant.int_value = self.value
        elif self.value_type == F32:
            pb_node.constant.float_value = self.value
        elif self.value_type == I1:
            pb_node.constant.bool_value = bool(self.value)
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
            from ...types import type_to_string
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
        pb_node.binary_op.result_type.CopyFrom(type_to_proto(self.infer_type()))

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
        pb_node.compare_op.operand_type.CopyFrom(type_to_proto(self._operand_type))

        # Context-aware child serialization
        pb_node.compare_op.left.CopyFrom(self.left._to_proto_impl(context))
        pb_node.compare_op.right.CopyFrom(self.right._to_proto_impl(context))

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
        pb_node.cast_op.value.CopyFrom(self.value._to_proto_impl(context))

        # New schema requires both source and target types
        pb_node.cast_op.source_type.CopyFrom(type_to_proto(self.value.infer_type()))
        pb_node.cast_op.target_type.CopyFrom(type_to_proto(self.target_type))
        return pb_node
