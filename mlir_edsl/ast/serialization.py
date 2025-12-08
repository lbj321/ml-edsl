"""Serialization context and protobuf mapping utilities for AST nodes"""

from typing import TYPE_CHECKING

# Import generated protobuf code
try:
    from .. import ast_pb2
except ImportError:
    # If protobuf hasn't been generated yet, define a placeholder
    ast_pb2 = None

if TYPE_CHECKING:
    from .base import Value


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
