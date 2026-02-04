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


# ==================== ENUM NAME LOOKUPS (for error messages) ====================

OP_NAMES = {
    ast_pb2.ADD: "add",
    ast_pb2.SUB: "sub",
    ast_pb2.MUL: "mul",
    ast_pb2.DIV: "div",
} if ast_pb2 else {}

PREDICATE_NAMES = {
    ast_pb2.GT: "gt", ast_pb2.LT: "lt",
    ast_pb2.EQ: "eq", ast_pb2.NE: "ne",
    ast_pb2.GE: "ge", ast_pb2.LE: "le",
    ast_pb2.SLT: "slt", ast_pb2.SLE: "sle",
    ast_pb2.SGT: "sgt", ast_pb2.SGE: "sge",
    ast_pb2.ULT: "ult", ast_pb2.ULE: "ule",
    ast_pb2.UGT: "ugt", ast_pb2.UGE: "uge",
    ast_pb2.OLT: "olt", ast_pb2.OLE: "ole",
    ast_pb2.OGT: "ogt", ast_pb2.OGE: "oge",
    ast_pb2.OEQ: "oeq", ast_pb2.ONE: "one",
    ast_pb2.UEQ: "ueq", ast_pb2.UNE: "une",
} if ast_pb2 else {}
