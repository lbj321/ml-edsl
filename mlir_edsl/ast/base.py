"""Base Value class for all AST nodes"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
from .operators import OperatorMixin
from ..types import ArrayType

# Import generated protobuf code
try:
    from .. import ast_pb2
except ImportError:
    # If protobuf hasn't been generated yet, define a placeholder
    ast_pb2 = None

if TYPE_CHECKING:
    from .serialization import SerializationContext


class Value(ABC, OperatorMixin):
    """Base class for all values in the EDSL"""
    _next_id = 0

    def __init__(self):
        """Every Value gets a unique ID for reference tracking"""
        self.id = Value._next_id
        Value._next_id += 1

    @abstractmethod
    def infer_type(self) -> Union[int, ArrayType]:
        """Infer the type of this value.

        Returns:
            int: Protobuf ValueType enum (I32=0, F32=1, I1=2) for scalar values
            ArrayType: For array values with size and element type

        Examples:
            Constant(5).infer_type()           # Returns 0 (I32)
            Constant(3.14).infer_type()        # Returns 1 (F32)
            ArrayLiteral([1,2,3], ...).infer_type()  # Returns ArrayType instance
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
        from .serialization import SerializationContext
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
