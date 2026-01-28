"""Function-related AST nodes: Parameter, CallOp"""

from typing import TYPE_CHECKING
from ..base import Value
from ...types import Type

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext

if TYPE_CHECKING:
    from ...types import Type


class Parameter(Value):
    """Represents a named parameter"""

    def __init__(self, name: str, value_type: Type):
        super().__init__()
        self.name = name
        self.value_type = value_type

    def infer_type(self) -> Type:
        """Parameters have declared types"""
        return self.value_type

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.function.parameter.name = self.name
        pb_node.function.parameter.type.CopyFrom(self.value_type.to_proto())
        return pb_node


class CallOp(Value):
    """Represents a function call operation"""

    def __init__(self, func_name: str, args: list[Value], return_type: Type):
        super().__init__()
        self.func_name = func_name
        self.args = args
        self.return_type = return_type

    def infer_type(self) -> Type:
        """Return type is explicitly declared"""
        return self.return_type

    def get_children(self) -> list['Value']:
        return self.args

    def _serialize_node(self, context: 'SerializationContext'):
        pb_node = ast_pb2.ASTNode()
        pb_node.function.call.func_name = self.func_name
        pb_node.function.call.return_type.CopyFrom(self.return_type.to_proto())
        for arg in self.args:
            pb_node.function.call.args.append(arg.to_proto(context))
        return pb_node
