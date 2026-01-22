"""Function-related AST nodes: Parameter, CallOp"""

from ..base import Value
from ...types import type_to_proto

# Import generated protobuf code
try:
    from ... import ast_pb2
except ImportError:
    ast_pb2 = None

from ..serialization import SerializationContext


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
        pb_node.parameter.type.CopyFrom(type_to_proto(self.value_type))
        # Value fields are unused by C++ backend (uses parameterMap instead)
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

        # Set return type using unified TypeSpec
        pb_node.call_op.return_type.CopyFrom(type_to_proto(self.return_type))

        # Context-aware child serialization
        for arg in self.args:
            pb_node.call_op.args.append(arg._to_proto_impl(context))

        return pb_node
