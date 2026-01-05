"""Function-related AST nodes: Parameter, CallOp"""

from ..base import Value

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
        pb_node.parameter.value_type = self.value_type
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

        # Set return type based on type (oneof field)
        from mlir_edsl.types import ArrayType
        if isinstance(self.return_type, ArrayType):
            # Array return type - populate array_return field
            pb_node.call_op.array_return.shape.extend(self.return_type.shape)
            pb_node.call_op.array_return.element_type = self.return_type.element_enum
        else:
            # Scalar return type - populate scalar_return field
            pb_node.call_op.scalar_return = self.return_type

        # Context-aware child serialization
        if context:
            for arg in self.args:
                pb_node.call_op.args.append(arg._to_proto_impl(context))
        else:
            for arg in self.args:
                pb_node.call_op.args.append(arg.to_proto())

        return pb_node
