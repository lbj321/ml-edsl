"""C++ MLIR backend integration"""
from typing import Union
from .ast import Value
from .types import I32, F32, I1

try:
    from . import _mlir_backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

try:
    from . import ast_pb2
    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False
    ast_pb2 = None

# Global instance to keep the builder alive
_global_builder = None

class MLIRValue(Value):
    """Wrapper for C++ MLIR values"""

    def __init__(self, cpp_value, builder, value_type: int):
        """
        Args:
            cpp_value: C++ mlir::Value
            builder: CppMLIRBuilder instance
            value_type: Protobuf ValueType enum (I32=0, F32=1, I1=2)
        """
        self.cpp_value = cpp_value
        self.builder = builder
        self.value_type = value_type  # Store as enum int

    @property
    def type(self):
        """Backward compatibility: returns enum"""
        return self.value_type

    def to_proto(self):
        """MLIRValue cannot be serialized to protobuf - it's already compiled MLIR"""
        raise RuntimeError("MLIRValue is already compiled MLIR and cannot be serialized to protobuf")

    def to_mlir(self, ssa_counter: dict) -> tuple[list[str], str]:
        """This should not be called when using C++ backend"""
        raise RuntimeError("to_mlir() should not be called with C++ backend")

class CppMLIRBuilder:
    """Minimal schema-driven Python wrapper for C++ MLIR builder

    All MLIR construction happens in C++ via protobuf AST serialization.
    This wrapper only exposes essential compilation and inspection methods.
    """

    def __init__(self):
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend not available. Build with CMake first.")

        self.builder = _mlir_backend.MLIRBuilder()
        self.builder.initialize_module()
        self.executor = _mlir_backend.MLIRExecutor()
        self.executor.initialize()

    # ==================== CORE COMPILATION ====================
    def compile_function_from_ast(self, name: str, params: list,
                                   return_type: int, ast_node: Value) -> None:
        """Compile complete function from AST - single protobuf entry point

        Args:
            name: Function name
            params: List of (param_name, ValueType_enum) tuples using ast_pb2 enums
            return_type: ValueType enum (ast_pb2.I32/F32/I1)
            ast_node: Root AST node to compile
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        # Build complete FunctionDef protobuf message
        func_def = ast_pb2.FunctionDef()
        func_def.name = name
        func_def.return_type = return_type  # Direct enum assignment!

        # Add parameters with enum types
        for param_name, param_type in params:
            param = func_def.params.add()
            param.name = param_name
            param.type = param_type  # Direct enum assignment!

        # Add function body
        func_def.body.CopyFrom(ast_node.to_proto())

        # Serialize everything to single buffer
        buffer = func_def.SerializeToString()

        # Single C++ call with single buffer - full schema enforcement!
        self.builder.compile_function(buffer)

    def execute_function(self, name: str, result: Value,
                        int_args: list = None, float_args: list = None) -> Union[int, float]:
        """Execute compiled function via JIT

        Args:
            name: Function name to execute
            result: AST node for type inference
            int_args: Integer arguments for function
            float_args: Float arguments for function

        Returns:
            Execution result (int or float)
        """
        llvm_ir = self.builder.get_llvm_ir_string()
        self.executor.clear()

        func_ptr = self.executor.compile_function(llvm_ir, name)
        if func_ptr is None:
            raise RuntimeError(f"JIT compilation failed: {self.executor.get_last_error()}")

        # Get return type directly from AST
        result_type = result.infer_type()

        int_args = int_args or []
        float_args = float_args or []

        # Call appropriate JIT function variant
        if result_type == I32:
            return self.executor.call_int32_function(func_ptr, int_args, float_args)
        elif result_type == F32:
            return self.executor.call_float_function(func_ptr, int_args, float_args)
        else:
            raise RuntimeError(f"Unsupported return type enum: {result_type}")

    # ==================== INSPECTION ====================
    def get_mlir_string(self) -> str:
        """Get generated MLIR IR as string"""
        return self.builder.get_mlir_string()

    def get_llvm_ir_string(self) -> str:
        """Get generated LLVM IR as string"""
        return self.builder.get_llvm_ir_string()

    # ==================== MANAGEMENT ====================
    def has_function(self, name: str) -> bool:
        """Check if function is already compiled"""
        return self.builder.has_function(name)

    def list_functions(self) -> list[str]:
        """Get names of all compiled functions"""
        return self.builder.list_functions()

    def clear_module(self):
        """Clear all functions from module"""
        self.builder.clear_module()

def get_backend():
    """Get the appropriate backend (C++ if available, fallback to string-based)"""
    global _global_builder
    
    if HAS_CPP_BACKEND:
        if _global_builder is None:
            _global_builder = CppMLIRBuilder()
        return _global_builder
    else:
        return None