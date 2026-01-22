"""C++ MLIR backend integration"""
import ctypes
from typing import Union
from .ast import Value
from .types import Type, ArrayType, type_to_proto, TYPE_TO_CTYPES

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

class CppMLIRBackend:
    """Schema-driven Python wrapper for C++ MLIR builder with ctypes execution"""

    def __init__(self):
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend not available. Build with CMake first.")

        self.builder = _mlir_backend.MLIRBuilder()
        self.builder.initialize_module()
        self.executor = _mlir_backend.MLIRExecutor()
        self.executor.initialize()

        # Set default optimization level (O2 = balanced performance)
        self.executor.set_optimization_level(2)

        # Cache for ctypes function wrappers
        self._function_cache = {}

    # ==================== CORE COMPILATION ====================
    def compile_function_from_ast(self, name: str, params: list,
                                   return_type: Type, ast_node: Value) -> None:
        """Compile complete function from AST - single protobuf entry point

        Args:
            name: Function name
            params: List of (param_name, type_spec) tuples where type_spec is Type
            return_type: Type instance (ScalarType or ArrayType)
            ast_node: Root AST node to compile
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        # Build complete FunctionDef protobuf message
        func_def = ast_pb2.FunctionDef()
        func_def.name = name

        # Add parameters with TypeSpec
        for param_name, param_type in params:
            param = func_def.params.add()
            param.name = param_name
            param.type.CopyFrom(type_to_proto(param_type))

        # Set return type using unified TypeSpec
        func_def.return_type.CopyFrom(type_to_proto(return_type))

        # Add function body (with SSA value reuse detection)
        func_def.body.CopyFrom(ast_node.to_proto_with_reuse())

        # Serialize and compile (errors propagate via exceptions)
        func_def_bytes = func_def.SerializeToString()
        self.builder.compile_function(func_def_bytes)

        # Build signature for executor
        sig = ast_pb2.FunctionSignature()
        sig.name = name

        # Add parameter types as TypeSpec
        for _, param_type in params:
            sig.param_types.append(type_to_proto(param_type))

        # Set return type
        sig.return_type.CopyFrom(type_to_proto(return_type))

        # Register signature with executor
        sig_bytes = sig.SerializeToString()
        self.executor.register_function_signature(sig_bytes)

    def execute_function(self, name: str, *args) -> Union[int, float, bool]:
        """Execute compiled function via JIT with ctypes

        Args:
            name: Function name to execute
            *args: Arguments in declaration order

        Returns:
            Execution result (int, float, or bool)
        """
        # Build ctypes function wrapper (cached)
        if name not in self._function_cache:
            # Get signature BEFORE clearing (it was registered during compile_function_from_ast)
            sig_bytes = self.executor.get_function_signature(name)
            sig = ast_pb2.FunctionSignature()
            sig.ParseFromString(sig_bytes)

            # Get LLVM IR and JIT compile
            llvm_ir = self.builder.get_llvm_ir_string()
            self.executor.clear()

            func_ptr = self.executor.compile_function(llvm_ir, name)
            if func_ptr is None:
                raise RuntimeError(f"JIT compilation failed: {self.executor.get_last_error()}")

            # Re-register signature (it was cleared by clear())
            self.executor.register_function_signature(sig_bytes)

            # Get function pointer as integer
            func_ptr_int = self.executor.get_function_pointer(name)

            # Determine return type from TypeSpec
            if sig.return_type.HasField('scalar'):
                # TYPE_TO_CTYPES uses ScalarTypeSpec.Kind directly
                c_return_type = TYPE_TO_CTYPES[sig.return_type.scalar.kind]
            elif sig.return_type.HasField('memref'):
                raise RuntimeError(
                    f"Cannot execute function '{name}' from Python: it returns an array type.\n"
                    f"Array-returning functions can only be called from within other @ml_function decorated functions.\n"
                    f"Hint: Create a wrapper function that extracts scalar values from the array."
                )
            else:
                raise RuntimeError(f"Function '{name}' has no return type specification")

            # Build ctypes function signature from TypeSpec param_types
            c_param_types = []
            for pt in sig.param_types:
                if pt.HasField('scalar'):
                    c_param_types.append(TYPE_TO_CTYPES[pt.scalar.kind])
                else:
                    raise RuntimeError("Array parameters not yet supported for JIT execution")

            # Create ctypes function wrapper
            func_type = ctypes.CFUNCTYPE(c_return_type, *c_param_types)
            ctypes_func = func_type(func_ptr_int)

            # Cache it
            self._function_cache[name] = ctypes_func

        # Call the cached function
        return self._function_cache[name](*args)

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
        self._function_cache.clear()

    def set_optimization_level(self, level: int):
        """Set LLVM optimization level

        Args:
            level: Optimization level (0=O0/none, 2=O2/default, 3=O3/aggressive)

        Note:
            Clears the function cache since optimization level affects compilation.
            Cached functions will be recompiled with new optimization level on next call.
        """
        if level not in [0, 2, 3]:
            raise ValueError(f"Invalid optimization level {level}. Must be 0, 2, or 3.")
        self.executor.set_optimization_level(level)
        self._function_cache.clear()

def get_backend():
    """Get the appropriate backend (C++ if available)"""
    global _global_builder

    if HAS_CPP_BACKEND:
        if _global_builder is None:
            _global_builder = CppMLIRBackend()
        return _global_builder
    else:
        return None