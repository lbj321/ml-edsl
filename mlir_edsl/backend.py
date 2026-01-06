"""C++ MLIR backend integration"""
import ctypes
from typing import Union
from .ast import Value
from .types import TYPE_TO_CTYPES

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
                                   return_type: Union[int, ArrayType], ast_node: Value) -> None:
        """Compile complete function from AST - single protobuf entry point

        Args:
            name: Function name
            params: List of (param_name, ValueType_enum) tuples using ast_pb2 enums
            return_type: ValueType enum (ast_pb2.I32/F32/I1) OR ArrayType instance
            ast_node: Root AST node to compile
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        # Build complete FunctionDef protobuf message
        func_def = ast_pb2.FunctionDef()
        func_def.name = name

        # Add parameters with enum types
        for param_name, param_type in params:
            param = func_def.params.add()
            param.name = param_name
            param.type = param_type  # Direct enum assignment!

        # Set return type based on type (oneof field)
        from .types import ArrayType
        if isinstance(return_type, ArrayType):
            # Array return type - populate array_return field
            func_def.array_return.shape.extend(return_type.shape)
            func_def.array_return.element_type = return_type.element_enum
        elif isinstance(return_type, int):
            # Scalar return type - populate scalar_return field
            func_def.scalar_return = return_type
        else:
            raise TypeError(
                f"Invalid return type: {type(return_type).__name__}. "
                f"Expected int (scalar enum) or ArrayType instance."
            )

        # Add function body (with SSA value reuse detection)
        func_def.body.CopyFrom(ast_node.to_proto_with_reuse())

        # Serialize and compile (returns FunctionSignature protobuf)
        func_def_bytes = func_def.SerializeToString()
        sig_bytes = self.builder.compile_function(func_def_bytes)

        # Register signature with executor
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

            # Determine return type from oneof field
            if sig.HasField('scalar_return'):
                c_return_type = TYPE_TO_CTYPES[sig.scalar_return]
            elif sig.HasField('array_return'):
                raise RuntimeError(
                    f"Cannot execute function '{name}' from Python: it returns an array type.\n"
                    f"Array-returning functions can only be called from within other @ml_function decorated functions.\n"
                    f"Hint: Create a wrapper function that extracts scalar values from the array."
                )
            else:
                raise RuntimeError(f"Function '{name}' has no return type specification")

            # Build ctypes function signature
            c_param_types = [TYPE_TO_CTYPES[pt] for pt in sig.param_types]

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