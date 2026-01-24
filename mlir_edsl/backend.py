"""C++ MLIR backend integration"""
import ctypes
from typing import Union
from .types import Type, ScalarType, type_to_proto

# ctypes mapping for JIT execution (backend-specific)
TYPE_TO_CTYPES = {
    ScalarType.I32: ctypes.c_int32,
    ScalarType.F32: ctypes.c_float,
    ScalarType.I1: ctypes.c_bool,
}

try:
    from . import _mlir_backend
except ImportError:
    _mlir_backend = None

try:
    from . import ast_pb2
except ImportError:
    ast_pb2 = None

HAS_CPP_BACKEND = _mlir_backend is not None
HAS_PROTOBUF = ast_pb2 is not None

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

    # ==================== COMPILATION HELPERS (PRIVATE) ====================
    def _build_function_def_proto(self, name: str, params: list,
                                   return_type: Type, ast_node) -> bytes:
        """Build and serialize FunctionDef protobuf.

        Args:
            name: Function name
            params: List of (param_name, param_type) tuples
            return_type: Function return type
            ast_node: Root AST node for function body

        Returns:
            Serialized FunctionDef protobuf bytes
        """
        func_def = ast_pb2.FunctionDef()
        func_def.name = name

        for param_name, param_type in params:
            param = func_def.params.add()
            param.name = param_name
            param.type.CopyFrom(type_to_proto(param_type))

        func_def.return_type.CopyFrom(type_to_proto(return_type))
        func_def.body.CopyFrom(ast_node.to_proto_with_reuse())

        return func_def.SerializeToString()

    def _build_signature_proto(self, name: str, params: list,
                                return_type: Type) -> bytes:
        """Build and serialize FunctionSignature protobuf.

        Args:
            name: Function name
            params: List of (param_name, param_type) tuples
            return_type: Function return type

        Returns:
            Serialized FunctionSignature protobuf bytes
        """
        sig = ast_pb2.FunctionSignature()
        sig.name = name

        for _, param_type in params:
            sig.param_types.append(type_to_proto(param_type))

        sig.return_type.CopyFrom(type_to_proto(return_type))

        return sig.SerializeToString()

    # ==================== CORE COMPILATION ====================
    def compile_function_from_ast(self, name: str, params: list,
                                   return_type: Type, ast_node) -> None:
        """Compile complete function from AST.

        Args:
            name: Function name
            params: List of (param_name, type_spec) tuples where type_spec is Type
            return_type: Type instance (ScalarType or ArrayType)
            ast_node: Root AST node to compile
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        # Build and compile function definition
        func_def_bytes = self._build_function_def_proto(name, params, return_type, ast_node)
        self.builder.compile_function(func_def_bytes)

        # Register signature with executor (needed for JIT execution)
        sig_bytes = self._build_signature_proto(name, params, return_type)
        self.executor.register_function_signature(sig_bytes)

    # ==================== EXECUTION HELPERS (PRIVATE) ====================
    def _typespec_to_ctype(self, type_spec, context: str) -> type:
        """Convert protobuf TypeSpec to ctypes type.

        Args:
            type_spec: Protobuf TypeSpec message
            context: Description for error messages (e.g., "return type", "parameter 0")

        Returns:
            Corresponding ctypes type (c_int32, c_float, c_bool)

        Raises:
            RuntimeError: If type is not supported for JIT execution
        """
        if type_spec.HasField('scalar'):
            return TYPE_TO_CTYPES[type_spec.scalar.kind]
        elif type_spec.HasField('memref'):
            if "return" in context:
                raise RuntimeError(
                    f"Cannot execute function from Python: it returns an array type.\n"
                    f"Array-returning functions can only be called from within other "
                    f"@ml_function decorated functions.\n"
                    f"Hint: Create a wrapper function that extracts scalar values from the array."
                )
            else:
                raise RuntimeError("Array parameters not yet supported for JIT execution")
        else:
            raise RuntimeError(f"Unknown type specification for {context}")

    def _jit_compile_and_get_pointer(self, name: str, sig_bytes: bytes) -> int:
        """JIT compile function and return function pointer.

        Handles the "signature dance" required because executor.clear() wipes
        the signature registry. We must:
        1. Get LLVM IR from builder
        2. Clear executor state (required before new JIT compilation)
        3. JIT compile the LLVM IR
        4. Re-register the signature (lost during clear)
        5. Return the function pointer

        Args:
            name: Function name
            sig_bytes: Serialized FunctionSignature protobuf

        Returns:
            Function pointer as integer

        Raises:
            RuntimeError: If JIT compilation fails
        """
        llvm_ir = self.builder.get_llvm_ir_string()
        self.executor.clear()

        func_ptr = self.executor.compile_function(llvm_ir, name)
        if func_ptr is None:
            raise RuntimeError(f"JIT compilation failed: {self.executor.get_last_error()}")

        self.executor.register_function_signature(sig_bytes)
        return self.executor.get_function_pointer(name)

    def _build_ctypes_wrapper(self, func_ptr: int, sig) -> ctypes.CFUNCTYPE:
        """Build ctypes function wrapper from signature.

        Args:
            func_ptr: Function pointer as integer
            sig: Parsed FunctionSignature protobuf

        Returns:
            Callable ctypes function wrapper

        Raises:
            RuntimeError: If types are not supported for JIT execution
        """
        c_return_type = self._typespec_to_ctype(sig.return_type, "return type")

        c_param_types = []
        for i, pt in enumerate(sig.param_types):
            c_param_types.append(self._typespec_to_ctype(pt, f"parameter {i}"))

        func_type = ctypes.CFUNCTYPE(c_return_type, *c_param_types)
        return func_type(func_ptr)

    # ==================== JIT EXECUTION ====================
    def execute_function(self, name: str, *args) -> Union[int, float, bool]:
        """Execute compiled function via JIT with ctypes.

        Args:
            name: Function name to execute
            *args: Arguments in declaration order

        Returns:
            Execution result (int, float, or bool)
        """
        if name not in self._function_cache:
            # Get signature (registered during compile_function_from_ast)
            sig_bytes = self.executor.get_function_signature(name)
            sig = ast_pb2.FunctionSignature()
            sig.ParseFromString(sig_bytes)

            # JIT compile and get function pointer
            func_ptr = self._jit_compile_and_get_pointer(name, sig_bytes)

            # Build and cache ctypes wrapper
            self._function_cache[name] = self._build_ctypes_wrapper(func_ptr, sig)

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