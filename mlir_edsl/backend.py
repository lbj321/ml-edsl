"""C++ MLIR backend integration"""
import ctypes
from typing import Union
from .types import Type, ScalarType


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

TYPE_TO_CTYPES = {
    ScalarType.I32: ctypes.c_int32,
    ScalarType.F32: ctypes.c_float,
    ScalarType.I1: ctypes.c_bool,
}

_global_builder = None


class CppMLIRBackend:
    """Schema-driven Python wrapper for C++ MLIR builder with ctypes execution.

    Two-phase compilation model:
    1. Definition phase: compile functions to MLIR, register signatures
    2. Execution phase: JIT compile module on first execute, then run from cache
    """

    def __init__(self):
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend not available. Build with CMake first.")

        self.builder = _mlir_backend.MLIRBuilder()
        self.builder.initialize_module()
        self.executor = _mlir_backend.MLIRExecutor()
        self.executor.initialize()
        self.executor.set_optimization_level(2)

    # ==================== COMPILATION HELPERS (PRIVATE) ====================
    def _build_function_def_proto(self, name: str, params: list,
                                   return_type: Type, ast_node) -> bytes:
        """Build and serialize FunctionDef protobuf."""
        func_def = ast_pb2.FunctionDef()
        func_def.name = name

        for param_name, param_type in params:
            param = func_def.params.add()
            param.name = param_name
            param.type.CopyFrom(param_type.to_proto())

        func_def.return_type.CopyFrom(return_type.to_proto())
        func_def.body.CopyFrom(ast_node.to_proto_with_reuse())

        return func_def.SerializeToString()

    def _build_signature_proto(self, name: str, params: list,
                                return_type: Type) -> bytes:
        """Build and serialize FunctionSignature protobuf."""
        sig = ast_pb2.FunctionSignature()
        sig.name = name

        for _, param_type in params:
            sig.param_types.append(param_type.to_proto())

        sig.return_type.CopyFrom(return_type.to_proto())

        return sig.SerializeToString()

    # ==================== CORE COMPILATION (DEFINITION PHASE) ====================
    def compile_function_from_ast(self, name: str, params: list,
                                   return_type: Type, ast_node) -> None:
        """Compile function from AST (definition phase).

        Adds function to MLIR module and registers signature.
        Invalidates JIT cache if previously finalized.
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        # Adding a new function invalidates existing JIT
        if not self.executor.is_jit_empty():
            self.executor.clear_jit()

        # Build and compile function definition
        func_def_bytes = self._build_function_def_proto(name, params, return_type, ast_node)
        self.builder.compile_function(func_def_bytes)

        # Register signature (persists across JIT clears)
        sig_bytes = self._build_signature_proto(name, params, return_type)
        self.executor.register_function_signature(sig_bytes)

    # ==================== FINALIZATION ====================
    def _ensure_finalized(self) -> None:
        """Ensure JIT compilation is done. Called automatically on first execute."""
        if not self.executor.is_jit_empty():
            return  # Already finalized

        llvm_ir = self.builder.get_llvm_ir_string()
        if not self.executor.compile_module(llvm_ir):
            raise RuntimeError(f"JIT compilation failed: {self.executor.get_last_error()}")

    def finalize(self) -> None:
        """Explicitly JIT compile all functions.

        Optional - happens automatically on first execute.
        Useful for measuring compile time or warming up before benchmarks.
        """
        self._ensure_finalized()

    # ==================== EXECUTION HELPERS (PRIVATE) ====================
    def _typespec_to_ctype(self, type_spec, context: str) -> type:
        """Convert protobuf TypeSpec to ctypes type."""
        if type_spec.HasField('scalar'):
            return TYPE_TO_CTYPES[type_spec.scalar.kind]
        elif type_spec.HasField('memref'):
            raise RuntimeError(f"Array types not supported for {context} in JIT execution")
        else:
            raise RuntimeError(f"Unknown type specification for {context}")

    def _build_ctypes_wrapper(self, func_ptr: int, sig) -> ctypes.CFUNCTYPE:
        """Build ctypes function wrapper from signature."""
        c_return_type = self._typespec_to_ctype(sig.return_type, "return type")

        c_param_types = []
        for i, pt in enumerate(sig.param_types):
            c_param_types.append(self._typespec_to_ctype(pt, f"parameter {i}"))

        func_type = ctypes.CFUNCTYPE(c_return_type, *c_param_types)
        return func_type(func_ptr)

    # ==================== JIT EXECUTION ====================
    def execute_function(self, name: str, *args) -> Union[int, float, bool]:
        """Execute compiled function via JIT with ctypes."""
        # Auto-finalize on first execute
        self._ensure_finalized()

        # Get signature and build wrapper
        sig_bytes = self.executor.get_function_signature(name)
        sig = ast_pb2.FunctionSignature()
        sig.ParseFromString(sig_bytes)

        func_ptr = self.executor.get_function_pointer(name)
        wrapper = self._build_ctypes_wrapper(func_ptr, sig)
        return wrapper(*args)

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
        """Clear all functions and reset completely."""
        self.builder.clear_module()
        self.executor.clear_all()

    def set_optimization_level(self, level: int):
        """Set LLVM optimization level.

        Invalidates JIT - will recompile on next execute.
        """
        if level not in [0, 2, 3]:
            raise ValueError(f"Invalid optimization level {level}. Must be 0, 2, or 3.")
        self.executor.set_optimization_level(level)
        self.executor.clear_jit()  # Signatures persist, just recompile


def get_backend():
    """Get the appropriate backend (C++ if available)"""
    global _global_builder

    if HAS_CPP_BACKEND:
        if _global_builder is None:
            _global_builder = CppMLIRBackend()
        return _global_builder
    else:
        return None
