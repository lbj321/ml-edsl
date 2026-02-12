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

_global_backend = None


class CppMLIRBackend:
    """Python wrapper for unified C++ MLIRCompiler.

    Two-phase compilation model:
    1. Building phase: compile functions to MLIR via compileFunction()
    2. Execution phase: auto-finalizes on first getFunctionPointer() call
    """

    def __init__(self):
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend not available. Build with CMake first.")

        self.compiler = _mlir_backend.MLIRCompiler()
        self._signatures: dict[str, tuple[list[Type], Type]] = {}
        self._ast_dumps: dict[str, str] = {}

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

    # ==================== CORE COMPILATION (DEFINITION PHASE) ====================
    def compile_function_from_ast(self, name: str, params: list,
                                   return_type: Type, ast_node) -> None:
        """Compile function from AST (definition phase).

        Adds function to MLIR module and registers signature.
        """
        if not HAS_PROTOBUF:
            raise RuntimeError("Protobuf not available. Run ./build.sh")

        func_def_bytes = self._build_function_def_proto(name, params, return_type, ast_node)
        self.compiler.compile_function(func_def_bytes)
        self._signatures[name] = ([pt for _, pt in params], return_type)

    # ==================== EXECUTION HELPERS (PRIVATE) ====================
    @staticmethod
    def _type_to_ctype(t: Type, context: str) -> type:
        """Convert a Type to its ctypes equivalent."""
        if isinstance(t, ScalarType):
            return TYPE_TO_CTYPES[t.kind]
        raise RuntimeError(f"Aggregate types not supported for {context} in JIT execution")

    # ==================== JIT EXECUTION ====================
    def execute_function(self, name: str, *args) -> Union[int, float, bool]:
        """Execute compiled function via JIT with ctypes."""
        param_types, return_type = self._signatures[name]
        c_ret = self._type_to_ctype(return_type, "return type")
        c_params = [self._type_to_ctype(pt, f"parameter {i}") for i, pt in enumerate(param_types)]
        func_type = ctypes.CFUNCTYPE(c_ret, *c_params)
        wrapper = func_type(self.compiler.get_function_pointer(name))
        return wrapper(*args)

    # ==================== MANAGEMENT ====================
    def has_function(self, name: str) -> bool:
        """Check if function is already compiled."""
        return self.compiler.has_function(name)

    def list_functions(self) -> list[str]:
        """Get names of all compiled functions."""
        return self.compiler.list_functions()

    def get_module_ir(self) -> str:
        """Get current MLIR module IR as string."""
        return self.compiler.get_module_ir()

    def get_lowering_snapshots(self) -> list[tuple[str, str]]:
        """Get IR snapshots from lowering pipeline. Only populated with SAVE_IR=1."""
        return self.compiler.get_lowering_snapshots()

    def clear_module(self):
        """Clear all functions and reset completely."""
        self.compiler.clear()
        self._signatures.clear()
        self._ast_dumps.clear()

    def set_optimization_level(self, level: int):
        """Set LLVM optimization level."""
        if level not in [0, 2, 3]:
            raise ValueError(f"Invalid optimization level {level}. Must be 0, 2, or 3.")
        self.compiler.set_optimization_level(level)


def get_backend():
    """Get the appropriate backend (C++ if available)"""
    global _global_backend

    if HAS_CPP_BACKEND:
        if _global_backend is None:
            _global_backend = CppMLIRBackend()
        return _global_backend
    else:
        return None
