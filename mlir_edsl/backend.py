"""C++ MLIR backend integration"""
import ctypes
from typing import Union
from .types import Type, ScalarType, ArrayType, TensorType

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


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

# Populated once numpy is confirmed available (same shape as TYPE_TO_CTYPES)
SCALAR_TYPE_TO_NUMPY_DTYPE: dict = {}
if _HAS_NUMPY:
    SCALAR_TYPE_TO_NUMPY_DTYPE = {
        ScalarType.F32: np.float32,
        ScalarType.I32: np.int32,
        ScalarType.I1: np.bool_,
    }

_global_backend = None



def _make_output_descriptor(array_type) -> tuple:
    """Allocate a zeroed output buffer and build its memref descriptor.

    Returns (c_types, c_vals, buf) matching the standard MLIR memref descriptor
    layout. buf is Python-owned and must be kept alive through the ctypes call.
    Returns np.ndarray as buf when numpy is available, otherwise ctypes array.
    """
    _ELEM_CTYPES = {
        ScalarType.I32: ctypes.c_int32,
        ScalarType.F32: ctypes.c_float,
        ScalarType.I1: ctypes.c_bool,
    }
    shape = array_type.shape
    ndim = len(shape)

    # Row-major strides: shape (2,3,4) → strides (12, 4, 1)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    if _HAS_NUMPY:
        dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
        buf = np.empty(shape, dtype=dtype)
        ptr = buf.ctypes.data_as(ctypes.c_void_p)
    elif array_type.element_type.kind in _ELEM_CTYPES:
        elem_ctype = _ELEM_CTYPES[array_type.element_type.kind]
        total = array_type.total_elements
        buf = (elem_ctype * total)()  # zeroed, Python-owned
        ptr = ctypes.cast(buf, ctypes.c_void_p)
    else:
        raise TypeError(
            f"Unsupported element type {array_type.element_type} for output descriptor"
        )

    c_types = (
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        + [ctypes.c_int64] * ndim
        + [ctypes.c_int64] * ndim
    )
    assert not any(d == -1 for d in shape), (
        f"Internal error: DYN return type reached output descriptor for {array_type}. "
        "Abstract evaluation should have resolved concrete shapes before compilation."
    )
    c_vals = [ptr, ptr, 0] + list(shape) + strides

    return c_types, c_vals, buf


def _make_memref_descriptor(data, array_type) -> tuple:
    """Build (c_types, c_vals, buffer) for the standard MLIR memref descriptor.

    MLIR lowers memref<NxT> to individual LLVM args:
        alloc_ptr, aligned_ptr, offset, size0[, size1, ...], stride0[, stride1, ...]

    Accepts np.ndarray (zero-copy).
    The buffer must be kept alive by the caller until after the ctypes call.
    """
    shape = array_type.shape
    ndim = len(shape)

    if _HAS_NUMPY and isinstance(data, np.ndarray):
        expected_dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
        if data.dtype != expected_dtype:
            raise TypeError(
                f"ndarray dtype {data.dtype} does not match expected "
                f"{expected_dtype} for {array_type}"
            )
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        ptr = data.ctypes.data_as(ctypes.c_void_p)
        strides = [s // data.itemsize for s in data.strides]
        buf = data
    else:
        raise TypeError(
            f"Expected np.ndarray for array parameter, got {type(data).__name__}"
        )

    c_types = (
        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        + [ctypes.c_int64] * ndim
        + [ctypes.c_int64] * ndim
    )
    # Use data.shape for size fields — array_type.shape contains -1 for DYN
    # dimensions, but MLIR needs the actual runtime sizes in the descriptor.
    c_vals = [ptr, ptr, 0] + list(data.shape) + strides

    return c_types, c_vals, buf


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
        self._func_sources: dict[str, str] = {}

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
    def execute_function(self, name: str, *args) -> Union[int, float, bool, list]:
        """Execute compiled function via JIT with ctypes."""
        param_types, return_type = self._signatures[name]

        flat_c_types = []
        flat_args = []
        live_buffers = []  # Keep ctypes arrays alive through the call

        for pt, val in zip(param_types, args):
            if isinstance(pt, ScalarType):
                flat_c_types.append(TYPE_TO_CTYPES[pt.kind])
                flat_args.append(val)
            elif isinstance(pt, (ArrayType, TensorType)):
                c_types, c_vals, buf = _make_memref_descriptor(val, pt)
                flat_c_types.extend(c_types)
                flat_args.extend(c_vals)
                live_buffers.append(buf)
            else:
                raise RuntimeError(f"Unsupported parameter type {pt} for execution")

        ptr = self.compiler.get_function_pointer(name)

        if isinstance(return_type, (ArrayType, TensorType)):
            # Aggregate return: append Python-allocated output descriptor.
            out_c_types, out_c_vals, out_buf = _make_output_descriptor(return_type)
            flat_c_types.extend(out_c_types)
            flat_args.extend(out_c_vals)
            live_buffers.append(out_buf)
            ctypes.CFUNCTYPE(None, *flat_c_types)(ptr)(*flat_args)
            if _HAS_NUMPY and isinstance(out_buf, np.ndarray):
                return out_buf
            else:
                return _unflatten(list(out_buf), return_type.shape)
        elif isinstance(return_type, ScalarType):
            c_ret = self._type_to_ctype(return_type, "return type")
            return ctypes.CFUNCTYPE(c_ret, *flat_c_types)(ptr)(*flat_args)
        else:
            raise RuntimeError(
                f"execute_function: unhandled return type {return_type}"
            )

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
        """Get IR snapshots from lowering pipeline."""
        return self.compiler.get_lowering_snapshots()

    def enable_snapshot_capture(self):
        """Enable IR snapshot capture for lowering passes."""
        self.compiler.enable_snapshot_capture()

    def get_failure_ir(self) -> str:
        """IR captured at lowering failure point. Always available (no SAVE_IR needed)."""
        return self.compiler.get_failure_ir()

    def inject_test_failure(self) -> None:
        """Testing only: inject a malformed op to trigger lowering failure."""
        self.compiler.inject_test_failure()

    def clear_module(self):
        """Clear all functions and reset completely."""
        self.compiler.clear()
        self._signatures.clear()
        self._ast_dumps.clear()
        self._func_sources.clear()

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
