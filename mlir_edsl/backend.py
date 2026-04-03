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

# ---------------------------------------------------------------------------
# _mlir_ciface_ calling convention helpers
#
# MLIR's C interface (emitCWrappers=true) passes every memref argument as a
# pointer to a StridedMemRefType struct rather than as 2*rank+3 flat fields.
# Aggregate returns are also passed as a StridedMemRefType* prepended as the
# first argument.
# ---------------------------------------------------------------------------

_STRIDED_MEMREF_CACHE: dict[int, type] = {}


def _strided_memref_type(ndim: int) -> type:
    """Return a ctypes.Structure class for StridedMemRefType<T, ndim>.

    Layout matches MLIR's C interface:
        basePtr   : c_void_p
        data      : c_void_p  ← data pointer Python cares about
        offset    : c_int64   ← always 0 for contiguous arrays
        size[N]   : c_int64 × ndim
        stride[N] : c_int64 × ndim
    """
    if ndim not in _STRIDED_MEMREF_CACHE:
        fields = (
            [("basePtr", ctypes.c_void_p), ("data", ctypes.c_void_p),
             ("offset", ctypes.c_int64)]
            + [(f"size{i}",   ctypes.c_int64) for i in range(ndim)]
            + [(f"stride{i}", ctypes.c_int64) for i in range(ndim)]
        )
        _STRIDED_MEMREF_CACHE[ndim] = type(
            f"StridedMemRef{ndim}D", (ctypes.Structure,), {"_fields_": fields}
        )
    return _STRIDED_MEMREF_CACHE[ndim]

_global_backend = None



def _make_output_descriptor(array_type) -> tuple:
    """Allocate a zeroed output buffer and build its StridedMemRefType struct.

    Returns (c_void_p arg, struct, buf).
    buf is Python-owned and must be kept alive through the ctypes call.
    """
    shape = array_type.shape
    ndim  = len(shape)

    assert not any(d == -1 for d in shape), (
        f"Internal error: DYN return type reached output descriptor for {array_type}. "
        "Abstract evaluation should have resolved concrete shapes before compilation."
    )

    # Row-major strides: shape (2,3,4) → strides (12, 4, 1)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
    buf   = np.empty(shape, dtype=dtype)
    ptr   = buf.ctypes.data_as(ctypes.c_void_p)

    StructType = _strided_memref_type(ndim)
    desc = StructType(basePtr=ptr.value, data=ptr.value, offset=0)
    for i, s in enumerate(shape):   setattr(desc, f"size{i}",   s)
    for i, s in enumerate(strides): setattr(desc, f"stride{i}", s)

    return ctypes.c_void_p(ctypes.addressof(desc)), desc, buf


def _make_memref_descriptor(data, array_type) -> tuple:
    """Build a StridedMemRefType struct for a memref input argument.

    Returns (c_void_p arg, struct, buffer).
    Accepts np.ndarray (zero-copy). buffer must be kept alive by the caller.
    """
    if not (_HAS_NUMPY and isinstance(data, np.ndarray)):
        raise TypeError(
            f"Expected np.ndarray for array parameter, got {type(data).__name__}"
        )

    expected_dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
    if data.dtype != expected_dtype:
        raise TypeError(
            f"ndarray dtype {data.dtype} does not match expected "
            f"{expected_dtype} for {array_type}"
        )
    if not data.flags['C_CONTIGUOUS']:
        data = np.ascontiguousarray(data)

    ndim    = len(array_type.shape)
    ptr     = data.ctypes.data_as(ctypes.c_void_p)
    strides = [s // data.itemsize for s in data.strides]

    StructType = _strided_memref_type(ndim)
    desc = StructType(basePtr=ptr.value, data=ptr.value, offset=0)
    # Use data.shape for size fields — array_type.shape may contain -1 for DYN dims.
    for i, s in enumerate(data.shape): setattr(desc, f"size{i}",   s)
    for i, s in enumerate(strides):    setattr(desc, f"stride{i}", s)

    return ctypes.c_void_p(ctypes.addressof(desc)), desc, data


def _make_result_dummy(array_type) -> tuple:
    """Zero-initialised StridedMemRefType struct for the _mlir_ciface_ phantom return.

    Returns (c_void_p arg, struct). Python ignores the struct's contents after the call
    since the actual result was written directly into the out-param buffer.
    """
    ndim = len(array_type.shape)
    StructType = _strided_memref_type(ndim)
    desc = StructType()   # all fields zero
    return ctypes.c_void_p(ctypes.addressof(desc)), desc


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
        """Execute compiled function via JIT using the _mlir_ciface_ convention.

        All memref arguments are passed as StridedMemRefType* pointers.
        Aggregate (tensor/array) returns use a phantom result pointer prepended
        as the first argument; Python reads the result from the out-param buffer.
        Pure-scalar functions are called directly with the plain function name.
        """
        param_types, return_type = self._signatures[name]
        live_buffers = []
        ptr = self.compiler.get_function_pointer(name)

        if isinstance(return_type, (ArrayType, TensorType)):
            # _mlir_ciface_ signature:
            #   void fn(StridedMemRef* result, StridedMemRef* inputs..., StridedMemRef* out_param)
            # result is a phantom — Python ignores it; out_param receives the data.
            res_arg, res_desc          = _make_result_dummy(return_type)
            out_arg, out_desc, out_buf = _make_output_descriptor(return_type)

            call_c_types = [ctypes.c_void_p]   # result ptr (prepended first)
            call_args    = [res_arg]

            for pt, val in zip(param_types, args):
                if isinstance(pt, ScalarType):
                    call_c_types.append(TYPE_TO_CTYPES[pt.kind])
                    call_args.append(val)
                elif isinstance(pt, (ArrayType, TensorType)):
                    desc_arg, desc_struct, _ = _make_memref_descriptor(val, pt)
                    call_c_types.append(ctypes.c_void_p)
                    call_args.append(desc_arg)
                    live_buffers.append(desc_struct)
                else:
                    raise RuntimeError(f"Unsupported parameter type {pt} for execution")

            call_c_types.append(ctypes.c_void_p)   # out-param ptr (appended last)
            call_args.append(out_arg)
            live_buffers.extend([res_desc, out_desc, out_buf])

            ctypes.CFUNCTYPE(None, *call_c_types)(ptr)(*call_args)
            return out_buf

        elif isinstance(return_type, ScalarType):
            # Scalar return: plain function or _mlir_ciface_ wrapper (both return scalar directly).
            # Memref inputs still use struct pointers via _mlir_ciface_.
            call_c_types = []
            call_args    = []
            for pt, val in zip(param_types, args):
                if isinstance(pt, ScalarType):
                    call_c_types.append(TYPE_TO_CTYPES[pt.kind])
                    call_args.append(val)
                elif isinstance(pt, (ArrayType, TensorType)):
                    desc_arg, desc_struct, _ = _make_memref_descriptor(val, pt)
                    call_c_types.append(ctypes.c_void_p)
                    call_args.append(desc_arg)
                    live_buffers.append(desc_struct)
                else:
                    raise RuntimeError(f"Unsupported parameter type {pt} for execution")
            c_ret = TYPE_TO_CTYPES[return_type.kind]
            return ctypes.CFUNCTYPE(c_ret, *call_c_types)(ptr)(*call_args)

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
