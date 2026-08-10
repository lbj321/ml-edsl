"""C++ MLIR backend integration"""
import ctypes
from typing import Union
import numpy as np
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

SCALAR_TYPE_TO_NUMPY_DTYPE = {
    ScalarType.F32: np.float32,
    ScalarType.I32: np.int32,
    ScalarType.I1: np.bool_,
}

_global_backend = None



# MLIR lowers a rank-N memref to individual flat LLVM scalar args, in this
# fixed order: alloc_ptr, aligned_ptr, offset, size_0..size_{N-1}, stride_0..
# stride_{N-1}. Every descriptor built in this module must follow this exact
# layout — defined once here so the ABI shape only needs to change in one
# place if it ever does.
_MEMREF_DESCRIPTOR_PREFIX_C_TYPES = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]


def _memref_descriptor_c_types(ndim: int) -> list:
    """ctypes types for a rank-`ndim` MLIR memref descriptor's flat scalar args."""
    return _MEMREF_DESCRIPTOR_PREFIX_C_TYPES + [ctypes.c_int64] * (2 * ndim)


def _memref_descriptor_tail(shape, strides) -> list:
    """The [sizes..., strides...] portion of a descriptor's c_vals — i.e.
    everything after the [alloc_ptr, aligned_ptr, offset] prefix."""
    return list(shape) + list(strides)


def _memref_descriptor_c_vals(ptr, tail: list, offset: int = 0) -> list:
    """Full c_vals for a memref descriptor: [alloc_ptr, aligned_ptr, offset]
    followed by `tail` (sizes + strides, see _memref_descriptor_tail)."""
    return [ptr, ptr, offset] + tail


def _row_major_strides(shape: tuple) -> list:
    """Row-major strides: shape (2,3,4) -> strides (12, 4, 1)."""
    ndim = len(shape)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _build_c_types_for_type(t: "Type") -> list:
    """Return the flat ctypes list for a single parameter or return type.

    Scalar → [c_int32 | c_float | c_bool]
    Aggregate → memref descriptor c_types, see _memref_descriptor_c_types.
    """
    if isinstance(t, ScalarType):
        return [TYPE_TO_CTYPES[t.kind]]
    if t.is_aggregate():
        return _memref_descriptor_c_types(len(t.shape))
    raise RuntimeError(f"_build_c_types_for_type: unsupported type {t}")


def _make_output_descriptor(array_type) -> tuple:
    """Allocate a zeroed output buffer and build its memref descriptor.

    Returns (c_types, c_vals, buf) matching the standard MLIR memref descriptor
    layout. buf is a numpy array, Python-owned and must be kept alive through
    the ctypes call.
    """
    shape = array_type.shape
    strides = _row_major_strides(shape)

    dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
    buf = np.empty(shape, dtype=dtype)
    ptr = buf.ctypes.data_as(ctypes.c_void_p)

    assert not any(d == -1 for d in shape), (
        f"Internal error: DYN return type reached output descriptor for {array_type}. "
        "Abstract evaluation should have resolved concrete shapes before compilation."
    )
    c_types = _memref_descriptor_c_types(len(shape))
    tail = _memref_descriptor_tail(shape, strides)
    c_vals = _memref_descriptor_c_vals(ptr, tail)

    return c_types, c_vals, buf


def _precompute_layout(array_type) -> tuple:
    """Precompute the shape/dtype/descriptor-tail for a shape-specialized
    compiled variant. Shape and strides are compile-time constants once a
    function has been specialized to concrete argument shapes, so this only
    needs to run once per CompiledFunction rather than on every call."""
    shape = tuple(array_type.shape)
    strides = _row_major_strides(shape)
    dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[array_type.element_type.kind]
    tail = _memref_descriptor_tail(shape, strides)
    return shape, dtype, tail


def _build_flat_args_for_param_fast(val, layout) -> tuple:
    """Hot-path version of _build_flat_args_for_param using a precomputed
    layout (shape, dtype, tail) from _precompute_layout. layout is None for
    scalar parameters."""
    if layout is None:
        return [val], None
    shape, dtype, tail = layout
    if val.dtype != dtype:
        raise TypeError(
            f"ndarray dtype {val.dtype} does not match expected {dtype}"
        )
    if not val.flags['C_CONTIGUOUS']:
        val = np.ascontiguousarray(val)
    ptr = val.ctypes.data_as(ctypes.c_void_p)
    return _memref_descriptor_c_vals(ptr, tail), val


def _make_output_descriptor_fast(layout) -> tuple:
    """Hot-path version of _make_output_descriptor using a precomputed
    layout (shape, dtype, tail) from _precompute_layout."""
    shape, dtype, tail = layout
    buf = np.empty(shape, dtype=dtype)
    ptr = buf.ctypes.data_as(ctypes.c_void_p)
    return _memref_descriptor_c_vals(ptr, tail), buf


def _make_memref_descriptor(data, array_type) -> tuple:
    """Build (c_types, c_vals, buffer) for the standard MLIR memref descriptor.

    MLIR lowers memref<NxT> to individual LLVM args:
        alloc_ptr, aligned_ptr, offset, size0[, size1, ...], stride0[, stride1, ...]

    Accepts np.ndarray (zero-copy).
    The buffer must be kept alive by the caller until after the ctypes call.
    """
    shape = array_type.shape

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

    c_types = _memref_descriptor_c_types(len(shape))
    # Use data.shape for size fields — array_type.shape contains -1 for DYN
    # dimensions, but MLIR needs the actual runtime sizes in the descriptor.
    tail = _memref_descriptor_tail(data.shape, strides)
    c_vals = _memref_descriptor_c_vals(ptr, tail)

    return c_types, c_vals, data


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
    def execute_function(self, name: str, *args) -> Union[int, float, bool, np.ndarray]:
        """Execute compiled function via JIT with ctypes."""
        param_types, return_type = self._signatures[name]

        flat_c_types = []
        flat_args = []
        live_buffers = []  # Keep arrays alive through the call

        for pt, val in zip(param_types, args):
            if isinstance(pt, ScalarType):
                flat_c_types.append(TYPE_TO_CTYPES[pt.kind])
                flat_args.append(val)
            elif pt.is_aggregate():
                c_types, c_vals, buf = _make_memref_descriptor(val, pt)
                flat_c_types.extend(c_types)
                flat_args.extend(c_vals)
                live_buffers.append(buf)
            else:
                raise RuntimeError(f"Unsupported parameter type {pt} for execution")

        ptr = self.compiler.get_function_pointer(name)

        if return_type.is_aggregate():
            # Aggregate return: append Python-allocated output descriptor.
            out_c_types, out_c_vals, out_buf = _make_output_descriptor(return_type)
            flat_c_types.extend(out_c_types)
            flat_args.extend(out_c_vals)
            live_buffers.append(out_buf)
            ctypes.CFUNCTYPE(None, *flat_c_types)(ptr)(*flat_args)
            return out_buf
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

    def get_unopt_llvm_ir(self) -> str:
        """Unoptimized LLVM IR from last compilation; empty if SAVE_IR not set."""
        return self.compiler.get_unopt_llvm_ir()

    def get_opt_llvm_ir(self) -> str:
        """Optimized LLVM IR from last compilation; empty if SAVE_IR not set."""
        return self.compiler.get_opt_llvm_ir()

    def get_function_pointer(self, name: str) -> int:
        """Get JIT-compiled function pointer, triggering finalization if needed."""
        return self.compiler.get_function_pointer(name)

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

    def set_target(self, target: str) -> None:
        """Set compilation target: 'cpu' or 'gpu'."""
        if target not in ("cpu", "gpu"):
            raise ValueError(f"Invalid target '{target}'. Must be 'cpu' or 'gpu'.")
        self.compiler.set_target(target)

    def execute_gpu_function(self, name: str, *args) -> np.ndarray:
        """Execute a GPU-compiled function with automatic H2D/D2H transfers."""
        param_types, return_type = self._signatures[name]

        # Build (data_ptr, shape) pairs for each input
        inputs = []
        live_buffers = []
        for pt, val in zip(param_types, args):
            if pt.is_aggregate():
                arr = val if val.flags['C_CONTIGUOUS'] else np.ascontiguousarray(val)
                live_buffers.append(arr)
                inputs.append((arr.ctypes.data, list(arr.shape)))
            else:
                raise RuntimeError(f"GPU execution only supports array/tensor params, got {pt}")

        out_shape = list(return_type.shape)
        dtype = SCALAR_TYPE_TO_NUMPY_DTYPE[return_type.element_type.kind]
        element_size = np.dtype(dtype).itemsize

        raw = self.compiler.execute_gpu_function(
            name, inputs, out_shape, element_size
        )
        result = np.frombuffer(raw, dtype=dtype).reshape(out_shape).copy()
        return result


def get_backend():
    """Get the appropriate backend (C++ if available)"""
    global _global_backend

    if HAS_CPP_BACKEND:
        if _global_backend is None:
            _global_backend = CppMLIRBackend()
        return _global_backend
    else:
        return None
