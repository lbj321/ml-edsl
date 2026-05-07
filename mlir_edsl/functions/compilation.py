"""Compilation and execution of ML functions"""
import ctypes
import os
from typing import Callable, Optional, Union

from ..ast import Value
from ..backend import (
    get_backend,
    _build_c_types_for_type,
    _build_flat_args_for_param,
    _make_output_descriptor,
    TYPE_TO_CTYPES,
)
from ..types import ScalarType
from .signature import FunctionSignature

_DUMP_AST = os.getenv("DUMP_AST", "").lower() in ("1", "true", "yes")

if _DUMP_AST:
    import shutil
    _ast_out = os.path.join(os.getcwd(), "ast_output")
    if os.path.exists(_ast_out):
        shutil.rmtree(_ast_out)


def _build_cfunc(name: str, signature: "FunctionSignature", backend):
    """Build and return (cfunc, ret_is_aggregate) once at compile time.

    Fetches the function pointer once (single pybind11 crossing) and wraps it
    in a CFUNCTYPE derived from the signature — both are invariant after compilation.
    """
    param_types = [signature.param_types[n] for n in signature.param_names]
    return_type = signature.return_type

    flat_c_types = []
    for pt in param_types:
        flat_c_types.extend(_build_c_types_for_type(pt))

    ret_is_aggregate = return_type.is_aggregate()
    if ret_is_aggregate:
        flat_c_types.extend(_build_c_types_for_type(return_type))
        c_ret = None
    elif isinstance(return_type, ScalarType):
        c_ret = TYPE_TO_CTYPES[return_type.kind]
    else:
        raise RuntimeError(f"_build_cfunc: unhandled return type {return_type}")

    ptr = backend.compiler.get_function_pointer(name)
    return ctypes.CFUNCTYPE(c_ret, *flat_c_types)(ptr), ret_is_aggregate


class CompiledFunction:
    """A compiled function ready for execution."""

    def __init__(self, name: str, signature: FunctionSignature, backend,
                 target: str = "cpu"):
        self.name = name
        self.signature = signature
        self._backend = backend
        self._target = target
        self._param_types = [signature.param_types[n] for n in signature.param_names]
        if target == "cpu":
            self._cfunc, self._ret_is_aggregate = _build_cfunc(name, signature, backend)
        else:
            self._cfunc = None
            self._ret_is_aggregate = None

    def call(self, ordered_args: list) -> Union[int, float, bool]:
        """Hot path: build flat_args only, invoke cached cfunc."""
        if self._target == "gpu":
            return self._backend.execute_gpu_function(self.name, *ordered_args)

        flat_args = []
        live_buffers = []

        for pt, val in zip(self._param_types, ordered_args):
            c_vals, buf = _build_flat_args_for_param(val, pt)
            flat_args.extend(c_vals)
            if buf is not None:
                live_buffers.append(buf)

        if self._ret_is_aggregate:
            _out_c_types, out_c_vals, out_buf = _make_output_descriptor(
                self.signature.return_type
            )
            flat_args.extend(out_c_vals)
            live_buffers.append(out_buf)
            self._cfunc(*flat_args)
            return out_buf

        return self._cfunc(*flat_args)

    def execute(self, args: tuple, kwargs: dict) -> Union[int, float, bool]:
        """Execute with runtime values."""
        self.signature.validate_runtime_args(args, kwargs)
        ordered_args = self.signature.order_args(args, kwargs)
        if self._target == "gpu":
            return self._backend.execute_gpu_function(self.name, *ordered_args)
        return self._backend.execute_function(self.name, *ordered_args)


def compile_function(signature: FunctionSignature, result_ast: Value,
                     target: str = "cpu") -> CompiledFunction:
    """Compile a function AST to MLIR.

    Args:
        signature: Parsed function signature with types
        result_ast: Pre-built AST from validation

    Returns:
        CompiledFunction ready for execution

    Raises:
        RuntimeError: If backend is not available
    """
    backend = get_backend()
    if backend is None:
        raise RuntimeError("C++ backend not available for JIT execution")

    # Already compiled? Return wrapper.
    if backend.has_function(signature.name):
        return CompiledFunction(signature.name, signature, backend)

    # Save AST dump before compilation
    if _DUMP_AST:
        _save_ast_dump(signature.name, result_ast)

    # Store AST dump on backend for HTML report (SAVE_IR=1)
    if os.getenv("SAVE_IR"):
        backend._ast_dumps[signature.name] = result_ast.dump()

    backend.set_target(target)

    # Compile to backend
    backend.compile_function_from_ast(
        signature.name,
        signature.make_param_list(),
        signature.return_type,
        result_ast
    )

    return CompiledFunction(signature.name, signature, backend, target=target)


def _save_ast_dump(name: str, ast: Value) -> None:
    """Save AST dump to ast_output/ directory."""
    out_dir = os.path.join(os.getcwd(), "ast_output")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.txt")
    if os.path.exists(path):
        counter = 1
        while os.path.exists(os.path.join(out_dir, f"{name}_{counter}.txt")):
            counter += 1
        path = os.path.join(out_dir, f"{name}_{counter}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ast.dump())
        f.write("\n")
