"""Compilation and execution of ML functions"""
from typing import Callable, Optional, Union

from ..ast import Value
from ..backend import get_backend
from .signature import FunctionSignature


class CompiledFunction:
    """A compiled function ready for execution."""

    def __init__(self, name: str, signature: FunctionSignature, backend):
        self.name = name
        self.signature = signature
        self._backend = backend

    def execute(self, args: tuple, kwargs: dict) -> Union[int, float, bool]:
        """Execute with runtime values."""
        self.signature.validate_runtime_args(args, kwargs)
        ordered_args = self.signature.order_args(args, kwargs)
        return self._backend.execute_function(self.name, *ordered_args)


def compile_function(signature: FunctionSignature, result_ast: Value) -> CompiledFunction:
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

    # Compile to backend
    backend.compile_function_from_ast(
        signature.name,
        signature.make_param_list(),
        signature.return_type,
        result_ast
    )

    return CompiledFunction(signature.name, signature, backend)
