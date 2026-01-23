"""Compilation and execution of ML functions"""
from typing import Callable, Union

from ..ast import Parameter
from ..backend import get_backend
from ..types import TypeSystem
from .context import symbolic_execution
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


def compile_function(func: Callable, signature: FunctionSignature) -> CompiledFunction:
    """Compile a Python function to MLIR.

    Args:
        func: The decorated Python function
        signature: Parsed function signature with types

    Returns:
        CompiledFunction ready for execution

    Raises:
        RuntimeError: If backend is not available
        TypeError: If return type doesn't match
    """
    backend = get_backend()
    if backend is None:
        raise RuntimeError("C++ backend not available for JIT execution")

    # Already compiled? Return wrapper.
    if backend.has_function(signature.name):
        return CompiledFunction(signature.name, signature, backend)

    # Build AST via symbolic execution
    result_ast = _execute_symbolic(func, signature)

    # Validate return type
    inferred_type = result_ast.infer_type()
    matches, error_msg = TypeSystem.types_match(inferred_type, signature.return_type)
    if not matches:
        raise TypeError(f"Return type mismatch in '{signature.name}':\n{error_msg}")

    # Compile to backend
    backend.compile_function_from_ast(
        signature.name,
        signature.make_param_list(),
        signature.return_type,
        result_ast
    )

    return CompiledFunction(signature.name, signature, backend)


def _execute_symbolic(func: Callable, signature: FunctionSignature):
    """Execute function with symbolic Parameter values to build AST."""
    symbolic_args = [
        Parameter(name, signature.param_types[name])
        for name in signature.param_names
    ]

    with symbolic_execution():
        return func(*symbolic_args)
