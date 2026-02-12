"""Function decorator and compilation utilities"""

import inspect
import os
import textwrap
from typing import Callable, Union
from ..ast import CallOp, Value
from .context import in_symbolic_context
from .signature import FunctionSignature
from .validation import validate_function_body
from .compilation import compile_function, CompiledFunction


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    def __init__(self, func: Callable):
        self.func = func
        self.signature = FunctionSignature.from_callable(func)
        self._compiled: CompiledFunction | None = None
        # Validate and cache AST for later compilation
        self._cached_ast: Value = validate_function_body(func, self.signature)

        # Store Python source on backend for HTML report (SAVE_IR=1)
        if os.getenv("SAVE_IR"):
            from ..backend import get_backend
            b = get_backend()
            if b is not None:
                try:
                    src = textwrap.dedent(inspect.getsource(func))
                except OSError:
                    src = f"# Could not retrieve source for {func.__name__}"
                b._func_sources[self.signature.name] = src

    def __call__(self, *args, **kwargs) -> Union[int, float, bool, Value]:
        """JIT compile and execute the function - returns numeric result OR AST node"""
        # Compile on first call using cached AST
        if self._compiled is None:
            self._compiled = compile_function(self.signature, self._cached_ast)

        if in_symbolic_context():
            ast_args = list(args) + list(kwargs.values())
            return CallOp(self.signature.name, ast_args, self.signature.return_type)

        return self._compiled.execute(args, kwargs)


def ml_function(func: Callable) -> MLFunction:
    """Decorator to mark functions for MLIR compilation

    Usage:
        @ml_function
        def my_add():
            return add(5, 3)

        # JIT execution
        result = my_add()  # Returns native result
    """
    return MLFunction(func)
