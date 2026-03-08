"""Function decorator and compilation utilities"""

import inspect
import os
import textwrap
from typing import Callable, Dict, Tuple, Union
from ..ast import CallOp, Value
from ..types import ArrayType
from .context import in_symbolic_context
from .signature import FunctionSignature
from .validation import validate_function_body
from .compilation import compile_function, CompiledFunction


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    _next_id = 0

    def __init__(self, func: Callable):
        self.func = func
        self._func_id = MLFunction._next_id
        MLFunction._next_id += 1

        self.signature = FunctionSignature.from_callable(func)

        self._compiled: CompiledFunction | None = None

        if self.signature.has_dynamic_dims:
            # Embed func_id so variant names (e.g. my_sum_0__4) are unique
            # across multiple wrappers with the same Python function name
            self.signature.name = f"{func.__name__}_{self._func_id}"
            self._compiled_variants: Dict[Tuple, CompiledFunction] = {}

        # Validate and cache AST for later compilation (also serves as early syntax check for DYN)
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
        if in_symbolic_context():
            ast_args = list(args) + list(kwargs.values())
            return CallOp(self.signature.name, ast_args, self.signature.return_type)

        if self.signature.has_dynamic_dims:
            return self._execute_dynamic(args, kwargs)

        if self._compiled is None:
            self._compiled = compile_function(self.signature, self._cached_ast)
        return self._compiled.execute(args, kwargs)

    def _execute_dynamic(self, args: tuple, kwargs: dict) -> Union[int, float, bool]:
        """Compile and execute a shape-specialized variant for the given input shapes."""
        # Validate ndim and static dims against the DYN signature before specializing
        self.signature.validate_runtime_args(args, kwargs)
        ordered = self.signature.order_args(args, kwargs)

        # Extract concrete shapes for DYN params
        concrete_shapes = {}
        for name, val in zip(self.signature.param_names, ordered):
            t = self.signature.param_types[name]
            if isinstance(t, ArrayType) and t.is_dynamic:
                concrete_shapes[name] = tuple(val.shape)

        shape_key = tuple(
            concrete_shapes[n]
            for n in self.signature.param_names
            if n in concrete_shapes
        )

        if shape_key not in self._compiled_variants:
            specialized_sig = self.signature.specialize(concrete_shapes)
            specialized_ast = validate_function_body(self.func, specialized_sig)
            self._compiled_variants[shape_key] = compile_function(specialized_sig, specialized_ast)

        variant = self._compiled_variants[shape_key]
        variant.signature.validate_runtime_args(args, kwargs)
        ordered = variant.signature.order_args(args, kwargs)
        from ..backend import get_backend
        return get_backend().execute_function(variant.name, *ordered)


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
