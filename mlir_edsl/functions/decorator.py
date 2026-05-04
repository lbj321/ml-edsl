"""Function decorator and compilation utilities"""

import inspect
import os
import textwrap
from typing import Callable, Dict, Tuple, Union
from ..ast import CallOp, Value
from .context import in_symbolic_context
from .signature import FunctionSignature
from .validation import validate_function_body
from .compilation import compile_function, CompiledFunction


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    _next_id = 0

    def __init__(self, func: Callable, target: str = "cpu"):
        self.func = func
        self._target = target
        self._func_id = MLFunction._next_id
        MLFunction._next_id += 1

        self.signature = FunctionSignature.from_callable(func)
        # Always embed func_id so two wrappers with the same Python function
        # name (e.g. both named "fn" but with different types) never collide in
        # the backend's has_function cache.
        self.signature.name = f"{func.__name__}_{self._func_id}"

        # All functions use the same variant cache. For static shapes the key
        # is always () so they compile once; for DYN shapes a new variant is
        # compiled per distinct shape tuple. No eager validation here — errors
        # surface at first __call__ (the JAX trade-off), which also lets
        # recursive functions reference themselves without a NameError.
        self._compiled_variants: Dict[Tuple, CompiledFunction] = {}

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
        return self._execute(args, kwargs)

    def _execute(self, args: tuple, kwargs: dict) -> Union[int, float, bool]:
        """Compile and execute a shape-specialised variant for the given inputs.

        For static shapes the key is always () — compiles once.
        For DYN shapes a new variant is compiled per distinct shape tuple.
        """
        self.signature.validate_runtime_args(args, kwargs)
        ordered = self.signature.order_args(args, kwargs)

        # Build concrete_shapes only for DYN params; static params already
        # have known shapes baked into their declared type.
        concrete_shapes = {}
        for name, val in zip(self.signature.param_names, ordered):
            t = self.signature.param_types[name]
            if t.is_aggregate() and t.is_dynamic:
                concrete_shapes[name] = tuple(val.shape)

        shape_key = tuple(
            concrete_shapes[n]
            for n in self.signature.param_names
            if n in concrete_shapes
        )

        if shape_key not in self._compiled_variants:
            specialized_sig = self.signature.specialize(concrete_shapes)
            specialized_ast, inferred_return = validate_function_body(self.func, specialized_sig)
            # Replace DYN return type with the concrete shape inferred by abstract evaluation
            ret = specialized_sig.return_type
            if ret.is_aggregate() and ret.is_dynamic:
                if inferred_return.is_dynamic:
                    raise TypeError(
                        f"Cannot determine return shape for DYN return type in '{self.func.__name__}': "
                        "abstract evaluation produced dynamic shape. "
                        "Output shape must be deterministic from input shapes."
                    )
                specialized_sig = FunctionSignature(
                    name=specialized_sig.name,
                    param_names=specialized_sig.param_names,
                    param_types=specialized_sig.param_types,
                    return_type=inferred_return,
                )
            self._compiled_variants[shape_key] = compile_function(
                specialized_sig, specialized_ast, target=self._target
            )
            if os.getenv("SAVE_IR"):
                from ..backend import get_backend
                b = get_backend()
                if b is not None:
                    src = b._func_sources.get(self.signature.name, "")
                    b._func_sources[specialized_sig.name] = src

        variant = self._compiled_variants[shape_key]
        variant.signature.validate_runtime_args(args, kwargs)
        ordered = variant.signature.order_args(args, kwargs)
        from ..backend import get_backend
        backend = get_backend()
        if self._target == "gpu":
            return backend.execute_gpu_function(variant.name, *ordered)
        return backend.execute_function(variant.name, *ordered)


def ml_function(func: Callable = None, *, target: str = "cpu"):
    """Decorator to mark functions for MLIR compilation.

    Usage:
        @ml_function
        def my_add(): ...

        @ml_function(target="gpu")
        def my_matmul(A: Tensor[f32, 32, 32], B: Tensor[f32, 32, 32]): ...
    """
    if func is not None:
        # Called as @ml_function (no parentheses)
        return MLFunction(func, target="cpu")
    # Called as @ml_function(target=...) — return decorator
    def decorator(f: Callable) -> MLFunction:
        return MLFunction(f, target=target)
    return decorator
