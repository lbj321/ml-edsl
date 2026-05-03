"""Function decorator and compilation utilities"""

import inspect
import os
import textwrap
from typing import Callable, Dict, Tuple, Union
from ..ast import CallOp, Value
from ..types import ArrayType, TensorType
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

        self._compiled: CompiledFunction | None = None
        self._cached_ast = None

        if self.signature.has_dynamic_dims:
            # DYN: validate eagerly as a syntax check; each shape re-validates in
            # _execute_dynamic anyway, so _cached_ast is only used for early errors.
            self._cached_ast, _ = validate_function_body(func, self.signature)
            self._compiled_variants: Dict[Tuple, CompiledFunction] = {}
        else:
            # Static: attempt eager validation to surface type errors at decoration
            # time. If the body references the function itself (recursion), the
            # Python name isn't bound yet and raises NameError — catch that case
            # only and defer to the first __call__ instead.
            try:
                self._cached_ast, _ = validate_function_body(func, self.signature)
            except NameError:
                pass  # recursive self-reference; validation deferred to first __call__

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
            if self._cached_ast is None:
                self._cached_ast, _ = validate_function_body(self.func, self.signature)
            self._compiled = compile_function(self.signature, self._cached_ast,
                                              target=self._target)
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
            if isinstance(t, (ArrayType, TensorType)) and t.is_dynamic:
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
            if isinstance(ret, (ArrayType, TensorType)) and ret.is_dynamic:
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
            self._compiled_variants[shape_key] = compile_function(specialized_sig, specialized_ast)
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
        return get_backend().execute_function(variant.name, *ordered)


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
