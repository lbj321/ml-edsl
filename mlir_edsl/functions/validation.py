"""Early validation of ml_function decorated functions"""
from typing import Callable

from ..ast import Parameter
from ..types import TypeSystem
from .signature import FunctionSignature


def validate_function_body(func: Callable, signature: FunctionSignature):
    """Execute function symbolically to catch type errors at decoration time.

    This runs the function with dummy Parameter objects to trigger type
    checking in operations before any actual compilation happens.

    Args:
        func: The decorated function
        signature: Parsed function signature with types

    Raises:
        TypeError: If type errors are detected in the function body
    """
    dummy_params = [
        Parameter(name, signature.param_types[name])
        for name in signature.param_names
    ]

    # Execute symbolically - this will trigger type checking in operations
    try:
        result_ast = func(*dummy_params)

        # Also validate return type matches
        inferred_type = result_ast.infer_type()
        matches, error_msg = TypeSystem.types_match(inferred_type, signature.return_type)
        if not matches:
            raise TypeError(f"Return type mismatch in '{signature.name}':\n{error_msg}")

    except TypeError:
        # Re-raise type errors (these are validation failures we want to catch)
        raise
    except Exception:
        # Ignore other errors (e.g., backend issues, runtime problems)
        # Those will be caught at execution time
        pass
