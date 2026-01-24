"""Early validation of ml_function decorated functions"""
from typing import Callable, Optional

from ..ast import Parameter, Value
from ..types import TypeSystem
from .context import symbolic_execution
from .signature import FunctionSignature


def validate_function_body(func: Callable, signature: FunctionSignature) -> Optional[Value]:
    """Execute function symbolically to catch type errors at decoration time.

    This runs the function with Parameter objects to trigger type checking
    in operations before any actual compilation happens.

    Args:
        func: The decorated function
        signature: Parsed function signature with types

    Returns:
        The result AST node (for reuse in compilation), or None if validation
        failed for non-type reasons.

    Raises:
        TypeError: If type errors are detected in the function body
    """
    symbolic_args = [
        Parameter(name, signature.param_types[name])
        for name in signature.param_names
    ]

    try:
        with symbolic_execution():
            result_ast = func(*symbolic_args)

        # Validate return type matches
        inferred_type = result_ast.infer_type()
        matches, error_msg = TypeSystem.types_match(inferred_type, signature.return_type)
        if not matches:
            raise TypeError(f"Return type mismatch in '{signature.name}':\n{error_msg}")

        return result_ast

    except TypeError:
        raise
    except Exception:
        # Other errors (e.g., backend issues) caught at execution time
        return None
