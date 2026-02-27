"""Function signature extraction and type handling"""
from typing import Callable, Dict, List, Any, Tuple, get_type_hints
from dataclasses import dataclass
import inspect

from ..types import Type, TypeSystem, TYPE_HINT_NAMESPACE, Array, Tensor, ArrayType, TensorType


@dataclass
class FunctionSignature:
    """Parsed function signature with MLIR types"""
    name: str
    param_names: List[str]
    param_types: Dict[str, Type]
    return_type: Type

    @classmethod
    def from_callable(cls, func: Callable) -> "FunctionSignature":
        """Extract and validate signature from a decorated function.

        Raises:
            TypeError: If any parameter or return type hint is missing
        """
        sig = inspect.signature(func)
        hints = _get_type_hints(func)

        # Validate all params have hints
        param_names = []
        param_types = {}
        for param_name, param in sig.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                raise TypeError(
                    f"@ml_function '{func.__name__}' parameter '{param_name}' missing type hint"
                )
            param_names.append(param_name)
            param_types[param_name] = TypeSystem.parse_type_hint(
                hints[param_name], f"parameter '{param_name}'"
            )

        # Validate return type
        if sig.return_annotation == inspect.Signature.empty:
            raise TypeError(f"@ml_function '{func.__name__}' missing return type")

        return_type = TypeSystem.parse_type_hint(hints['return'], "return type")

        return cls(
            name=func.__name__,
            param_names=param_names,
            param_types=param_types,
            return_type=return_type
        )

    def order_args(self, args: tuple, kwargs: dict) -> List[Any]:
        """Convert args/kwargs to ordered list matching param_names."""
        combined = {}
        for i, val in enumerate(args):
            if i < len(self.param_names):
                combined[self.param_names[i]] = val
        combined.update(kwargs)
        return [combined[name] for name in self.param_names]

    def validate_runtime_args(self, args: tuple, kwargs: dict):
        """Validate runtime arguments match declared types."""
        combined = {}
        for i, val in enumerate(args):
            if i < len(self.param_names):
                combined[self.param_names[i]] = val
        combined.update(kwargs)

        for name, value in combined.items():
            TypeSystem.validate_value_matches_type(
                value, self.param_types[name], name
            )

    def make_param_list(self) -> List[Tuple[str, Type]]:
        """Return list of (name, type) tuples for backend compilation."""
        return [(name, self.param_types[name]) for name in self.param_names]


def _get_type_hints(func: Callable) -> dict:
    """Get type hints with MLIR type namespace."""
    localns = {
        **TYPE_HINT_NAMESPACE,            # i32, f32, i1
        'Array': Array, 'Tensor': Tensor,
        'ArrayType': ArrayType, 'TensorType': TensorType,
    }
    return get_type_hints(
        func,
        globalns={'int': int, 'float': float, 'bool': bool},
        localns=localns,
    )
