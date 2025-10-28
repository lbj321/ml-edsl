"""Function decorator and compilation utilities"""

from typing import Callable, Any, Union, Tuple, Dict, get_type_hints
from .backend import get_backend
from .ast import Parameter, Constant, BinaryOp, CallOp, CompareOp, IfOp
from .types import I32, F32, I1, type_to_string
from .type_system import TypeSystem
import inspect


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    def __init__(self, func: Callable):
        self.func = func
        self.func_name = func.__name__
        # Validate type hints at decoration time, not execution time
        self._validate_type_hints()
        # Do a symbolic dry-run to catch type errors early
        self._validate_function_body()

    def __call__(self, *args, **kwargs):
        """JIT compile and execute the function - returns numeric result"""

        backend = get_backend()
        if backend is None:
            raise RuntimeError("C++ backend not available for JIT execution")

        param_map, runtime_values = self._create_parameter_map(args, kwargs)
        self._last_param_map = param_map

        param_list = []
        int_args = []
        float_args = []
        for param_name, param_obj in param_map.items():
            param_list.append((param_name, param_obj.value_type))
            if param_obj.value_type == I32:
                int_args.append(runtime_values[param_name])
            elif param_obj.value_type == F32:
                float_args.append(runtime_values[param_name])

        if not backend.has_function(self.func_name):
            result_ast = self._execute_symbolic(param_map)
            declared_type = self._get_return_type()
            inferred_type = result_ast.infer_type()

            # Validate types match
            matches, error_msg = TypeSystem.types_match(inferred_type, declared_type)
            if not matches:
                raise TypeError(f"Return type mismatch in '{self.func_name}':\n{error_msg}")

            backend.compile_function_from_ast(self.func_name, param_list, declared_type, result_ast)
        else:
            result_ast = self._execute_symbolic(param_map)

        return backend.execute_function(self.func_name, result_ast, int_args, float_args)
    
    def execute(self) -> Union[int, float]:
        """Convenience method - same as calling the function directly"""
        # For zero-parameter functions, just call __call__ with no args
        return self.__call__()

    def _validate_type_hints(self):
        """Ensure all parameters and return type have type hints"""
        signature = inspect.signature(self.func)
        # Check parameters first (they come before return type in function signature)
        for param_name, param in signature.parameters.items():
            if param.annotation == inspect.Parameter.empty:
                raise TypeError(f"@ml_function '{self.func_name}' parameter '{param_name}' missing type hint")
        # Then check return type
        if signature.return_annotation == inspect.Signature.empty:
            raise TypeError(f"@ml_function '{self.func_name}' missing return type")

    def _validate_function_body(self):
        """Execute function symbolically to catch type errors at decoration time"""
        signature = inspect.signature(self.func)
        type_hints = self._get_type_hints()

        # Create dummy Parameter objects with correct types from hints
        dummy_params = []
        for param_name in signature.parameters.keys():
            value_type = TypeSystem.parse_type_hint(
                type_hints[param_name],
                f"parameter '{param_name}'"
            )
            dummy_params.append(Parameter(param_name, value_type))

        # Execute symbolically - this will trigger type checking in operations
        try:
            result_ast = self.func(*dummy_params)

            # Also validate return type matches
            declared_type = self._get_return_type()
            inferred_type = result_ast.infer_type()
            matches, error_msg = TypeSystem.types_match(inferred_type, declared_type)
            if not matches:
                raise TypeError(f"Return type mismatch in '{self.func_name}':\n{error_msg}")

        except TypeError:
            # Re-raise type errors (these are validation failures we want to catch)
            raise
        except Exception:
            # Ignore other errors (e.g., backend issues, runtime problems)
            # Those will be caught at execution time
            pass

    def _get_type_hints(self) -> dict:
        """Get type hints with proper namespace for MLIR types"""
        from .types import i32, f32, i1
        return get_type_hints(
            self.func,
            globalns={'int': int, 'float': float, 'bool': bool},
            localns={'i32': i32, 'f32': f32, 'i1': i1}
        )

    def _create_parameter_map(self, args: Tuple,
                              kwargs: Dict[str, Any]) -> Tuple[Dict[str, Parameter], Dict[str, Any]]:
        """Create parameter map with type hints validation

        Returns:
            Tuple of (parameter_map with type enums, runtime_values dict)
        """
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())
        type_hints = self._get_type_hints()

        all_args = {}
        for i, arg_value in enumerate(args):
            if i < len(param_names):
                all_args[param_names[i]] = arg_value
        all_args.update(kwargs)

        parameter_map = {}
        runtime_values = {}

        for param_name, arg_value in all_args.items():
            value_type = TypeSystem.parse_type_hint(type_hints[param_name], f"parameter '{param_name}'")
            TypeSystem.validate_value_matches_type(arg_value, value_type, param_name)
            parameter_map[param_name] = Parameter(param_name, value_type)
            runtime_values[param_name] = arg_value

        return parameter_map, runtime_values

    def _get_return_type(self) -> int:
        """Get return type enum from type hints"""
        type_hints = self._get_type_hints()
        return TypeSystem.parse_type_hint(type_hints['return'], "return type")
    
    def _execute_symbolic(self, param_map: Dict):
        
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())

        symbolic_args = []
        for param_name in param_names:
            if param_name in param_map:
                symbolic_args.append(param_map[param_name])
            else:
                raise ValueError(f"Missing parameter: {param_name}")
        
        return self.func(*symbolic_args)
    


def ml_function(func: Callable) -> MLFunction:
    """Decorator to mark functions for MLIR compilation
    
    Usage:
        @ml_function
        def my_add():
            return add(5, 3)
        
        # Regular call (prints MLIR)
        my_add()
        
        # JIT execution  
        result = my_add.execute()  # Returns native result
    """
    return MLFunction(func)