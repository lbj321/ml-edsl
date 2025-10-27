"""Function decorator and compilation utilities"""

from typing import Callable, Any, Union, Tuple, Dict
from .backend import get_backend
from .ast import Parameter, Constant, BinaryOp, CallOp, CompareOp, IfOp
from .types import I32, F32, I1, type_to_string
import inspect


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.func_name = func.__name__
    
    def __call__(self, *args, **kwargs):
        """JIT compile and execute the function - returns numeric result"""

        backend = get_backend()
        if backend is None:
            raise RuntimeError("C++ backend not available for JIT execution")

        # Prepare arguments with type enums
        param_map, runtime_values = self._create_parameter_map(args, kwargs)
        self._last_param_map = param_map

        param_list = []
        int_args = []
        float_args = []
        for param_name, param_obj in param_map.items():
            param_list.append((param_name, param_obj.value_type))
            # Use runtime values for execution
            if param_obj.value_type == I32:
                int_args.append(runtime_values[param_name])
            elif param_obj.value_type == F32:
                float_args.append(runtime_values[param_name])

        # Check if function already compiled
        if not backend.has_function(self.func_name):
            # Compile function (only on first call)
            result_ast = self._execute_symbolic(param_map)

            # Get return type directly from AST
            return_type = result_ast.infer_type()

            # Single call compiles complete function
            backend.compile_function_from_ast(
                self.func_name,
                param_list,
                return_type,
                result_ast
            )
        else:
            # Function already compiled, just need result_ast for execution
            result_ast = self._execute_symbolic(param_map)

        # JIT compile and execute
        return backend.execute_function(self.func_name, result_ast, int_args, float_args)
    
    def execute(self) -> Union[int, float]:
        """Convenience method - same as calling the function directly"""
        # For zero-parameter functions, just call __call__ with no args
        return self.__call__()
    
    def _create_parameter_map(self, args: Tuple,
                              kwargs: Dict[str, Any]) -> Tuple[Dict[str, Parameter], Dict[str, Any]]:
        """Create parameter map with schema-based type enums

        Returns:
            Tuple of (parameter_map with type enums, runtime_values dict)
        """
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())

        parameter_map = {}
        runtime_values = {}

        # Process positional arguments
        for i, arg_value in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                # Infer type enum from Python runtime value
                value_type = I32 if isinstance(arg_value, int) else F32
                parameter_map[param_name] = Parameter(param_name, value_type)
                runtime_values[param_name] = arg_value

        # Process keyword arguments
        for param_name, arg_value in kwargs.items():
            if param_name in param_names:
                value_type = I32 if isinstance(arg_value, int) else F32
                parameter_map[param_name] = Parameter(param_name, value_type)
                runtime_values[param_name] = arg_value

        return parameter_map, runtime_values
    
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