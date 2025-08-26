"""Function decorator and compilation utilities"""

from typing import Callable, Any, Union, Tuple, Dict
from .backend import get_backend
from .ast import Parameter
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

        # Handle parameterized vs zero-parameter cases
        if args or kwargs:
            # Parameterized function
            param_map = self._create_parameter_map(args, kwargs)
            self._last_param_map = param_map
            result = self._execute_symbolic(param_map)
            
            param_list = []
            int_args = []
            float_args = []
            for param_name, param_obj in param_map.items():
                param_list.append((param_name, param_obj.type))
                if param_obj.type == "i32":
                    int_args.append(param_obj.value)
                elif param_obj.type == "f32":
                    float_args.append(param_obj.value)

            backend.reset()
            backend.builder.create_function_with_params_setup(param_list)
            mlir_result = backend.convert_ast_to_mlir_value(result)
            backend.builder.finalize_function_with_params(self.func_name, mlir_result.cpp_value)
        else:
            # Zero-parameter function
            result = self.func()
            int_args = []
            float_args = []
            
            backend.reset()
            backend.builder.create_function_with_params_setup([])  # Empty params
            mlir_result = backend.convert_ast_to_mlir_value(result)
            backend.builder.finalize_function_with_params(self.func_name, mlir_result.cpp_value)

        # JIT compile and execute
        return backend.execute_function(self.func_name, result, int_args, float_args)
    
    def execute(self) -> Union[int, float]:
        """Convenience method - same as calling the function directly"""
        # For zero-parameter functions, just call __call__ with no args
        return self.__call__()
    
    def _create_parameter_map(self, args: Tuple,
                              kwargs: Dict[str, Any]) -> Dict[str, Parameter]:
        
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())

        parameter_map = {}

        for i, arg_value in enumerate(args):
            if i < len(param_names):
                param_name = param_names[i]
                parameter_map[param_name] = Parameter(param_name, arg_value)

        for param_name, arg_value in kwargs.items():
            if param_name in param_names:
                parameter_map[param_name] = Parameter(param_name, arg_value)

        return parameter_map
    
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