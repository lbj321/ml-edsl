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

        # Prepare arguments
        param_map = self._create_parameter_map(args, kwargs)
        self._last_param_map = param_map

        param_list = []
        int_args = []
        float_args = []
        for param_name, param_obj in param_map.items():
            param_list.append((param_name, param_obj.type))
            if param_obj.type == "i32":
                int_args.append(param_obj.value)
            elif param_obj.type == "f32":
                float_args.append(param_obj.value)

        # Check if function already compiled
        if not backend.has_function(self.func_name):
            # Compile function (only on first call)
            result_ast = self._execute_symbolic(param_map)
            return_type = self._infer_return_type(result_ast)

            # Single call compiles complete function (declaration + body + return)
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
    
    def _infer_return_type(self, result_ast) -> str:
        """Recursively infer type from AST tree"""
        
        if isinstance(result_ast, Constant):
            return "i32" if isinstance(result_ast.value, int) else "f32"
        
        elif isinstance(result_ast, Parameter):
            return result_ast.type
        
        elif isinstance(result_ast, BinaryOp):
            # Get types of both operands
            left_type = self._infer_return_type(result_ast.left)
            right_type = self._infer_return_type(result_ast.right)
            # Apply promotion rules (matches your C++ backend)
            return "f32" if left_type == "f32" or right_type == "f32" else "i32"
        
        elif isinstance(result_ast, CallOp):
            # Function calls have explicit return types
            return result_ast.type  # "i32" or "f32"
        
        elif isinstance(result_ast, CompareOp):
            return "i1"  # Boolean result
        
        elif isinstance(result_ast, IfOp):
            # Return type matches then/else branches
            return self._infer_return_type(result_ast.then_value)
        
        else:
            raise ValueError(f"Cannot infer type for AST node: {type(result_ast).__name__}")


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