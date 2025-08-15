"""Function decorator and compilation utilities"""

from typing import Callable, Any, Union
from .backend import get_backend


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""
    
    def __init__(self, func: Callable):
        self.func = func
        self.func_name = func.__name__
    
    def __call__(self, *args, **kwargs):
        """Regular function call - returns AST result"""
        result = self.func(*args, **kwargs)
        
        # Use C++ backend to generate MLIR for display
        backend = get_backend()
        if backend is None:
            raise RuntimeError("C++ backend not available - required for MLIR generation")
            
        mlir_value = backend.convert_ast_to_mlir_value(result)
        backend.builder.create_function(self.func_name, mlir_value.cpp_value)
        mlir_code = backend.builder.get_mlir_string()
        print(mlir_code)
        backend.reset()
        
        return result
    
    def execute(self) -> Union[int, float]:
        """JIT compile and execute the function"""
        backend = get_backend()
        if backend is None:
            raise RuntimeError("C++ backend not available for JIT execution")
        
        # Execute the function to get the AST result
        result = self.func()
        
        # Use the backend to JIT compile and execute
        return backend.execute_function(self.func_name, result)

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