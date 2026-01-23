"""Function decorator and compilation utilities"""

from typing import Callable, Any, Union, Tuple, Dict, get_type_hints
from ..backend import get_backend
from ..ast import Parameter, Constant, BinaryOp, CallOp, CompareOp, IfOp
from ..types import Type, ScalarType, ArrayType, TypeSystem, i32, f32, i1
from .context import symbolic_execution, in_symbolic_context
import inspect


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    def __init__(self, func: Callable):
        self.func = func
        self.func_name = func.__name__
        self._backend = None  # Store backend reference for IR access
        self._compiled = False  # Track compilation status
        # Validate type hints at decoration time, not execution time
        self._validate_type_hints()
        # Do a symbolic dry-run to catch type errors early
        self._validate_function_body()

    def __call__(self, *args, **kwargs):
        """JIT compile and execute the function - returns numeric result OR AST node"""
        from ..ast import Value

        # Check if we're in symbolic execution context (called from within another @ml_function)
        # This happens during decoration/validation when building the AST
        if in_symbolic_context():
            # Return CallOp AST node instead of executing
            ast_args = list(args) + list(kwargs.values())
            return_type = self._get_return_type()
            return CallOp(self.func_name, ast_args, return_type)

        # Also check if any argument is an AST node (handles cases where flag isn't set)
        is_symbolic = any(isinstance(arg, Value) for arg in args) or \
                      any(isinstance(v, Value) for v in kwargs.values())

        if is_symbolic:
            # Return CallOp AST node instead of executing
            ast_args = list(args) + list(kwargs.values())
            return_type = self._get_return_type()
            return CallOp(self.func_name, ast_args, return_type)
        else:
            # Normal execution path (called from Python with runtime values)
            self._ensure_compiled(args, kwargs)
            return self._execute(args, kwargs)
    
    def execute(self) -> Union[int, float]:
        """Convenience method - same as calling the function directly"""
        # For zero-parameter functions, just call __call__ with no args
        return self.__call__()

    # ==================== COMPILATION (Internal) ====================

    def _ensure_compiled(self, args: tuple, kwargs: dict):
        """Ensure function is compiled, compiling if necessary

        This method is idempotent - safe to call multiple times.
        Only compiles once, subsequent calls are no-ops.
        """
        backend = get_backend()
        if backend is None:
            raise RuntimeError("C++ backend not available for JIT execution")

        self._backend = backend

        # Already compiled? Skip.
        if backend.has_function(self.func_name):
            self._compiled = True
            return

        # Perform compilation
        param_map, runtime_values = self._create_parameter_map(args, kwargs)
        self._last_param_map = param_map

        # Build parameter list
        param_list = []
        for param_name, param_obj in param_map.items():
            param_list.append((param_name, param_obj.value_type))

        # Execute symbolically to get AST
        result_ast = self._execute_symbolic(param_map)

        # Type validation
        declared_type = self._get_return_type()
        inferred_type = result_ast.infer_type()
        matches, error_msg = TypeSystem.types_match(inferred_type, declared_type)
        if not matches:
            raise TypeError(f"Return type mismatch in '{self.func_name}':\n{error_msg}")

        # Compile to backend
        backend.compile_function_from_ast(
            self.func_name,
            param_list,
            declared_type,
            result_ast
        )

        self._compiled = True

    # ==================== EXECUTION (Internal) ====================

    def _execute(self, args: tuple, kwargs: dict) -> Union[int, float, bool]:
        """Execute the compiled function

        Precondition: Function must be compiled (call _ensure_compiled first)

        Args:
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Execution result (int, float, or bool)
        """
        # Convert args/kwargs to ordered argument list
        param_map, runtime_values = self._create_parameter_map(args, kwargs)
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())
        ordered_args = [runtime_values[name] for name in param_names]

        # Execute
        return self._backend.execute_function(self.func_name, *ordered_args)
    

    
    # ==================== HELPER METHODS ====================

    def _get_type_hints(self) -> dict:
        """Get type hints with proper namespace for MLIR types"""
        from ..types import i32, f32, i1
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

    def _get_return_type(self) -> Type:
        """Get return type from type hints"""
        type_hints = self._get_type_hints()
        return TypeSystem.parse_type_hint(type_hints['return'], "return type")
    
    def _execute_symbolic(self, param_map: Dict):
        """Execute function symbolically to build AST

        Uses symbolic_execution context to ensure nested
        @ml_function calls return CallOp nodes.
        """
        signature = inspect.signature(self.func)
        param_names = list(signature.parameters.keys())

        symbolic_args = []
        for param_name in param_names:
            if param_name in param_map:
                symbolic_args.append(param_map[param_name])
            else:
                raise ValueError(f"Missing parameter: {param_name}")

        with symbolic_execution():
            return self.func(*symbolic_args)


    # ==================== VALIDATION ====================

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
        
    # ==================== IR INSPECTION ====================

    def get_mlir_string(self) -> str:
        """Get MLIR IR for this function

        Returns:
            String containing the generated MLIR IR

        Raises:
            RuntimeError: If function hasn't been compiled yet
        """
        if not self._compiled or self._backend is None:
            raise RuntimeError(
                f"Function '{self.func_name}' not compiled yet. "
                f"Call the function first to trigger compilation."
            )
        return self._backend.get_mlir_string()

    def get_llvm_ir_string(self) -> str:
        """Get LLVM IR for this function

        Returns:
            String containing the generated LLVM IR

        Raises:
            RuntimeError: If function hasn't been compiled yet
        """
        if not self._compiled or self._backend is None:
            raise RuntimeError(
                f"Function '{self.func_name}' not compiled yet. "
                f"Call the function first to trigger compilation."
            )
        return self._backend.get_llvm_ir_string()

    def print_ir(self, title: str = None):
        """Print both MLIR and LLVM IR

        Args:
            title: Optional title to print before IR
        """
        if title:
            print(f"\n{'='*60}")
            print(title)

        print(f"\n{'='*60}")
        print("MLIR:")
        print('='*60)
        print(self.get_mlir_string())

        print(f"\n{'='*60}")
        print("LLVM IR:")
        print('='*60)
        print(self.get_llvm_ir_string())


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