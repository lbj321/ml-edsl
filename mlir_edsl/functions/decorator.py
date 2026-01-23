"""Function decorator and compilation utilities"""

from typing import Callable, Union
from ..backend import get_backend
from ..ast import Parameter, CallOp
from ..types import TypeSystem
from .context import symbolic_execution, in_symbolic_context
from .signature import FunctionSignature


class MLFunction:
    """Wrapper for JIT-compiled ML functions"""

    def __init__(self, func: Callable):
        self.func = func
        self.signature = FunctionSignature.from_callable(func)
        self._backend = None  # Store backend reference for IR access
        self._compiled = False  # Track compilation status
        # Do a symbolic dry-run to catch type errors early
        self._validate_function_body()

    @property
    def func_name(self) -> str:
        """Function name (for backward compatibility)."""
        return self.signature.name

    def __call__(self, *args, **kwargs):
        """JIT compile and execute the function - returns numeric result OR AST node"""
        from ..ast import Value

        # Check if we're in symbolic execution context (called from within another @ml_function)
        # This happens during decoration/validation when building the AST
        if in_symbolic_context():
            # Return CallOp AST node instead of executing
            ast_args = list(args) + list(kwargs.values())
            return CallOp(self.signature.name, ast_args, self.signature.return_type)

        # Also check if any argument is an AST node (handles cases where flag isn't set)
        is_symbolic = any(isinstance(arg, Value) for arg in args) or \
                      any(isinstance(v, Value) for v in kwargs.values())

        if is_symbolic:
            # Return CallOp AST node instead of executing
            ast_args = list(args) + list(kwargs.values())
            return CallOp(self.signature.name, ast_args, self.signature.return_type)
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
        if backend.has_function(self.signature.name):
            self._compiled = True
            return

        # Execute symbolically to get AST
        result_ast = self._execute_symbolic()

        # Type validation
        inferred_type = result_ast.infer_type()
        matches, error_msg = TypeSystem.types_match(inferred_type, self.signature.return_type)
        if not matches:
            raise TypeError(f"Return type mismatch in '{self.signature.name}':\n{error_msg}")

        # Compile to backend
        backend.compile_function_from_ast(
            self.signature.name,
            self.signature.make_param_list(),
            self.signature.return_type,
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
        # Validate and order arguments
        self.signature.validate_runtime_args(args, kwargs)
        ordered_args = self.signature.order_args(args, kwargs)

        # Execute
        return self._backend.execute_function(self.signature.name, *ordered_args)
    

    def _execute_symbolic(self):
        """Execute function symbolically to build AST

        Uses symbolic_execution context to ensure nested
        @ml_function calls return CallOp nodes.
        """
        symbolic_args = [
            Parameter(name, self.signature.param_types[name])
            for name in self.signature.param_names
        ]

        with symbolic_execution():
            return self.func(*symbolic_args)


    def _validate_function_body(self):
        """Execute function symbolically to catch type errors at decoration time"""
        dummy_params = [
            Parameter(name, self.signature.param_types[name])
            for name in self.signature.param_names
        ]

        # Execute symbolically - this will trigger type checking in operations
        try:
            result_ast = self.func(*dummy_params)

            # Also validate return type matches
            inferred_type = result_ast.infer_type()
            matches, error_msg = TypeSystem.types_match(inferred_type, self.signature.return_type)
            if not matches:
                raise TypeError(f"Return type mismatch in '{self.signature.name}':\n{error_msg}")

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
                f"Function '{self.signature.name}' not compiled yet. "
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
                f"Function '{self.signature.name}' not compiled yet. "
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