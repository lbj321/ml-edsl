"""C++ MLIR backend integration"""
from typing import Union
from .ast import Value, BinaryOp, Constant, Parameter, CompareOp, IfOp
from .loop_ops import LoopOp

try:
    from . import _mlir_backend
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

# Global instance to keep the builder alive
_global_builder = None

class MLIRValue(Value):
    """Wrapper for C++ MLIR values"""
    
    def __init__(self, cpp_value, builder, value_type: str):
        self.cpp_value = cpp_value
        self.builder = builder
        self.type = value_type
    
    def to_mlir(self, ssa_counter: dict) -> tuple[list[str], str]:
        """This should not be called when using C++ backend"""
        raise RuntimeError("to_mlir() should not be called with C++ backend")

class CppMLIRBuilder:
    """Python wrapper for C++ MLIR builder"""
    
    def __init__(self):
        if not HAS_CPP_BACKEND:
            raise RuntimeError("C++ backend not available. Build with CMake first.")
        
        self.builder = _mlir_backend.MLIRBuilder()
        self.builder.initialize_module()
        self.executor = _mlir_backend.MLIRExecutor()
        self.executor.initialize()
    
    def constant(self, value: Union[int, float]) -> MLIRValue:
        """Create a constant value"""
        cpp_value = self.builder.build_constant(value)
        value_type = "i32" if isinstance(value, int) else "f32"
        return MLIRValue(cpp_value, self, value_type)  # Pass self, not self.builder
    
    def get_parameter(self, name: str, param_type: str) -> MLIRValue:
        cpp_value = self.builder.get_parameter(name)
        return MLIRValue(cpp_value, self, param_type)
    
    def add(self, left: MLIRValue, right: MLIRValue) -> MLIRValue:
        """Create addition operation"""
        cpp_result = self.builder.build_add(left.cpp_value, right.cpp_value)
        result_type = "f32" if left.type == "f32" or right.type == "f32" else "i32"
        return MLIRValue(cpp_result, self, result_type)  # Pass self, not self.builder
    
    def sub(self, left: MLIRValue, right: MLIRValue) -> MLIRValue:
        """Create subtraction operation"""
        cpp_result = self.builder.build_sub(left.cpp_value, right.cpp_value)
        result_type = "f32" if left.type == "f32" or right.type == "f32" else "i32"
        return MLIRValue(cpp_result, self, result_type)  # Pass self, not self.builder
    
    def mul(self, left: MLIRValue, right: MLIRValue) -> MLIRValue:
        """Create multiplication operation"""
        cpp_result = self.builder.build_mul(left.cpp_value, right.cpp_value)
        result_type = "f32" if left.type == "f32" or right.type == "f32" else "i32"
        return MLIRValue(cpp_result, self, result_type)  # Pass self, not self.builder
    
    def div(self, left: MLIRValue, right: MLIRValue) -> MLIRValue:
        """Create division operation"""
        cpp_result = self.builder.build_div(left.cpp_value, right.cpp_value)
        result_type = "f32" if left.type == "f32" or right.type == "f32" else "i32"
        return MLIRValue(cpp_result, self, result_type)  # Pass self, not self.builder
    
    def compare(self, predicate: str, left: MLIRValue, right:MLIRValue) -> MLIRValue:
        cpp_result = self.builder.build_compare(predicate, left.cpp_value, right.cpp_value)
        return MLIRValue(cpp_result, self, "i1")
    
    def if_else(self, condition: MLIRValue, then_value: MLIRValue, else_value: MLIRValue) -> MLIRValue:
        cpp_result = self.builder.build_if(condition.cpp_value, then_value.cpp_value, else_value.cpp_value)
        result_type = then_value.type
        return MLIRValue(cpp_result, self, result_type)
    
    def for_loop(self, start: MLIRValue, end: MLIRValue, step: MLIRValue, 
                 init_value: MLIRValue, operation: LoopOp) -> MLIRValue:
        """Create for loop with predefined operation"""
        cpp_result = self.builder.build_for_with_op(
            start.cpp_value, end.cpp_value, step.cpp_value, 
            init_value.cpp_value, operation.value
        )
        result_type = init_value.type  # Result type matches init_value type
        return MLIRValue(cpp_result, self, result_type)
    
    def while_loop(self, init_value: MLIRValue, target: MLIRValue, 
                   operation: LoopOp, condition: str) -> MLIRValue:
        """Create while loop with predefined operation"""
        cpp_result = self.builder.build_while_with_op(
            init_value.cpp_value, target.cpp_value, 
            operation.value, condition
        )
        result_type = init_value.type  # Result type matches init_value type
        return MLIRValue(cpp_result, self, result_type)
    
    def create_function_with_params_setup(self, param_list: list):
        """Set up function parameters without finalizing"""
        self.builder.create_function_with_params_setup(param_list)
    
    def finalize_function_with_params(self, name: str, result: MLIRValue):
        """Finalize function with parameters"""
        self.builder.finalize_function_with_params(name, result.cpp_value)
    
    def get_parameter(self, name: str, param_type: str = "i32") -> MLIRValue:
        """Get a function parameter by name"""
        cpp_value = self.builder.get_parameter(name)
        return MLIRValue(cpp_value, self, param_type)
    
    def _infer_result_type(self, ast_node: Value) -> str:
        """Infer the result type from AST node without building MLIR"""
        if isinstance(ast_node, Constant):
            return "i32" if isinstance(ast_node.value, int) else "f32"
        elif isinstance(ast_node, Parameter):
            return ast_node.type
        elif isinstance(ast_node, BinaryOp):
            # For binary ops, promote to float if either operand is float
            left_type = self._infer_result_type(ast_node.left)
            right_type = self._infer_result_type(ast_node.right)
            return "f32" if left_type == "f32" or right_type == "f32" else "i32"
        else:
            return "i32"  # default

    def execute_function(self, name: str, result: Union[MLIRValue, Value], 
                        int_args: list = None, float_args: list = None) -> Union[int, float]:
        """Create function, compile with JIT, and execute it"""

        llvm_ir = self.builder.get_llvm_ir_string()
        
        # Clear JIT to avoid symbol collisions
        self.executor.clear()
        
        # Compile and execute
        func_ptr = self.executor.compile_function(llvm_ir, name)
        if func_ptr is None:
            error = self.executor.get_last_error()
            raise RuntimeError(f"JIT compilation failed: {error}")
        
        # Get result type without rebuilding MLIR
        if isinstance(result, MLIRValue):
            result_type = result.type
        else:
            result_type = self._infer_result_type(result)
        
        # Prepare parameter lists (default to empty if None)
        int_args = int_args or []
        float_args = float_args or []
        
        # Call function based on return type
        if result_type == "i32":
            return self.executor.call_int32_function(func_ptr, int_args, float_args)
        elif result_type == "f32":
            return self.executor.call_float_function(func_ptr, int_args, float_args)
        else:
            raise RuntimeError(f"Unsupported return type: {result_type}")
        
        # Note: We don't reset here since the function might be called multiple times
    
    def get_mlir_string(self) -> str:
        """Get MLIR string"""
        return self.builder.get_mlir_string()

    def get_llvm_ir_string(self) -> str:
        """Get LLVM IR string"""
        return self.builder.get_llvm_ir_string()
    
    def reset(self):
        """Reset for a new function"""
        self.builder.reset()
    
    def convert_ast_to_mlir_value(self, ast_node: Value) -> MLIRValue:
        """Convert AST node to MLIRValue for C++ backend"""
        if isinstance(ast_node, Constant):
            return self.constant(ast_node.value)
        elif isinstance(ast_node, Parameter):
            return self.get_parameter(ast_node.name, ast_node.type)
        elif isinstance(ast_node, BinaryOp):
            left_mlir = self.convert_ast_to_mlir_value(ast_node.left)
            right_mlir = self.convert_ast_to_mlir_value(ast_node.right)
            
            if ast_node.op == "add":
                return self.add(left_mlir, right_mlir)
            elif ast_node.op == "sub":
                return self.sub(left_mlir, right_mlir)
            elif ast_node.op == "mul":
                return self.mul(left_mlir, right_mlir)
            elif ast_node.op == "div":
                return self.div(left_mlir, right_mlir)
            else:
                raise ValueError(f"Unsupported operation: {ast_node.op}")
        elif isinstance(ast_node, CompareOp):
            left_mlir = self.convert_ast_to_mlir_value(ast_node.left)
            right_mlir = self.convert_ast_to_mlir_value(ast_node.right)
            return self.compare(ast_node.predicate, left_mlir, right_mlir)
        elif isinstance(ast_node, IfOp):
            cond_mlir = self.convert_ast_to_mlir_value(ast_node.condition)
            then_mlir = self.convert_ast_to_mlir_value(ast_node.then_value)
            else_mlir = self.convert_ast_to_mlir_value(ast_node.else_value)
            return self.if_else(cond_mlir, then_mlir, else_mlir)
        else:
            raise ValueError(f"Unsupported AST node type: {type(ast_node)}")

def get_backend():
    """Get the appropriate backend (C++ if available, fallback to string-based)"""
    global _global_builder
    
    if HAS_CPP_BACKEND:
        if _global_builder is None:
            _global_builder = CppMLIRBuilder()
        return _global_builder
    else:
        return None