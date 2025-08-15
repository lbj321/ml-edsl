"""C++ MLIR backend integration"""
from typing import Union
from .ast import Value, BinaryOp, Constant

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
    
    def create_function(self, name: str, result: MLIRValue) -> str:
        """Create a function and return MLIR string"""
        self.builder.create_function(name, result.cpp_value)
        mlir_output = self.builder.get_mlir_string()
        self.reset()  # Auto-reset for next function
        return mlir_output
    
    def execute_function(self, name: str, result: Union[MLIRValue, Value]) -> Union[int, float]:
        """Create function, compile with JIT, and execute it"""
        # Convert AST to MLIRValue if needed
        if not isinstance(result, MLIRValue):
            result = self.convert_ast_to_mlir_value(result)
        
        self.builder.create_function(name, result.cpp_value)
        llvm_ir = self.builder.get_llvm_ir_string()
        
        # Compile and execute
        func_ptr = self.executor.compile_function(llvm_ir, name)
        if func_ptr is None:
            error = self.executor.get_last_error()
            raise RuntimeError(f"JIT compilation failed: {error}")
        
        # Call function based on return type
        if result.type == "i32":
            return self.executor.call_int32_function(func_ptr)
        elif result.type == "f32":
            return self.executor.call_float_function(func_ptr)
        else:
            raise RuntimeError(f"Unsupported return type: {result.type}")
        
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
        else:
            raise ValueError(f"Unsupported AST node type: {type(ast_node)}")

def get_backend():
    """Get the appropriate backend (C++ if available, fallback to string-based)"""
    global _global_builder
    
    if HAS_CPP_BACKEND:
        if _global_builder is None:
            _global_builder = CppMLIRBuilder()
        else:
            _global_builder.reset()  # Always reset when retrieved
        return _global_builder
    else:
        return None