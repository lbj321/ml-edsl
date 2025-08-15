"""Abstract Syntax Tree nodes for the EDSL with SSA generation"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING
from ._mlir_backend import MLIRBuilder



class Value(ABC):
    """Base class for all values in the EDSL"""
    
    @abstractmethod
    def to_mlir(self, builder: 'MLIRBuilder'):
        """Convert this value to MLIR using C++ backend
        
        Args:
            builder: MLIRBuilder instance for generating MLIR
            
        Returns:
            mlir::Value object from the C++ backend
        """
        raise NotImplementedError("to_mlir() not implemented")


class Constant(Value):
    """Represents a constant integer value"""
    
    def __init__(self, value: Union[int, float]):
        self.value = value
        self.type = "i32" if isinstance(value, int) else "f32"
    
    def to_mlir(self, builder: 'MLIRBuilder'):
        """Generate SSA: '%N = arith.constant {value} : i32'"""
        
        if isinstance(self.value, int):
            return builder.build_constant(self.value)
        else:
            return builder.build_constant(float(self.value))

class BinaryOp(Value):
    """Represents a binary operation like addition"""
    
    def __init__(self, op: str, left: Value, right: Value):
        self.op = op
        self.left = left
        self.right = right
    
    def to_mlir(self, builder: 'MLIRBuilder'):
        """Generate SSA for binary operation"""

        lhs_val = self.left.to_mlir(builder)
        rhs_val = self.right.to_mlir(builder)

        if self.op == "add":
            return builder.build_add(lhs_val, rhs_val)
        elif self.op == "sub":
            return builder.build_sub(lhs_val, rhs_val)
        elif self.op == "mul":
            return builder.build_mul(lhs_val, rhs_val)
        elif self.op == "div":
            return builder.build_div(lhs_val, rhs_val)
        else:
            raise NotImplementedError(f"Operation '{self.op}' not valid")
