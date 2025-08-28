"""Abstract Syntax Tree nodes for the EDSL with SSA generation"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ._mlir_backend import MLIRBuilder



class Value(ABC):
    """Base class for all values in the EDSL"""

class Constant(Value):
    """Represents a constant integer value"""
    
    def __init__(self, value: Union[int, float]):
        self.value = value
        self.type = "i32" if isinstance(value, int) else "f32"


class BinaryOp(Value):
    """Represents a binary operation like addition"""
    
    def __init__(self, op: str, left: Value, right: Value):
        self.op = op
        self.left = left
        self.right = right


class Parameter(Value):
    def __init__(self, name: str, value: Union[int, float]):
        self.name = name
        self.value = value
        self.type = "i32" if isinstance(value, int) else "f32"


class CompareOp(Value):
    def __init__(self, predicate: str, left: Value, right: Value):
        self.predicate = predicate
        self.left = left
        self.right = right
        self.type = "i1"


class IfOp(Value):
    def __init__(self, condition: CompareOp, then_value: Value, else_value: Value):
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value
        self.type = then_value.type
