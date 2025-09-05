"""Control flow and loop operation constants for MLIR EDSL"""

from enum import Enum


class LoopOp(Enum):
    """Supported operations for loop bodies"""
    ADD = "add"
    MUL = "mul" 
    SUB = "sub"
    DIV = "div"


class ConditionalOp(Enum):
    """Supported conditional operations"""
    IF = "if"
    ELIF = "elif"
    ELSE = "else"


class ComparisonOp(Enum):
    """Supported comparison operations"""
    GT = "gt"      # Greater than
    LT = "lt"      # Less than
    EQ = "eq"      # Equal
    NE = "ne"      # Not equal
    GE = "ge"      # Greater than or equal
    LE = "le"      # Less than or equal