"""Abstract Syntax Tree nodes for the EDSL with protobuf serialization"""

from abc import ABC, abstractmethod
from typing import Union, TYPE_CHECKING

# Import generated protobuf code
try:
    from . import ast_pb2
except ImportError:
    # If protobuf hasn't been generated yet, define a placeholder
    ast_pb2 = None

# Import type system
from .types import I32, F32, I1, is_numeric_type, is_integer_type, ArrayType

if TYPE_CHECKING:
    from ._mlir_backend import MLIRBuilder


# ==================== SERIALIZATION CONTEXT ====================
class SerializationContext:
    """Tracks Value reuse during AST serialization for SSA value reuse"""

    def __init__(self):
        self.use_counts = {}     # value.id -> int (how many times referenced)
        self.serialized = set()  # set of value.id that have been serialized

    def count_uses(self, value: 'Value'):
        """Recursively count how many times each Value appears in the tree"""
        if value.id in self.use_counts:
            # Already seen - increment count but don't traverse again
            self.use_counts[value.id] += 1
            return

        # First encounter
        self.use_counts[value.id] = 1

        # Generic traversal - works for ALL node types!
        for child in value.get_children():
            self.count_uses(child)

    def is_reused(self, value: 'Value') -> bool:
        """Check if a value is used more than once"""
        return self.use_counts.get(value.id, 0) > 1

    def mark_serialized(self, value: 'Value'):
        """Mark a value as already serialized"""
        self.serialized.add(value.id)

    def is_serialized(self, value: 'Value') -> bool:
        """Check if a value has already been serialized"""
        return value.id in self.serialized


# ==================== PROTOBUF ENUM MAPPINGS ====================
# These mappings convert string operations/predicates to protobuf enums
# Shared across multiple AST node classes during serialization

def _binary_op_to_proto(operation: str):
    """Convert operation string to protobuf BinaryOpType enum"""
    op_map = {
        "add": ast_pb2.ADD,
        "sub": ast_pb2.SUB,
        "mul": ast_pb2.MUL,
        "div": ast_pb2.DIV,
    }
    return op_map[operation]


def _predicate_to_proto(predicate: str):
    """Convert predicate string to protobuf ComparisonPredicate enum"""
    pred_map = {
        "gt": ast_pb2.GT, "lt": ast_pb2.LT,
        "eq": ast_pb2.EQ, "ne": ast_pb2.NE,
        "ge": ast_pb2.GE, "le": ast_pb2.LE,
        "slt": ast_pb2.SLT, "sle": ast_pb2.SLE,
        "sgt": ast_pb2.SGT, "sge": ast_pb2.SGE,
        "ult": ast_pb2.ULT, "ule": ast_pb2.ULE,
        "ugt": ast_pb2.UGT, "uge": ast_pb2.UGE,
        "olt": ast_pb2.OLT, "ole": ast_pb2.OLE,
        "ogt": ast_pb2.OGT, "oge": ast_pb2.OGE,
        "oeq": ast_pb2.OEQ, "one": ast_pb2.ONE,
        "ueq": ast_pb2.UEQ, "une": ast_pb2.UNE,
    }
    if predicate not in pred_map:
        raise ValueError(f"Unknown predicate: {predicate}")
    return pred_map[predicate]


# ==================== JAX-STYLE .at[] SYNTAX ====================

class _AtIndexer:
    """Helper class for .at[] syntax (JAX-style)

    This enables: arr.at[index].set(value)

    When you write arr.at[index], this class captures the index
    and returns an _AtSetter that provides the .set() method.
    """

    def __init__(self, array: 'Value'):
        self._array = array

    def __getitem__(self, index):
        """Capture the index when user writes arr.at[index]

        Returns:
            _AtSetter object that provides .set() method
        """
        return _AtSetter(self._array, index)


class _AtSetter:
    """Helper class for .at[idx].set() syntax

    This is returned by _AtIndexer.__getitem__ and provides
    the .set() method for functional array updates.
    """

    def __init__(self, array: 'Value', index):
        self._array = array
        self._index = index

    def set(self, value):
        """Functional array update - returns new array with element set

        This creates an ArrayStore node representing the updated array.
        Since MLIR uses SSA (Static Single Assignment), arrays are immutable,
        so this returns a new array rather than modifying the original.

        Args:
            value: The value to store (can be Python literal or Value node)

        Returns:
            ArrayStore node representing the updated array

        Example:
            arr = Array[4, i32]([1, 2, 3, 4])
            arr = arr.at[1].set(99)  # Returns new array [1, 99, 3, 4]
        """
        return ArrayStore(self._array, self._index, value)

    def get(self):
        """Explicit element access: arr.at[i].get()

        This is equivalent to arr[i] but follows the .at[] convention.

        Returns:
            ArrayAccess node for reading the element

        Example:
            value = arr.at[1].get()  # Same as arr[1]
        """
        return ArrayAccess(self._array, self._index)


class Value(ABC):
    """Base class for all values in the EDSL"""
    _next_id = 0

    def __init__(self):
        """Every Value gets a unique ID for reference tracking"""
        self.id = Value._next_id
        Value._next_id += 1

    @abstractmethod
    def infer_type(self) -> Union[int, ArrayType]:
        """Infer the type of this value.

        Returns:
            int: Protobuf ValueType enum (I32=0, F32=1, I1=2) for scalar values
            ArrayType: For array values with size and element type

        Examples:
            Constant(5).infer_type()           # Returns 0 (I32)
            Constant(3.14).infer_type()        # Returns 1 (F32)
            ArrayLiteral([1,2,3], ...).infer_type()  # Returns ArrayType instance
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement infer_type()")

    @abstractmethod
    def to_proto(self, context: 'SerializationContext' = None):
        """Convert this AST node to protobuf Value message

        Args:
            context: Optional serialization context for SSA value reuse

        Raises:
            NotImplementedError: If subclass doesn't implement serialization
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement to_proto()")

    def get_children(self) -> list['Value']:
        """Return list of child Value nodes. Override in subclasses with children."""
        return []  # Default: no children (for leaf nodes like Constant, Parameter)

    def to_proto_with_reuse(self):
        """Serialize with SSA value reuse detection (two-pass approach)"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        # Pass 1: Count how many times each Value is referenced
        context = SerializationContext()
        context.count_uses(self)

        # Pass 2: Serialize with let bindings for reused values
        return self._to_proto_impl(context)

    def _to_proto_impl(self, context: 'SerializationContext'):
        """Serialize this value, emitting let bindings or references as needed"""
        # If this value was already serialized, emit a reference
        if context.is_serialized(self):
            pb_node = ast_pb2.ASTNode()
            pb_node.value_ref.node_id = self.id
            return pb_node

        # If this value will be reused, wrap it in a let binding
        if context.is_reused(self):
            context.mark_serialized(self)
            pb_node = ast_pb2.ASTNode()
            pb_node.let_binding.node_id = self.id
            pb_node.let_binding.value.CopyFrom(self.to_proto(context))
            return pb_node

        # Otherwise, inline it normally
        return self.to_proto(context)

    # Arithmetic operators
    def __add__(self, other):
        """Overload + operator: x + y"""
        from .ops import add
        return add(self, other)

    def __radd__(self, other):
        """Reverse add: 5 + x"""
        from .ops import add
        return add(other, self)

    def __sub__(self, other):
        """Overload - operator: x - y"""
        from .ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        """Reverse sub: 5 - x"""
        from .ops import sub
        return sub(other, self)

    def __mul__(self, other):
        """Overload * operator: x * y"""
        from .ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        """Reverse mul: 5 * x"""
        from .ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        """Overload / operator: x / y"""
        from .ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        """Reverse div: 5 / x"""
        from .ops import div
        return div(other, self)

    # Comparison operators
    def __lt__(self, other):
        """Overload < operator: x < y"""
        from .ops import lt
        return lt(self, other)

    def __le__(self, other):
        """Overload <= operator: x <= y"""
        from .ops import le
        return le(self, other)

    def __gt__(self, other):
        """Overload > operator: x > y"""
        from .ops import gt
        return gt(self, other)

    def __ge__(self, other):
        """Overload >= operator: x >= y"""
        from .ops import ge
        return ge(self, other)

    def __eq__(self, other):
        """Overload == operator: x == y"""
        from .ops import eq
        return eq(self, other)

    def __ne__(self, other):
        """Overload != operator: x != y"""
        from .ops import ne
        return ne(self, other)

    # Array subscript operators
    def __getitem__(self, index):
        """Enable arr[i] syntax for array element reads"""
        return ArrayAccess(self, index)

    def __setitem__(self, index, value):
        """Block direct assignment with helpful error message

        Direct assignment arr[i] = value doesn't work in SSA form
        because Python discards the return value of __setitem__.

        Instead, use the functional .at[] syntax:
            arr = arr.at[i].set(value)
        """
        raise TypeError(
            f"MLIR arrays use SSA (Static Single Assignment) and cannot be mutated in-place.\n"
            f"\n"
            f"❌ Instead of:  arr[{index}] = {value}\n"
            f"✅ Use:         arr = arr.at[{index}].set({value})\n"
            f"\n"
            f"The .at[] syntax returns a new array, which you must assign back.\n"
            f"This makes the SSA semantics explicit and matches JAX's design."
        )

    @property
    def at(self):
        """Enable arr.at[i].set(v) syntax (JAX-style)

        Returns:
            _AtIndexer object that captures the index

        Example:
            arr = Array[4, i32]([10, 20, 30, 40])
            arr = arr.at[1].set(99)       # Returns new array
            arr = arr.at[2].set(88)       # Can chain updates
        """
        return _AtIndexer(self)


class Constant(Value):
    """Represents a constant integer or float value"""

    def __init__(self, value: Union[int, float], value_type=None):
        super().__init__()
        self.value = value
        # Allow explicit type or infer from value
        if value_type is not None:
            self.value_type = value_type
        else:
            self.value_type = I32 if isinstance(value, int) else F32

    def infer_type(self) -> int:
        """Constants know their own type"""
        return self.value_type

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.constant.value_type = self.value_type
        if self.value_type == I32:
            pb_node.constant.int_value = self.value
        else:
            pb_node.constant.float_value = self.value
        return pb_node


class BinaryOp(Value):
    """Represents a binary operation - STRICT TYPE MATCHING ENFORCED"""

    def __init__(self, op: str, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

    def infer_type(self) -> int:
        """STRICT: Both operands must have the same type"""
        left_type = self.left.infer_type()
        right_type = self.right.infer_type()

        if left_type != right_type:
            from .types import type_to_string
            raise TypeError(
                f"Binary operation '{self.op}' requires matching types.\n"
                f"  Left operand type:  {type_to_string(left_type)}\n"
                f"  Right operand type: {type_to_string(right_type)}\n"
                f"  Hint: Use cast() to convert types explicitly"
            )

        return left_type

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.binary_op.op_type = _binary_op_to_proto(self.op)
        pb_node.binary_op.result_type = self.infer_type()

        # Context-aware child serialization
        if context:
            pb_node.binary_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.binary_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            # Backward compatibility: no context = old behavior
            pb_node.binary_op.left.CopyFrom(self.left.to_proto())
            pb_node.binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class Parameter(Value):
    """Represents a named parameter"""

    def __init__(self, name: str, value_type):
        super().__init__()
        self.name = name
        self.value_type = value_type

    def infer_type(self) -> int:
        """Parameters have declared types"""
        return self.value_type

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.parameter.name = self.name
        pb_node.parameter.value_type = self.value_type
        # Value fields are unused by C++ backend (uses parameterMap instead)
        return pb_node


class CompareOp(Value):
    """Represents a comparison operation"""

    def __init__(self, predicate: str, left: Value, right: Value):
        super().__init__()
        self.predicate = predicate
        self.left = left
        self.right = right

        # Validate and infer operand types
        left_type = left.infer_type()
        right_type = right.infer_type()
        if left_type == I1 or right_type == I1:
            raise TypeError("Cannot compare boolean values")

        # Compute promoted operand type (same rule as BinaryOp)
        # F32 + anything = F32, otherwise I32
        # TODO: Fix
        if left_type == F32 or right_type == F32:
            self._operand_type = F32
        else:
            self._operand_type = I32

    def infer_type(self) -> int:
        """Comparisons always return bool"""
        return I1

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.compare_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.compare_op.operand_type = self._operand_type

        # Context-aware child serialization
        if context:
            pb_node.compare_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.compare_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            pb_node.compare_op.left.CopyFrom(self.left.to_proto())
            pb_node.compare_op.right.CopyFrom(self.right.to_proto())

        return pb_node


class IfOp(Value):
    """Represents a conditional if-then-else operation"""

    def __init__(self, condition: Value, then_value: Value, else_value: Value):
        super().__init__()
        self.condition = condition
        self.then_value = then_value
        self.else_value = else_value

        # Type checking at construction
        if condition.infer_type() != I1:
            raise TypeError(f"If condition must be bool (I1), got {condition.infer_type()}")

        then_type = then_value.infer_type()
        else_type = else_value.infer_type()

        if then_type != else_type:
            raise TypeError(
                f"If branches must have same type: then={then_type}, else={else_type}"
            )

        self._inferred_type = then_type

    def infer_type(self) -> int:
        """Return the type of both branches (guaranteed same)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.condition, self.then_value, self.else_value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        if context:
            pb_node.if_op.condition.CopyFrom(self.condition._to_proto_impl(context))
            pb_node.if_op.then_value.CopyFrom(self.then_value._to_proto_impl(context))
            pb_node.if_op.else_value.CopyFrom(self.else_value._to_proto_impl(context))
        else:
            pb_node.if_op.condition.CopyFrom(self.condition.to_proto())
            pb_node.if_op.then_value.CopyFrom(self.then_value.to_proto())
            pb_node.if_op.else_value.CopyFrom(self.else_value.to_proto())

        pb_node.if_op.result_type = self._inferred_type
        return pb_node


class CallOp(Value):
    """Represents a function call operation"""

    def __init__(self, func_name: str, args: list[Value], return_type):
        super().__init__()
        self.func_name = func_name
        self.args = args
        self.return_type = return_type

    def infer_type(self) -> int:
        """Return type is explicitly declared"""
        return self.return_type

    def get_children(self) -> list['Value']:
        return self.args

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.call_op.func_name = self.func_name
        pb_node.call_op.return_type = self.return_type

        # Context-aware child serialization
        if context:
            for arg in self.args:
                pb_node.call_op.args.append(arg._to_proto_impl(context))
        else:
            for arg in self.args:
                pb_node.call_op.args.append(arg.to_proto())

        return pb_node


class ForLoopOp(Value):
    """Represents a for loop operation (scf.for) - STRICT TYPE ENFORCEMENT

    Loop bounds (start, end, step) must all be integers (I32).
    Accumulator (init_value) must all be integers (I32).

    Represents: for(i = start; i < end; i += step) { accumulator = accumulator op i }

    Examples:
        - For(start=0, end=10, step=1, init=0, op="add")      # int loop, int accumulator
    """

    def __init__(self, start: Value, end: Value, step: Value,
                 init_value: Value, operation: str):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.init_value = init_value
        self.operation = operation

        # Get all types
        start_type = start.infer_type()
        end_type = end.infer_type()
        step_type = step.infer_type()
        init_type = init_value.infer_type()

        # STRICT: Loop bounds (start, end, step) must all be the same type
        if not (start_type == end_type == step_type):
            raise TypeError(
                f"ForLoopOp requires loop bounds to have the same type. "
                f"Got: start={start_type}, end={end_type}, step={step_type}"
            )

        # STRICT: Loop bounds must be integers (scf.for requires this)
        if not is_integer_type(start_type):
            raise TypeError(
                f"ForLoopOp requires integer loop bounds (I32). "
                f"Got: {start_type}. Use integer indices with float accumulator if needed."
            )

        # STRICT: Accumulator must be numeric
        if not is_integer_type(init_type):
            raise TypeError(
                f"ForLoopOp accumulator must be numeric (I32). Got: {init_type}"
            )

        # Result type matches the accumulator type
        self._inferred_type = init_type

    def infer_type(self) -> int:
        """Return the loop result type (same as accumulator)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.start, self.end, self.step, self.init_value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        if context:
            pb_node.for_loop_op.start.CopyFrom(self.start._to_proto_impl(context))
            pb_node.for_loop_op.end.CopyFrom(self.end._to_proto_impl(context))
            pb_node.for_loop_op.step.CopyFrom(self.step._to_proto_impl(context))
            pb_node.for_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))
        else:
            pb_node.for_loop_op.start.CopyFrom(self.start.to_proto())
            pb_node.for_loop_op.end.CopyFrom(self.end.to_proto())
            pb_node.for_loop_op.step.CopyFrom(self.step.to_proto())
            pb_node.for_loop_op.init_value.CopyFrom(self.init_value.to_proto())

        pb_node.for_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.for_loop_op.result_type = self._inferred_type

        return pb_node


class WhileLoopOp(Value):
    """Represents a while loop operation (scf.while) - STRICT TYPE ENFORCEMENT

    init_value and target must be the same type.
    Supports I32 or F32 (not I1).

    Represents: while(current predicate target) { current = current op constant }
    """

    def __init__(self, init_value: Value, target: Value,
                 operation: str, predicate: str):
        super().__init__()
        self.init_value = init_value
        self.target = target
        self.operation = operation
        self.predicate = predicate

        # Get types
        init_type = init_value.infer_type()
        target_type = target.infer_type()

        # STRICT: Both must be the same type
        if init_type != target_type:
            raise TypeError(
                f"WhileLoopOp requires init_value and target to have the same type. "
                f"Got: init_value={init_type}, target={target_type}"
            )

        # Only allow I32 or F32
        if init_type not in (I32, F32):
            raise TypeError(
                f"WhileLoopOp only supports I32 or F32, got {init_type}"
            )

        # Result type is trivial - same as inputs
        self._inferred_type = init_type

    def infer_type(self) -> int:
        """Return the loop result type (same as inputs)"""
        return self._inferred_type

    def get_children(self) -> list['Value']:
        return [self.init_value, self.target]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        if context:
            pb_node.while_loop_op.init_value.CopyFrom(self.init_value._to_proto_impl(context))
            pb_node.while_loop_op.target.CopyFrom(self.target._to_proto_impl(context))
        else:
            pb_node.while_loop_op.init_value.CopyFrom(self.init_value.to_proto())
            pb_node.while_loop_op.target.CopyFrom(self.target.to_proto())

        pb_node.while_loop_op.operation = _binary_op_to_proto(self.operation)
        pb_node.while_loop_op.predicate = _predicate_to_proto(self.predicate)
        pb_node.while_loop_op.result_type = self._inferred_type

        return pb_node


class CastOp(Value):
    """Explicit type cast operation"""

    def __init__(self, value: Value, target_type: int):
        """Create a cast operation

        Args:
            value: Value to cast
            target_type: Target MLIR type enum (I32, F32, I1)
        """
        super().__init__()
        self.value = value
        self.target_type = target_type

    def infer_type(self) -> int:
        """Cast always produces the target type"""
        return self.target_type

    def get_children(self) -> list['Value']:
        return [self.value]

    def to_proto(self, context: 'SerializationContext' = None):
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Context-aware child serialization
        if context:
            pb_node.cast_op.value.CopyFrom(self.value._to_proto_impl(context))
        else:
            pb_node.cast_op.value.CopyFrom(self.value.to_proto())

        pb_node.cast_op.target_type = self.target_type
        return pb_node


# ==================== ARRAY OPERATIONS ====================

class ArrayLiteral(Value):
    """
    Array creation: Array[4, i32]([1, 2, 3, 4])

    Compile-time type checking:
    - Validates size matches number of elements
    - Validates all elements match declared element type (strict!)
    """

    def __init__(self, elements: list, array_type: ArrayType):
        super().__init__()
        self.elements = elements
        self.array_type = array_type

        # COMPILE-TIME TYPE CHECKING
        self._validate_size()
        self._validate_element_types()

    def _validate_size(self):
        """Ensure number of elements matches declared size"""
        if len(self.elements) != self.array_type.size:
            raise TypeError(
                f"Array size mismatch: declared Array[{self.array_type.size}, ...] "
                f"but got {len(self.elements)} elements"
            )

    def _validate_element_types(self):
        """Ensure all elements match the declared element type (strict!)"""
        expected_enum = self.array_type.element_enum

        for i, elem in enumerate(self.elements):
            # Convert Python literals to AST nodes if needed
            elem_node = self._ensure_ast_node(elem)
            self.elements[i] = elem_node  # Update with AST node

            # Infer element type
            elem_type = elem_node.infer_type()

            # Strict type checking - must be scalar and match exactly
            if isinstance(elem_type, ArrayType):
                raise TypeError(
                    f"Array element at index {i} cannot be an array. "
                    f"Nested arrays not supported yet."
                )

            if elem_type != expected_enum:
                raise TypeError(
                    f"Array element type mismatch at index {i}: "
                    f"expected {self.array_type.element_type.name}, "
                    f"got {self._enum_to_name(elem_type)}. "
                    f"Use cast() for explicit type conversion."
                )

    def _ensure_ast_node(self, elem):
        """Convert Python literal to AST node if needed"""
        if isinstance(elem, Value):
            return elem

        # Convert Python literals
        if isinstance(elem, bool):
            # Must check bool before int (bool is subclass of int)
            return Constant(elem, I1)
        elif isinstance(elem, int):
            return Constant(elem, I32)
        elif isinstance(elem, float):
            return Constant(elem, F32)
        else:
            raise TypeError(f"Invalid array element: {elem}")

    def _enum_to_name(self, enum_val):
        """Helper: convert enum to readable name"""
        return {I32: "i32", F32: "f32", I1: "i1"}.get(enum_val, f"unknown({enum_val})")

    def infer_type(self) -> Union[int, ArrayType]:
        """ArrayLiteral returns its full ArrayType"""
        return self.array_type  # Returns ArrayType instance, not int!

    def get_children(self) -> list['Value']:
        """Return element nodes for serialization traversal"""
        return self.elements

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayLiteral to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Set array type specification
        pb_node.array_literal.array_type.size = self.array_type.size
        pb_node.array_literal.array_type.element_type = self.array_type.element_enum

        # Serialize each element (with context-aware serialization)
        for elem in self.elements:
            if context:
                pb_node.array_literal.elements.append(elem._to_proto_impl(context))
            else:
                pb_node.array_literal.elements.append(elem.to_proto())

        return pb_node


class ArrayAccess(Value):
    """
    Array element read: arr[index]

    Compile-time type checking:
    - Array must be ArrayType
    - Index must be i32
    - Result type is the array's element type
    """

    def __init__(self, array: Value, index):
        super().__init__()
        self.array = array

        # Convert index to AST node if it's a Python int
        if isinstance(index, int):
            self.index = Constant(index, I32)
        elif isinstance(index, Value):
            self.index = index
        else:
            raise TypeError(f"Array index must be int or Value, got {type(index)}")

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array access is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot index into non-array type. "
                f"Expected array, got {self._type_to_str(array_type)}"
            )

        # Check that index is i32
        index_type = self.index.infer_type()
        if isinstance(index_type, ArrayType) or index_type != I32:
            raise TypeError(
                f"Array index must be i32, got {self._type_to_str(index_type)}. "
                f"Use cast() to convert to i32."
            )

        # Store the array type for infer_type()
        self._array_type = array_type

    def _type_to_str(self, typ):
        """Helper: convert type to readable string"""
        if isinstance(typ, ArrayType):
            return repr(typ)
        return {I32: "i32", F32: "f32", I1: "i1"}.get(typ, f"unknown({typ})")

    def infer_type(self) -> Union[int, ArrayType]:
        """Array access returns the element type (scalar enum)"""
        # If we index into Array[10, i32], we get back i32
        return self._array_type.element_enum  # Returns I32/F32/I1 (int)

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array, self.index]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayAccess to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Serialize array and index (context-aware)
        if context:
            pb_node.array_access.array.CopyFrom(self.array._to_proto_impl(context))
            pb_node.array_access.index.CopyFrom(self.index._to_proto_impl(context))
        else:
            pb_node.array_access.array.CopyFrom(self.array.to_proto())
            pb_node.array_access.index.CopyFrom(self.index.to_proto())

        return pb_node


class ArrayStore(Value):
    """
    Array element write: arr[index] = value

    Compile-time type checking:
    - Array must be ArrayType
    - Index must be i32
    - Value type must match array element type exactly
    """

    def __init__(self, array: Value, index, value):
        super().__init__()
        self.array = array

        # Convert index to AST node
        if isinstance(index, int):
            self.index = Constant(index, I32)
        elif isinstance(index, Value):
            self.index = index
        else:
            raise TypeError(f"Array index must be int or Value")

        # Convert value to AST node if needed
        if isinstance(value, bool):
            # Must check bool before int (bool is subclass of int)
            self.value = Constant(value, I1)
        elif isinstance(value, int):
            self.value = Constant(value, I32)
        elif isinstance(value, float):
            self.value = Constant(value, F32)
        elif isinstance(value, Value):
            self.value = value
        else:
            raise TypeError(f"Array value must be int/float/bool or Value")

        # COMPILE-TIME TYPE CHECKING
        self._validate_types()

    def _validate_types(self):
        """Validate array store is type-safe"""
        # Check that we're indexing an array
        array_type = self.array.infer_type()
        if not isinstance(array_type, ArrayType):
            raise TypeError(
                f"Cannot use []= on non-array type: {self._type_to_str(array_type)}"
            )

        # Check index is i32
        index_type = self.index.infer_type()
        if isinstance(index_type, ArrayType) or index_type != I32:
            raise TypeError(f"Array index must be i32, got {self._type_to_str(index_type)}")

        # Check value type matches array element type (STRICT!)
        expected_enum = array_type.element_enum
        actual_type = self.value.infer_type()

        if isinstance(actual_type, ArrayType):
            raise TypeError(
                f"Cannot store array into array element. "
                f"Expected {array_type.element_type.name}, got {actual_type}"
            )

        if actual_type != expected_enum:
            raise TypeError(
                f"Cannot store {self._type_to_str(actual_type)} into "
                f"Array[..., {array_type.element_type.name}]. "
                f"Use cast() for explicit conversion."
            )

        # Store array type for later
        self._array_type = array_type

    def _type_to_str(self, typ):
        """Helper: convert type to readable string"""
        if isinstance(typ, ArrayType):
            return repr(typ)
        return {I32: "i32", F32: "f32", I1: "i1"}.get(typ, f"unknown({typ})")

    def infer_type(self) -> Union[int, ArrayType]:
        """Store doesn't produce a value, but return array type for consistency"""
        return self._array_type

    def get_children(self) -> list['Value']:
        """Return child nodes"""
        return [self.array, self.index, self.value]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize ArrayStore to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()

        # Serialize array, index, and value (context-aware)
        if context:
            pb_node.array_store.array.CopyFrom(self.array._to_proto_impl(context))
            pb_node.array_store.index.CopyFrom(self.index._to_proto_impl(context))
            pb_node.array_store.value.CopyFrom(self.value._to_proto_impl(context))
        else:
            pb_node.array_store.array.CopyFrom(self.array.to_proto())
            pb_node.array_store.index.CopyFrom(self.index.to_proto())
            pb_node.array_store.value.CopyFrom(self.value.to_proto())

        return pb_node


class ArrayBinaryOp(Value):
    """Element-wise binary operation on arrays with broadcasting support

    Supports three modes:
    - Array + Array: Element-wise with matching shapes
    - Array + Scalar: Broadcasting scalar to all elements
    - Scalar + Array: Broadcasting scalar to all elements

    Examples:
        arr1 + arr2          # Array[4,i32] + Array[4,i32] -> Array[4,i32]
        arr * 2              # Array[4,i32] * Constant(2) -> Array[4,i32]
        3.0 + arr            # Constant(3.0) + Array[4,f32] -> Array[4,f32]
    """

    def __init__(self, op: str, left: Value, right: Value):
        super().__init__()
        self.op = op
        self.left = left
        self.right = right

        # Type inference and validation
        left_type = left.infer_type()
        right_type = right.infer_type()

        # Determine operation mode and result type
        self._infer_broadcast_mode(left_type, right_type)

    def _infer_broadcast_mode(self, left_type, right_type):
        """Determine broadcasting mode and validate types"""
        left_is_array = isinstance(left_type, ArrayType)
        right_is_array = isinstance(right_type, ArrayType)

        if left_is_array and right_is_array:
            # ARRAY + ARRAY: Shapes must match exactly
            if left_type != right_type:
                raise TypeError(
                    f"Array shapes must match for element-wise {self.op}.\n"
                    f"  Left:  {left_type}\n"
                    f"  Right: {right_type}"
                )
            self._result_type = left_type
            self._broadcast_mode = "NONE"

        elif left_is_array and not right_is_array:
            # ARRAY + SCALAR: Validate scalar type matches array element type
            if right_type != left_type.element_enum:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Array element type: {left_type.element_type.name}\n"
                    f"  Scalar type: {self._enum_to_name(right_type)}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = left_type
            self._broadcast_mode = "SCALAR_RIGHT"

        elif not left_is_array and right_is_array:
            # SCALAR + ARRAY: Validate scalar type matches array element type
            if left_type != right_type.element_enum:
                raise TypeError(
                    f"Scalar type must match array element type.\n"
                    f"  Scalar type: {self._enum_to_name(left_type)}\n"
                    f"  Array element type: {right_type.element_type.name}\n"
                    f"  Use cast() for explicit conversion"
                )
            self._result_type = right_type
            self._broadcast_mode = "SCALAR_LEFT"

        else:
            # SCALAR + SCALAR: This should use BinaryOp, not ArrayBinaryOp
            raise TypeError(
                f"ArrayBinaryOp requires at least one array operand.\n"
                f"For scalar operations, use BinaryOp instead."
            )

    def _enum_to_name(self, enum_val):
        """Helper: convert enum to readable name"""
        return {I32: "i32", F32: "f32", I1: "i1"}.get(enum_val, f"unknown({enum_val})")

    def infer_type(self) -> ArrayType:
        """Element-wise operations preserve array type"""
        return self._result_type

    def get_children(self) -> list['Value']:
        return [self.left, self.right]

    def to_proto(self, context: 'SerializationContext' = None):
        """Serialize to protobuf"""
        if ast_pb2 is None:
            raise RuntimeError("Protobuf code not generated. Run ./build.sh first.")

        pb_node = ast_pb2.ASTNode()
        pb_node.array_binary_op.op_type = _binary_op_to_proto(self.op)

        # Set result type (array shape)
        pb_node.array_binary_op.result_type.size = self._result_type.size
        pb_node.array_binary_op.result_type.element_type = self._result_type.element_enum

        # Set broadcast mode
        broadcast_map = {
            "NONE": ast_pb2.NONE,
            "SCALAR_LEFT": ast_pb2.SCALAR_LEFT,
            "SCALAR_RIGHT": ast_pb2.SCALAR_RIGHT,
        }
        pb_node.array_binary_op.broadcast = broadcast_map[self._broadcast_mode]

        # Context-aware child serialization
        if context:
            pb_node.array_binary_op.left.CopyFrom(self.left._to_proto_impl(context))
            pb_node.array_binary_op.right.CopyFrom(self.right._to_proto_impl(context))
        else:
            pb_node.array_binary_op.left.CopyFrom(self.left.to_proto())
            pb_node.array_binary_op.right.CopyFrom(self.right.to_proto())

        return pb_node
