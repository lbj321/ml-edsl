"""Operator overloads for Value nodes (mixin class)"""


class OperatorMixin:
    """Mixin providing operator overloads for Value nodes

    This mixin is inherited by the Value base class to provide
    all arithmetic, comparison, and array indexing operators.
    """

    # Arithmetic operators
    def __add__(self, other):
        """Overload + operator: x + y"""
        from ..ops import add
        return add(self, other)

    def __radd__(self, other):
        """Reverse add: 5 + x"""
        from ..ops import add
        return add(other, self)

    def __sub__(self, other):
        """Overload - operator: x - y"""
        from ..ops import sub
        return sub(self, other)

    def __rsub__(self, other):
        """Reverse sub: 5 - x"""
        from ..ops import sub
        return sub(other, self)

    def __mul__(self, other):
        """Overload * operator: x * y"""
        from ..ops import mul
        return mul(self, other)

    def __rmul__(self, other):
        """Reverse mul: 5 * x"""
        from ..ops import mul
        return mul(other, self)

    def __truediv__(self, other):
        """Overload / operator: x / y"""
        from ..ops import div
        return div(self, other)

    def __rtruediv__(self, other):
        """Reverse div: 5 / x"""
        from ..ops import div
        return div(other, self)

    # Comparison operators
    def __lt__(self, other):
        """Overload < operator: x < y"""
        from ..ops import lt
        return lt(self, other)

    def __le__(self, other):
        """Overload <= operator: x <= y"""
        from ..ops import le
        return le(self, other)

    def __gt__(self, other):
        """Overload > operator: x > y"""
        from ..ops import gt
        return gt(self, other)

    def __ge__(self, other):
        """Overload >= operator: x >= y"""
        from ..ops import ge
        return ge(self, other)

    def __eq__(self, other):
        """Overload == operator: x == y"""
        from ..ops import eq
        return eq(self, other)

    def __ne__(self, other):
        """Overload != operator: x != y"""
        from ..ops import ne
        return ne(self, other)

    # Array subscript operators
    def __getitem__(self, index):
        """Enable arr[i] syntax for array element reads"""
        from .nodes.arrays import ArrayAccess
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
        from .helpers import _AtIndexer
        return _AtIndexer(self)
