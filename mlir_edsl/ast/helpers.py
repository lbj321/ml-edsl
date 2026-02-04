"""Helper utilities and JAX-style .at[] array indexing syntax"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import Value


# ==================== VALUE CONVERSION ====================

def to_value(x):
    """Convert Python scalar to Value, or pass through existing Values.

    Args:
        x: Python scalar (int, float, bool) or existing Value

    Returns:
        Value node (Constant for scalars, unchanged for Values)

    Raises:
        TypeError: If x cannot be converted (via Constant)
    """
    from .base import Value
    from .nodes.scalars import Constant
    return x if isinstance(x, Value) else Constant(x)


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
        # Import here to avoid circular dependency
        from .nodes.arrays import ArrayStore
        return ArrayStore(self._array, self._index, value)

    def get(self):
        """Explicit element access: arr.at[i].get()

        This is equivalent to arr[i] but follows the .at[] convention.

        Returns:
            ArrayAccess node for reading the element

        Example:
            value = arr.at[1].get()  # Same as arr[1]
        """
        # Import here to avoid circular dependency
        from .nodes.arrays import ArrayAccess
        return ArrayAccess(self._array, self._index)
