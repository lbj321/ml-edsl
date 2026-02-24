"""Test array execution end-to-end (Steps 4-6)

This test validates that array operations execute correctly with the JIT compiler.
"""

import pytest
from mlir_edsl import ml_function, Array, i32, f32


class TestArrayExecution:
    """Test array operations execute correctly with JIT"""

    def test_array_access_execution(self, backend):
        """Test array element access returns correct value"""
        @ml_function
        def array_access() -> i32:
            arr = Array[i32, 4]([10, 20, 30, 40])
            return arr[2]

        result = array_access()
        assert result == 30

    def test_array_store_execution(self, backend):
        """Test array element store and retrieve using .at[] syntax"""
        @ml_function
        def array_store() -> i32:
            arr = Array[i32, 3]([1, 2, 3])
            arr = arr.at[1].set(99)
            return arr[1]

        result = array_store()
        assert result == 99

    def test_array_reuse_execution(self, backend):
        """Test array reuse in computation"""
        @ml_function
        def array_reuse() -> i32:
            arr = Array[i32, 4]([10, 20, 30, 40])
            x = arr[0]
            y = arr[3]
            return x + y

        result = array_reuse()
        assert result == 50  # 10 + 40

    def test_float_array_execution(self, backend):
        """Test float array operations"""
        @ml_function
        def float_array() -> f32:
            arr = Array[f32, 3]([1.5, 2.5, 3.5])
            return arr[1]

        result = float_array()
        assert abs(result - 2.5) < 0.001

    def test_array_with_computed_index(self, backend):
        """Test array access with computed index"""
        @ml_function
        def computed_index() -> i32:
            arr = Array[i32, 5]([10, 20, 30, 40, 50])
            idx = 1 + 1
            return arr[idx]

        result = computed_index()
        assert result == 30  # arr[2]

    def test_array_multiple_stores(self, backend):
        """Test multiple store operations on same array using .at[] syntax"""
        @ml_function
        def multiple_stores() -> i32:
            arr = Array[i32, 4]([1, 2, 3, 4])
            arr = arr.at[0].set(100)
            arr = arr.at[2].set(300)
            x = arr[0]
            y = arr[2]
            return x + y

        result = multiple_stores()
        assert result == 400  # 100 + 300

    def test_direct_assignment_blocked(self):
        """Test that direct assignment raises helpful error message"""
        # Match key parts of the error message
        with pytest.raises(TypeError, match=r"arr = arr\.at\[.*\]\.set"):
            @ml_function
            def bad_store() -> i32:
                arr = Array[i32, 3]([1, 2, 3])
                arr[1] = 99  # Should raise TypeError with helpful message
                return arr[1]
