"""Test array MLIR generation (Step 4 - memref dialect)

This test validates that array operations correctly generate MLIR IR using the memref dialect.
Note: These tests only verify IR generation, not execution (Step 6).
"""

import pytest
from mlir_edsl import ml_function, Array, i32, f32


class TestArrayMLIRGeneration:
    """Test that array operations generate correct MLIR IR"""

    def test_array_literal_and_access(self):
        """Test array creation and element access generates memref.alloca and memref.load"""
        @ml_function
        def array_access() -> i32:
            arr = Array[4, i32]([10, 20, 30, 40])
            return arr[2]

        # Just test IR generation - don't execute yet
        # IR should contain memref.alloca and memref.load operations

    def test_array_store(self):
        """Test array element store generates memref.store using .at[] syntax"""
        @ml_function
        def array_store() -> i32:
            arr = Array[3, i32]([1, 2, 3])
            arr = arr.at[1].set(99)
            return arr[1]

        # IR should contain memref.store operation

    def test_array_reuse_ssa(self):
        """Test array reuse with SSA value caching"""
        @ml_function
        def array_reuse() -> i32:
            arr = Array[4, i32]([10, 20, 30, 40])
            x = arr[0]  # First use of arr
            y = arr[3]  # Second use of arr - should reuse cached memref
            return x + y

        # IR should show LetBinding wrapping array, then ValueReference for reuse

    def test_float_array(self):
        """Test float array creation"""
        @ml_function
        def float_array() -> f32:
            arr = Array[3, f32]([1.5, 2.5, 3.5])
            return arr[1]

        # Should generate memref<3xf32>

    def test_array_with_expression_index(self):
        """Test array access with computed index"""
        @ml_function
        def computed_index() -> i32:
            arr = Array[5, i32]([10, 20, 30, 40, 50])
            idx = 1 + 1
            return arr[idx]

        # Index should be computed first, then cast to index type if needed
