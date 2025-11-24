"""Tests for Array type system (Phase 7 - Step 1)

This test suite validates:
- Array[size, dtype] type creation syntax
- Type validation (size must be positive int, dtype must be ScalarType)
- Array type equality and hashing
- MLIR string generation (memref<NxT>)
- Integration with TypeSystem.parse_type_hint()
"""

import pytest
from mlir_edsl import Array, i32, f32, i1
from mlir_edsl.types import ArrayType, ScalarType, TypeSystem


# ==================== ARRAY TYPE CREATION ====================

class TestArrayTypeCreation:
    """Test Array[size, dtype] type creation"""

    def test_array_type_creation_i32(self):
        """Test creating Array[10, i32] type"""
        arr_type = Array[10, i32]

        assert isinstance(arr_type, ArrayType)
        assert arr_type.size == 10
        assert arr_type.element_type == i32
        assert arr_type.element_enum == 0  # I32 protobuf enum

    def test_array_type_creation_f32(self):
        """Test creating Array[5, f32] type"""
        arr_type = Array[5, f32]

        assert isinstance(arr_type, ArrayType)
        assert arr_type.size == 5
        assert arr_type.element_type == f32
        assert arr_type.element_enum == 1  # F32 protobuf enum

    def test_array_type_creation_i1(self):
        """Test creating Array[3, i1] type"""
        arr_type = Array[3, i1]

        assert isinstance(arr_type, ArrayType)
        assert arr_type.size == 3
        assert arr_type.element_type == i1
        assert arr_type.element_enum == 2  # I1 protobuf enum

    def test_array_type_various_sizes(self):
        """Test arrays with various sizes"""
        arr1 = Array[1, i32]
        arr100 = Array[100, i32]
        arr1000 = Array[1000, f32]

        assert arr1.size == 1
        assert arr100.size == 100
        assert arr1000.size == 1000


# ==================== ARRAY TYPE VALIDATION ====================

class TestArrayTypeValidation:
    """Test Array type validation"""

    def test_array_requires_two_parameters(self):
        """Test that Array requires requires 2 parameters"""
        with pytest.raises(TypeError, match="requires 2 parameters"):
            Array[10]  # Missing element type

    def test_array_size_must_be_positive_int(self):
        """Test that array size must be positive integer"""
        with pytest.raises(TypeError, match="positive integer"):
            Array[0, i32]  # Size 0 invalid

        with pytest.raises(TypeError, match="positive integer"):
            Array[-5, i32]  # Negative size invalid

        with pytest.raises(TypeError, match="positive integer"):
            Array[3.5, i32]  # Float size invalid

        with pytest.raises(TypeError, match="positive integer"):
            Array["10", i32]  # String size invalid

    def test_array_element_type_must_be_scalar_type(self):
        """Test that element type must be i32, f32, or i1"""
        with pytest.raises(TypeError, match="must be i32, f32, or i1"):
            Array[10, int]  # Python int, not i32

        with pytest.raises(TypeError, match="must be i32, f32, or i1"):
            Array[10, float]  # Python float, not f32

        with pytest.raises(TypeError, match="must be i32, f32, or i1"):
            Array[10, "i32"]  # String, not ScalarType

    def test_array_reject_nested_arrays(self):
        """Test that nested arrays are rejected (for now)"""
        inner_array = Array[5, i32]

        # This should fail - element type must be ScalarType
        with pytest.raises(TypeError, match="must be i32, f32, or i1"):
            Array[10, inner_array]  # Array of arrays not supported yet


# ==================== ARRAY TYPE EQUALITY ====================

class TestArrayTypeEquality:
    """Test Array type equality and hashing"""

    def test_array_type_equality_same(self):
        """Test that same array types are equal"""
        arr1 = Array[10, i32]
        arr2 = Array[10, i32]

        assert arr1 == arr2
        assert not (arr1 != arr2)

    def test_array_type_equality_different_size(self):
        """Test that different sizes make different types"""
        arr1 = Array[10, i32]
        arr2 = Array[5, i32]

        assert arr1 != arr2
        assert not (arr1 == arr2)

    def test_array_type_equality_different_element_type(self):
        """Test that different element types make different types"""
        arr1 = Array[10, i32]
        arr2 = Array[10, f32]

        assert arr1 != arr2
        assert not (arr1 == arr2)

    def test_array_type_not_equal_to_scalar(self):
        """Test that array type is not equal to scalar type"""
        arr = Array[10, i32]

        assert arr != i32
        assert arr != f32

    def test_array_type_hashable(self):
        """Test that ArrayType can be used as dict key"""
        arr_type = Array[10, i32]

        type_dict = {arr_type: "test_value"}
        assert type_dict[arr_type] == "test_value"

    def test_array_types_in_set(self):
        """Test that ArrayTypes work in sets (requires __hash__)"""
        arr1 = Array[10, i32]
        arr2 = Array[10, i32]  # Same type
        arr3 = Array[5, i32]   # Different size

        type_set = {arr1, arr2, arr3}
        assert len(type_set) == 2  # arr1 and arr2 are same, only counted once


# ==================== MLIR STRING GENERATION ====================

class TestArrayMLIRString:
    """Test MLIR string generation"""

    def test_array_to_mlir_string_i32(self):
        """Test Array[10, i32] -> memref<10xi32>"""
        arr_type = Array[10, i32]
        assert arr_type.to_mlir_string() == "memref<10xi32>"

    def test_array_to_mlir_string_f32(self):
        """Test Array[5, f32] -> memref<5xf32>"""
        arr_type = Array[5, f32]
        assert arr_type.to_mlir_string() == "memref<5xf32>"

    def test_array_to_mlir_string_i1(self):
        """Test Array[3, i1] -> memref<3xi1>"""
        arr_type = Array[3, i1]
        assert arr_type.to_mlir_string() == "memref<3xi1>"

    def test_array_to_mlir_string_large_size(self):
        """Test MLIR string with large array size"""
        arr_type = Array[1024, i32]
        assert arr_type.to_mlir_string() == "memref<1024xi32>"


# ==================== ARRAY TYPE REPRESENTATION ====================

class TestArrayTypeRepresentation:
    """Test Array type __repr__"""

    def test_array_repr_i32(self):
        """Test readable representation of Array[10, i32]"""
        arr_type = Array[10, i32]
        assert repr(arr_type) == "Array[10, i32]"

    def test_array_repr_f32(self):
        """Test readable representation of Array[5, f32]"""
        arr_type = Array[5, f32]
        assert repr(arr_type) == "Array[5, f32]"


# ==================== TYPE SYSTEM INTEGRATION ====================

class TestTypeSystemIntegration:
    """Test integration with TypeSystem.parse_type_hint()"""

    def test_parse_array_type_hint(self):
        """Test that TypeSystem can parse Array type hints"""
        arr_type = Array[10, i32]
        parsed = TypeSystem.parse_type_hint(arr_type, context="parameter")

        # Should return the ArrayType itself, not an enum
        assert isinstance(parsed, ArrayType)
        assert parsed == arr_type

    def test_parse_scalar_type_hint_still_works(self):
        """Test that scalar type hints still work"""
        parsed_i32 = TypeSystem.parse_type_hint(i32)
        parsed_f32 = TypeSystem.parse_type_hint(f32)
        parsed_int = TypeSystem.parse_type_hint(int)

        # Scalars return enum values
        assert isinstance(parsed_i32, int)
        assert isinstance(parsed_f32, int)
        assert isinstance(parsed_int, int)

    def test_parse_mixed_types(self):
        """Test parsing both scalar and array types"""
        scalar = TypeSystem.parse_type_hint(i32)
        array = TypeSystem.parse_type_hint(Array[10, i32])

        # Scalar returns int enum, Array returns ArrayType
        assert isinstance(scalar, int)
        assert isinstance(array, ArrayType)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
