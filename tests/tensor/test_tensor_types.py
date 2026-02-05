"""Tests for Tensor type system (Phase 7 - Tensor Dialect Step 1)

This test suite validates:
- Tensor[size, dtype] type creation syntax
- Type validation (size must be positive int, dtype must be ScalarType)
- Tensor type equality and hashing
- MLIR string generation (tensor<NxT>)
- Tensor is NOT an ArrayType (sibling types, not inheritance)
- Integration with TypeSystem.parse_type_hint()
"""

import pytest
from mlir_edsl.types import Tensor, TensorType, ArrayType, ScalarType, TypeSystem
from mlir_edsl import i32, f32, i1, Array


# ==================== TENSOR TYPE CREATION ====================

class TestTensorTypeCreation:
    """Test Tensor[size, dtype] type creation"""

    def test_tensor_type_creation_f32(self):
        """Test creating Tensor[4, f32] type"""
        t_type = Tensor[4, f32]

        assert isinstance(t_type, TensorType)
        assert t_type.size == 4
        assert t_type.element_type == f32

    def test_tensor_type_creation_i32(self):
        """Test creating Tensor[10, i32] type"""
        t_type = Tensor[10, i32]

        assert isinstance(t_type, TensorType)
        assert t_type.size == 10
        assert t_type.element_type == i32

    def test_tensor_type_creation_i1(self):
        """Test creating Tensor[3, i1] type"""
        t_type = Tensor[3, i1]

        assert isinstance(t_type, TensorType)
        assert t_type.size == 3
        assert t_type.element_type == i1

    def test_tensor_type_various_sizes(self):
        """Test tensors with various sizes"""
        t1 = Tensor[1, f32]
        t100 = Tensor[100, i32]
        t1000 = Tensor[1000, f32]

        assert t1.size == 1
        assert t100.size == 100
        assert t1000.size == 1000

    def test_tensor_type_2d(self):
        """Test creating 2D tensor type"""
        t_type = Tensor[2, 3, f32]

        assert isinstance(t_type, TensorType)
        assert t_type.shape == (2, 3)
        assert t_type.ndim == 2
        assert t_type.element_type == f32

    def test_tensor_type_3d(self):
        """Test creating 3D tensor type"""
        t_type = Tensor[2, 3, 4, i32]

        assert isinstance(t_type, TensorType)
        assert t_type.shape == (2, 3, 4)
        assert t_type.ndim == 3
        assert t_type.element_type == i32

    def test_tensor_total_elements(self):
        """Test total_elements computation"""
        assert Tensor[4, f32].total_elements == 4
        assert Tensor[2, 3, i32].total_elements == 6
        assert Tensor[2, 3, 4, f32].total_elements == 24

    def test_tensor_size_only_for_1d(self):
        """Test that .size raises for multi-dimensional tensors"""
        t2d = Tensor[2, 3, f32]

        with pytest.raises(AttributeError, match="only available for 1D"):
            _ = t2d.size


# ==================== TENSOR TYPE VALIDATION ====================

class TestTensorTypeValidation:
    """Test Tensor type validation"""

    def test_tensor_requires_two_parameters(self):
        """Test that Tensor requires at least 2 parameters"""
        with pytest.raises(TypeError, match="requires parameters"):
            Tensor[10]

    def test_tensor_size_must_be_positive_int(self):
        """Test that tensor size must be positive integer"""
        with pytest.raises(TypeError, match="positive integer"):
            Tensor[0, f32]

        with pytest.raises(TypeError, match="positive integer"):
            Tensor[-5, f32]

        with pytest.raises(TypeError, match="positive integer"):
            Tensor[3.5, f32]

        with pytest.raises(TypeError, match="positive integer"):
            Tensor["10", f32]

    def test_tensor_element_type_must_be_scalar_type(self):
        """Test that element type must be i32, f32, or i1"""
        with pytest.raises(TypeError, match="element type"):
            Tensor[10, int]

        with pytest.raises(TypeError, match="element type"):
            Tensor[10, float]

        with pytest.raises(TypeError, match="element type"):
            Tensor[10, "f32"]

    def test_tensor_max_3d(self):
        """Test that 4D+ tensors are rejected"""
        with pytest.raises(TypeError, match="1D, 2D, and 3D"):
            Tensor[2, 3, 4, 5, f32]


# ==================== TENSOR IS NOT ARRAY ====================

class TestTensorNotArray:
    """Test that TensorType and ArrayType are distinct siblings"""

    def test_tensor_not_instance_of_array(self):
        """Test that TensorType is not an ArrayType"""
        t_type = Tensor[4, f32]

        assert not isinstance(t_type, ArrayType)

    def test_array_not_instance_of_tensor(self):
        """Test that ArrayType is not a TensorType"""
        a_type = Array[4, f32]

        assert not isinstance(a_type, TensorType)

    def test_tensor_not_equal_to_array_same_shape(self):
        """Test that Tensor[4, f32] != Array[4, f32]"""
        t_type = Tensor[4, f32]
        a_type = Array[4, f32]

        assert t_type != a_type
        assert a_type != t_type


# ==================== TENSOR TYPE EQUALITY ====================

class TestTensorTypeEquality:
    """Test Tensor type equality and hashing"""

    def test_tensor_type_equality_same(self):
        """Test that same tensor types are equal"""
        t1 = Tensor[4, f32]
        t2 = Tensor[4, f32]

        assert t1 == t2
        assert not (t1 != t2)

    def test_tensor_type_equality_different_size(self):
        """Test that different sizes make different types"""
        t1 = Tensor[4, f32]
        t2 = Tensor[8, f32]

        assert t1 != t2

    def test_tensor_type_equality_different_element_type(self):
        """Test that different element types make different types"""
        t1 = Tensor[4, f32]
        t2 = Tensor[4, i32]

        assert t1 != t2

    def test_tensor_type_not_equal_to_scalar(self):
        """Test that tensor type is not equal to scalar type"""
        t = Tensor[4, f32]

        assert t != f32
        assert t != i32

    def test_tensor_type_hashable(self):
        """Test that TensorType can be used as dict key"""
        t_type = Tensor[4, f32]

        type_dict = {t_type: "test_value"}
        assert type_dict[t_type] == "test_value"

    def test_tensor_types_in_set(self):
        """Test that TensorTypes work in sets"""
        t1 = Tensor[4, f32]
        t2 = Tensor[4, f32]  # Same type
        t3 = Tensor[8, f32]  # Different size

        type_set = {t1, t2, t3}
        assert len(type_set) == 2

    def test_tensor_and_array_different_hash(self):
        """Test that tensor and array with same shape have different hashes"""
        t = Tensor[4, f32]
        a = Array[4, f32]

        # They should not collide in a set
        type_set = {t, a}
        assert len(type_set) == 2


# ==================== MLIR STRING GENERATION ====================

class TestTensorMLIRString:
    """Test MLIR string generation"""

    def test_tensor_to_mlir_string_f32(self):
        """Test Tensor[4, f32] -> tensor<4xf32>"""
        assert Tensor[4, f32].to_mlir_string() == "tensor<4xf32>"

    def test_tensor_to_mlir_string_i32(self):
        """Test Tensor[10, i32] -> tensor<10xi32>"""
        assert Tensor[10, i32].to_mlir_string() == "tensor<10xi32>"

    def test_tensor_to_mlir_string_2d(self):
        """Test Tensor[2, 3, f32] -> tensor<2x3xf32>"""
        assert Tensor[2, 3, f32].to_mlir_string() == "tensor<2x3xf32>"

    def test_tensor_to_mlir_string_3d(self):
        """Test Tensor[2, 3, 4, i32] -> tensor<2x3x4xi32>"""
        assert Tensor[2, 3, 4, i32].to_mlir_string() == "tensor<2x3x4xi32>"


# ==================== TENSOR TYPE REPRESENTATION ====================

class TestTensorTypeRepresentation:
    """Test Tensor type __repr__"""

    def test_tensor_repr_1d(self):
        """Test repr of Tensor[4, f32]"""
        assert repr(Tensor[4, f32]) == "Tensor[4, f32]"

    def test_tensor_repr_2d(self):
        """Test repr of Tensor[2, 3, i32]"""
        assert repr(Tensor[2, 3, i32]) == "Tensor[2, 3, i32]"

    def test_tensor_repr_3d(self):
        """Test repr of Tensor[2, 3, 4, f32]"""
        assert repr(Tensor[2, 3, 4, f32]) == "Tensor[2, 3, 4, f32]"


# ==================== CATEGORY PREDICATES ====================

class TestTensorCategoryPredicates:
    """Test type category predicates"""

    def test_tensor_is_aggregate(self):
        """Test that tensor is an aggregate type"""
        t = Tensor[4, f32]

        assert t.is_aggregate()
        assert not t.is_scalar()

    def test_tensor_numeric_from_element_type(self):
        """Test that numeric predicate delegates to element type"""
        assert Tensor[4, f32].is_numeric()
        assert Tensor[4, i32].is_numeric()
        assert not Tensor[4, i1].is_numeric()

    def test_tensor_float_from_element_type(self):
        """Test that float predicate delegates to element type"""
        assert Tensor[4, f32].is_float()
        assert not Tensor[4, i32].is_float()

    def test_tensor_integer_from_element_type(self):
        """Test that integer predicate delegates to element type"""
        assert Tensor[4, i32].is_integer()
        assert not Tensor[4, f32].is_integer()

    def test_tensor_cannot_cast(self):
        """Test that tensors cannot be cast"""
        t = Tensor[4, f32]
        assert not t.can_cast_to(f32)
        assert not t.can_cast_to(Tensor[4, i32])


# ==================== TYPE SYSTEM INTEGRATION ====================

class TestTensorTypeSystemIntegration:
    """Test integration with TypeSystem.parse_type_hint()"""

    def test_parse_tensor_type_hint(self):
        """Test that TypeSystem can parse Tensor type hints"""
        t_type = Tensor[4, f32]
        parsed = TypeSystem.parse_type_hint(t_type, context="parameter")

        assert isinstance(parsed, TensorType)
        assert parsed == t_type

    def test_parse_tensor_does_not_return_array(self):
        """Test that parsing a TensorType doesn't return ArrayType"""
        t_type = Tensor[4, f32]
        parsed = TypeSystem.parse_type_hint(t_type)

        assert not isinstance(parsed, ArrayType)
        assert isinstance(parsed, TensorType)


# ==================== PROTOBUF SERIALIZATION ====================

class TestTensorTypeProtobuf:
    """Test protobuf serialization of TensorType"""

    def test_tensor_type_to_proto(self):
        """Test TensorType serializes to protobuf with tensor field"""
        t_type = Tensor[4, f32]
        proto = t_type.to_proto()

        assert proto.HasField("tensor")
        assert list(proto.tensor.shape) == [4]
        assert proto.tensor.element_type.HasField("scalar")

    def test_tensor_type_to_proto_2d(self):
        """Test 2D TensorType serializes shape correctly"""
        t_type = Tensor[2, 3, i32]
        proto = t_type.to_proto()

        assert proto.HasField("tensor")
        assert list(proto.tensor.shape) == [2, 3]

    def test_tensor_type_to_proto_element_type(self):
        """Test tensor protobuf includes correct element type"""
        from mlir_edsl import ast_pb2

        t_f32 = Tensor[4, f32]
        assert t_f32.to_proto().tensor.element_type.scalar.kind == ast_pb2.ScalarTypeSpec.F32

        t_i32 = Tensor[4, i32]
        assert t_i32.to_proto().tensor.element_type.scalar.kind == ast_pb2.ScalarTypeSpec.I32

    def test_tensor_proto_differs_from_array_proto(self):
        """Test that tensor and array serialize to different protobuf fields"""
        t_proto = Tensor[4, f32].to_proto()
        a_proto = Array[4, f32].to_proto()

        assert t_proto.HasField("tensor")
        assert not t_proto.HasField("memref")

        assert a_proto.HasField("memref")
        assert not a_proto.HasField("tensor")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
