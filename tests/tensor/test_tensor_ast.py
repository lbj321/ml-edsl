"""Tests for Tensor AST nodes (Phase 7 - Tensor Dialect Step 1)

This test suite validates:
- TensorFromElements creation with compile-time type checking
- TensorExtract (t[i]) with type validation
- Type inference for tensor operations
- __getitem__ dispatch (tensor[i] -> TensorExtract, not ArrayAccess)
- Protobuf serialization
"""

import pytest
from mlir_edsl.types import Tensor, TensorType, ScalarType, i32, f32, i1, Array
from mlir_edsl.ast import TensorFromElements, TensorExtract, Constant, ArrayAccess
from mlir_edsl.ast.serialization import SerializationContext


# ==================== TENSOR FROM ELEMENTS CREATION ====================

class TestTensorFromElementsCreation:
    """Test TensorFromElements AST node creation"""

    def test_tensor_from_elements_f32(self):
        """Test creating tensor literal with f32 elements"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])

        assert isinstance(t, TensorFromElements)
        assert len(t.elements) == 4
        assert t.tensor_type == Tensor[4, f32]

    def test_tensor_from_elements_i32(self):
        """Test creating tensor literal with i32 elements"""
        t = Tensor[3, i32]([10, 20, 30])

        assert isinstance(t, TensorFromElements)
        assert len(t.elements) == 3
        assert t.tensor_type == Tensor[3, i32]

    def test_tensor_elements_converted_to_ast_nodes(self):
        """Test that Python literals are converted to Constant nodes"""
        t = Tensor[2, i32]([10, 20])

        assert all(isinstance(elem, Constant) for elem in t.elements)
        assert t.elements[0].value == 10
        assert t.elements[1].value == 20

    def test_tensor_from_elements_2d(self):
        """Test creating 2D tensor from nested list"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])

        assert isinstance(t, TensorFromElements)
        assert len(t.elements) == 6  # Flattened
        assert t.tensor_type == Tensor[2, 3, i32]

    def test_tensor_from_elements_3d(self):
        """Test creating 3D tensor from triply nested list"""
        t = Tensor[2, 2, 2, i32]([[[1, 2], [3, 4]],
                                    [[5, 6], [7, 8]]])

        assert isinstance(t, TensorFromElements)
        assert len(t.elements) == 8  # Flattened
        assert t.tensor_type == Tensor[2, 2, 2, i32]


# ==================== TENSOR FROM ELEMENTS TYPE CHECKING ====================

class TestTensorFromElementsTypeChecking:
    """Test compile-time type checking in TensorFromElements"""

    def test_tensor_size_mismatch(self):
        """Test that size mismatch is caught"""
        with pytest.raises(TypeError, match="Tensor size mismatch"):
            Tensor[4, f32]([1.0, 2.0, 3.0])  # 3 elements, expected 4

    def test_tensor_element_type_mismatch(self):
        """Test that element type mismatch is caught"""
        with pytest.raises(TypeError, match="Tensor element type mismatch.*index 0"):
            Tensor[3, f32]([1, 2, 3])  # int elements in f32 tensor

    def test_tensor_i32_rejects_float(self):
        """Test that i32 tensor rejects float elements"""
        with pytest.raises(TypeError, match="expected i32.*got f32"):
            Tensor[2, i32]([1.0, 2.0])

    def test_tensor_2d_row_count_mismatch(self):
        """Test that 2D tensor validates row count"""
        with pytest.raises(TypeError, match="2D tensor expects 2 rows"):
            Tensor[2, 3, i32]([[1, 2, 3]])  # 1 row, expected 2

    def test_tensor_2d_column_count_mismatch(self):
        """Test that 2D tensor validates column count"""
        with pytest.raises(TypeError, match="expected 3 elements"):
            Tensor[2, 3, i32]([[1, 2], [3, 4]])  # 2 cols, expected 3

    def test_tensor_3d_structure_mismatch(self):
        """Test that 3D tensor validates structure"""
        with pytest.raises(TypeError, match="3D tensor expects 2 matrices"):
            Tensor[2, 2, 2, i32]([[[1, 2], [3, 4]]])  # 1 matrix, expected 2


# ==================== TENSOR FROM ELEMENTS TYPE INFERENCE ====================

class TestTensorFromElementsTypeInference:
    """Test type inference for TensorFromElements"""

    def test_tensor_infer_type_returns_tensor_type(self):
        """Test that TensorFromElements.infer_type() returns TensorType"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])

        inferred = t.infer_type()
        assert isinstance(inferred, TensorType)
        assert inferred == Tensor[4, f32]

    def test_tensor_infer_type_preserves_shape_and_element_type(self):
        """Test that inferred type has correct shape and element type"""
        t = Tensor[3, i32]([10, 20, 30])

        inferred = t.infer_type()
        assert inferred.size == 3
        assert inferred.element_type == i32

    def test_tensor_2d_infer_type(self):
        """Test type inference for 2D tensor"""
        t = Tensor[2, 3, f32]([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]])

        inferred = t.infer_type()
        assert isinstance(inferred, TensorType)
        assert inferred.shape == (2, 3)


# ==================== TENSOR EXTRACT ====================

class TestTensorExtract:
    """Test TensorExtract AST node (t[i])"""

    def test_tensor_extract_creation(self):
        """Test creating TensorExtract node"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        extract = TensorExtract(t, 2)

        assert isinstance(extract, TensorExtract)
        assert extract.tensor == t
        assert extract.indices[0].value == 2

    def test_tensor_extract_with_value_index(self):
        """Test TensorExtract with Value node as index"""
        t = Tensor[3, i32]([10, 20, 30])
        idx = Constant(1, i32)
        extract = TensorExtract(t, idx)

        assert extract.indices[0] == idx

    def test_tensor_extract_infer_type_returns_element_type(self):
        """Test that TensorExtract returns element type, not tensor type"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        extract = TensorExtract(t, 0)

        inferred = extract.infer_type()
        assert isinstance(inferred, ScalarType)
        assert inferred == f32

    def test_tensor_extract_i32_returns_i32(self):
        """Test extracting from i32 tensor returns i32"""
        t = Tensor[3, i32]([10, 20, 30])
        extract = TensorExtract(t, 1)

        assert extract.infer_type() == i32

    def test_tensor_extract_2d(self):
        """Test extracting from 2D tensor"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])
        extract = TensorExtract(t, (1, 2))

        assert len(extract.indices) == 2
        assert extract.infer_type() == i32


# ==================== TENSOR EXTRACT TYPE CHECKING ====================

class TestTensorExtractTypeChecking:
    """Test compile-time type checking for TensorExtract"""

    def test_tensor_extract_requires_tensor(self):
        """Test that indexing non-tensor fails"""
        scalar = Constant(42, i32)

        with pytest.raises(TypeError, match="Cannot index into non-tensor"):
            TensorExtract(scalar, 0)

    def test_tensor_extract_index_must_be_i32(self):
        """Test that index must be i32"""
        t = Tensor[2, f32]([1.0, 2.0])
        float_idx = Constant(1.5, f32)

        with pytest.raises(TypeError, match="Tensor index .* must be i32"):
            TensorExtract(t, float_idx)

    def test_tensor_extract_dimension_mismatch(self):
        """Test that 2D tensor requires 2 indices"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])

        with pytest.raises(TypeError, match="2D tensor requires 2 indices"):
            TensorExtract(t, 0)  # Only 1 index for 2D tensor

    def test_tensor_extract_too_many_indices(self):
        """Test that 1D tensor rejects 2 indices"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])

        with pytest.raises(TypeError, match="1D tensor requires 1 indices"):
            TensorExtract(t, (0, 1))


# ==================== SUBSCRIPT DISPATCH ====================

class TestSubscriptDispatch:
    """Test that __getitem__ correctly dispatches to TensorExtract vs ArrayAccess"""

    def test_tensor_getitem_creates_tensor_extract(self):
        """Test that tensor[i] creates TensorExtract, not ArrayAccess"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        result = t[2]

        assert isinstance(result, TensorExtract)
        assert not isinstance(result, ArrayAccess)

    def test_array_getitem_still_creates_array_access(self):
        """Test that array[i] still creates ArrayAccess"""
        arr = Array[4, i32]([1, 2, 3, 4])
        result = arr[2]

        assert isinstance(result, ArrayAccess)
        assert not isinstance(result, TensorExtract)

    def test_tensor_getitem_type_inference(self):
        """Test that tensor[i] has correct type inference"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        result = t[0]

        assert result.infer_type() == f32


# ==================== GET_CHILDREN FOR SERIALIZATION ====================

class TestTensorGetChildren:
    """Test get_children() for serialization traversal"""

    def test_tensor_from_elements_get_children_returns_elements(self):
        """Test that TensorFromElements.get_children() returns element nodes"""
        t = Tensor[3, i32]([10, 20, 30])
        children = t.get_children()

        assert len(children) == 3
        assert all(isinstance(child, Constant) for child in children)

    def test_tensor_extract_get_children_returns_tensor_and_index(self):
        """Test that TensorExtract.get_children() returns [tensor, index]"""
        t = Tensor[2, f32]([1.0, 2.0])
        extract = TensorExtract(t, 0)
        children = extract.get_children()

        assert len(children) == 2
        assert children[0] == t
        assert children[1] == extract.indices[0]

    def test_tensor_extract_2d_get_children(self):
        """Test get_children for 2D tensor extract"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])
        extract = TensorExtract(t, (1, 2))
        children = extract.get_children()

        assert len(children) == 3  # tensor + 2 indices


# ==================== PROTOBUF SERIALIZATION ====================

class TestTensorProtobufSerialization:
    """Test protobuf serialization for tensor operations"""

    def test_tensor_from_elements_to_proto(self):
        """Test that TensorFromElements serializes to protobuf"""
        t = Tensor[3, f32]([1.0, 2.0, 3.0])
        context = SerializationContext()
        pb = t.to_proto(context)

        assert pb.HasField("tensor")
        assert pb.tensor.HasField("from_elements")
        assert pb.tensor.from_elements.type.HasField("tensor")
        assert list(pb.tensor.from_elements.type.tensor.shape) == [3]
        assert len(pb.tensor.from_elements.elements) == 3

    def test_tensor_extract_to_proto(self):
        """Test that TensorExtract serializes to protobuf"""
        t = Tensor[2, i32]([10, 20])
        extract = TensorExtract(t, 0)
        context = SerializationContext()
        pb = extract.to_proto(context)

        assert pb.HasField("tensor")
        assert pb.tensor.HasField("extract")
        assert pb.tensor.extract.HasField("tensor")
        assert len(pb.tensor.extract.indices) == 1

    def test_tensor_from_elements_with_reuse(self):
        """Test TensorFromElements serialization with SSA value reuse"""
        t = Tensor[3, i32]([1, 2, 3])
        pb = t.to_proto_with_reuse()

        assert pb.HasField("tensor")
        assert pb.tensor.HasField("from_elements")

    def test_nested_tensor_operations_to_proto(self):
        """Test serialization of tensor extract containing tensor literal"""
        t = Tensor[3, f32]([1.0, 2.0, 3.0])
        extract = TensorExtract(t, 1)

        context = SerializationContext()
        pb = extract.to_proto(context)

        # The nested tensor should be serialized
        assert pb.HasField("tensor")
        assert pb.tensor.HasField("extract")
        assert pb.tensor.extract.tensor.HasField("tensor")
        assert pb.tensor.extract.tensor.tensor.HasField("from_elements")

    def test_tensor_proto_uses_tensor_field_not_array(self):
        """Test that tensor nodes use 'tensor' protobuf field, not 'array'"""
        t = Tensor[2, f32]([1.0, 2.0])
        context = SerializationContext()
        pb = t.to_proto(context)

        assert pb.HasField("tensor")
        assert not pb.HasField("array")


# ==================== CONSTRUCTION SYNTAX INTEGRATION ====================

class TestTensorConstructionSyntax:
    """Test that Tensor[N, T]([...]) syntax works end-to-end"""

    def test_tensor_type_call_creates_tensor_from_elements(self):
        """Test that Tensor[4, f32]([...]) creates TensorFromElements"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])

        assert isinstance(t, TensorFromElements)
        assert t.tensor_type == Tensor[4, f32]
        assert len(t.elements) == 4

    def test_tensor_construction_validates_types(self):
        """Test that Tensor construction validates element types"""
        with pytest.raises(TypeError, match="type mismatch"):
            Tensor[3, f32]([1, 2, 3])  # int elements in f32 tensor

    def test_tensor_construction_validates_size(self):
        """Test that Tensor construction validates size"""
        with pytest.raises(TypeError, match="size mismatch"):
            Tensor[3, f32]([1.0, 2.0])  # 2 elements when 3 expected

    def test_tensor_construction_then_extract(self):
        """Test full construction + extract chain"""
        t = Tensor[4, i32]([10, 20, 30, 40])
        val = t[2]

        assert isinstance(val, TensorExtract)
        assert val.infer_type() == i32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
