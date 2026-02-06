"""Tests for TensorInsert AST node and .at[].set() dispatch

This test suite validates:
- TensorInsert creation with compile-time type checking
- .at[].set() dispatch for tensors vs arrays
- Type inference (returns same TensorType)
- Protobuf serialization
- End-to-end execution
"""

import pytest
from mlir_edsl import ml_function, i32, f32, Tensor, Array
from mlir_edsl.ast import TensorFromElements, TensorExtract, TensorInsert, Constant, ArrayStore
from mlir_edsl.ast.serialization import SerializationContext
from mlir_edsl.types import TensorType, ScalarType
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


# ==================== TENSOR INSERT CREATION ====================

class TestTensorInsertCreation(MLIRTestBase):
    """Test TensorInsert AST node creation"""

    def test_tensor_insert_creation(self):
        """Test creating TensorInsert node"""
        t = Tensor[4, i32]([1, 2, 3, 4])
        insert = TensorInsert(t, 1, 99)

        assert isinstance(insert, TensorInsert)
        assert insert.tensor == t
        assert insert.indices[0].value == 1
        assert insert.value.value == 99

    def test_tensor_insert_with_value_index(self):
        """Test TensorInsert with Value node as index"""
        t = Tensor[3, f32]([1.0, 2.0, 3.0])
        idx = Constant(2, i32)
        insert = TensorInsert(t, idx, 9.0)

        assert insert.indices[0] == idx

    def test_tensor_insert_2d(self):
        """Test TensorInsert for 2D tensor"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])
        insert = TensorInsert(t, (1, 2), 99)

        assert len(insert.indices) == 2
        assert insert.indices[0].value == 1
        assert insert.indices[1].value == 2


# ==================== TENSOR INSERT TYPE CHECKING ====================

class TestTensorInsertTypeChecking(MLIRTestBase):
    """Test compile-time type checking for TensorInsert"""

    def test_tensor_insert_requires_tensor(self):
        """Test that inserting into non-tensor fails"""
        scalar = Constant(42, i32)

        with pytest.raises(TypeError, match="non-tensor type"):
            TensorInsert(scalar, 0, 99)

    def test_tensor_insert_index_must_be_i32(self):
        """Test that index must be i32"""
        t = Tensor[2, f32]([1.0, 2.0])
        float_idx = Constant(1.5, f32)

        with pytest.raises(TypeError, match="Tensor index .* must be i32"):
            TensorInsert(t, float_idx, 9.0)

    def test_tensor_insert_dimension_mismatch(self):
        """Test that 2D tensor requires 2 indices"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])

        with pytest.raises(TypeError, match="2D tensor requires 2 indices"):
            TensorInsert(t, 0, 99)  # Only 1 index for 2D tensor

    def test_tensor_insert_value_type_must_match(self):
        """Test that value type must match tensor element type"""
        t = Tensor[3, i32]([1, 2, 3])

        with pytest.raises(TypeError, match="Cannot insert f32 into.*i32"):
            TensorInsert(t, 1, 9.0)  # float into i32 tensor

    def test_tensor_insert_i32_into_f32_fails(self):
        """Test that i32 cannot be inserted into f32 tensor"""
        t = Tensor[3, f32]([1.0, 2.0, 3.0])

        with pytest.raises(TypeError, match="Cannot insert i32 into.*f32"):
            TensorInsert(t, 1, 99)  # int into f32 tensor


# ==================== TENSOR INSERT TYPE INFERENCE ====================

class TestTensorInsertTypeInference(MLIRTestBase):
    """Test type inference for TensorInsert"""

    def test_tensor_insert_returns_same_tensor_type(self):
        """Test that TensorInsert.infer_type() returns the same TensorType"""
        t = Tensor[4, i32]([1, 2, 3, 4])
        insert = TensorInsert(t, 1, 99)

        inferred = insert.infer_type()
        assert isinstance(inferred, TensorType)
        assert inferred == Tensor[4, i32]

    def test_tensor_insert_preserves_shape(self):
        """Test that inferred type has same shape"""
        t = Tensor[2, 3, f32]([[1.0, 2.0, 3.0],
                                [4.0, 5.0, 6.0]])
        insert = TensorInsert(t, (0, 1), 9.0)

        inferred = insert.infer_type()
        assert inferred.shape == (2, 3)
        assert inferred.element_type == f32


# ==================== .at[].set() DISPATCH ====================

class TestAtSetDispatch(MLIRTestBase):
    """Test that .at[].set() correctly dispatches to TensorInsert vs ArrayStore"""

    def test_tensor_at_set_creates_tensor_insert(self):
        """Test that tensor.at[i].set(v) creates TensorInsert"""
        t = Tensor[4, i32]([1, 2, 3, 4])
        result = t.at[1].set(99)

        assert isinstance(result, TensorInsert)
        assert not isinstance(result, ArrayStore)

    def test_array_at_set_still_creates_array_store(self):
        """Test that array.at[i].set(v) still creates ArrayStore"""
        arr = Array[4, i32]([1, 2, 3, 4])
        result = arr.at[1].set(99)

        assert isinstance(result, ArrayStore)
        assert not isinstance(result, TensorInsert)

    def test_tensor_at_set_type_inference(self):
        """Test that tensor.at[i].set(v) has correct type inference"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        result = t.at[0].set(9.0)

        assert result.infer_type() == Tensor[4, f32]

    def test_tensor_at_set_2d(self):
        """Test .at[i,j].set() for 2D tensor"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])
        result = t.at[1, 2].set(99)

        assert isinstance(result, TensorInsert)
        assert len(result.indices) == 2

    def test_scalar_at_set_fails(self):
        """Test that .at[].set() on scalar fails with clear error"""
        scalar = Constant(42, i32)

        with pytest.raises(TypeError, match="requires array or tensor"):
            scalar.at[0].set(99)


# ==================== .at[].get() DISPATCH ====================

class TestAtGetDispatch(MLIRTestBase):
    """Test that .at[].get() correctly dispatches to TensorExtract vs ArrayAccess"""

    def test_tensor_at_get_creates_tensor_extract(self):
        """Test that tensor.at[i].get() creates TensorExtract"""
        t = Tensor[4, i32]([1, 2, 3, 4])
        result = t.at[1].get()

        assert isinstance(result, TensorExtract)

    def test_scalar_at_get_fails(self):
        """Test that .at[].get() on scalar fails with clear error"""
        scalar = Constant(42, i32)

        with pytest.raises(TypeError, match="requires array or tensor"):
            scalar.at[0].get()


# ==================== GET_CHILDREN FOR SERIALIZATION ====================

class TestTensorInsertGetChildren(MLIRTestBase):
    """Test get_children() for TensorInsert serialization traversal"""

    def test_tensor_insert_get_children(self):
        """Test that TensorInsert.get_children() returns [tensor, indices..., value]"""
        t = Tensor[3, i32]([1, 2, 3])
        insert = TensorInsert(t, 1, 99)
        children = insert.get_children()

        assert len(children) == 3  # tensor + 1 index + value
        assert children[0] == t
        assert children[1] == insert.indices[0]
        assert children[2] == insert.value

    def test_tensor_insert_2d_get_children(self):
        """Test get_children for 2D tensor insert"""
        t = Tensor[2, 3, i32]([[1, 2, 3],
                                [4, 5, 6]])
        insert = TensorInsert(t, (1, 2), 99)
        children = insert.get_children()

        assert len(children) == 4  # tensor + 2 indices + value


# ==================== PROTOBUF SERIALIZATION ====================

class TestTensorInsertProtobuf(MLIRTestBase):
    """Test protobuf serialization for TensorInsert"""

    def test_tensor_insert_to_proto(self):
        """Test that TensorInsert serializes to protobuf"""
        t = Tensor[3, i32]([1, 2, 3])
        insert = TensorInsert(t, 1, 99)
        context = SerializationContext()
        pb = insert.to_proto(context)

        assert pb.HasField("tensor")
        assert pb.tensor.HasField("insert")
        assert pb.tensor.insert.HasField("tensor")
        assert pb.tensor.insert.HasField("value")
        assert len(pb.tensor.insert.indices) == 1

    def test_tensor_insert_proto_has_result_type(self):
        """Test that TensorInsert proto includes result type"""
        t = Tensor[4, f32]([1.0, 2.0, 3.0, 4.0])
        insert = TensorInsert(t, 0, 9.0)
        context = SerializationContext()
        pb = insert.to_proto(context)

        assert pb.tensor.insert.HasField("result_type")
        assert pb.tensor.insert.result_type.HasField("tensor")
        assert list(pb.tensor.insert.result_type.tensor.shape) == [4]


# ==================== END-TO-END EXECUTION ====================

class TestTensorInsertExecution(MLIRTestBase):
    """Test end-to-end execution of tensor.insert"""

    def test_tensor_insert_basic_execution(self):
        """Test basic tensor insert and extract"""
        @ml_function
        def insert_and_extract() -> i32:
            t = Tensor[4, i32]([10, 20, 30, 40])
            t = t.at[1].set(99)
            return t[1]

        result = insert_and_extract()
        assert result == 99

    def test_tensor_insert_preserves_other_elements(self):
        """Test that insert preserves other elements"""
        @ml_function
        def insert_check_others() -> i32:
            t = Tensor[4, i32]([10, 20, 30, 40])
            t = t.at[1].set(99)
            # Sum of unchanged elements
            return t[0] + t[2] + t[3]

        result = insert_check_others()
        assert result == 10 + 30 + 40  # 80

    def test_tensor_insert_multiple(self):
        """Test multiple inserts"""
        @ml_function
        def multiple_inserts() -> i32:
            t = Tensor[4, i32]([1, 2, 3, 4])
            t = t.at[0].set(10)
            t = t.at[3].set(40)
            return t[0] + t[3]

        result = multiple_inserts()
        assert result == 10 + 40  # 50

    def test_tensor_insert_f32(self):
        """Test tensor insert with f32"""
        @ml_function
        def insert_f32() -> f32:
            t = Tensor[3, f32]([1.0, 2.0, 3.0])
            t = t.at[1].set(9.5)
            return t[1]

        result = insert_f32()
        assert abs(result - 9.5) < 0.001

    def test_tensor_insert_2d_execution(self):
        """Test 2D tensor insert"""
        @ml_function
        def insert_2d() -> i32:
            t = Tensor[2, 3, i32]([[1, 2, 3],
                                    [4, 5, 6]])
            t = t.at[1, 2].set(99)
            return t[1, 2]

        result = insert_2d()
        assert result == 99

    def test_tensor_insert_with_computed_value(self):
        """Test inserting a computed value"""
        @ml_function
        def insert_computed() -> i32:
            t = Tensor[3, i32]([10, 20, 30])
            val = t[0] + t[1]  # 30
            t = t.at[2].set(val)
            return t[2]

        result = insert_computed()
        assert result == 30

    def test_tensor_insert_chain(self):
        """Test chained inserts"""
        @ml_function
        def chain_inserts() -> i32:
            t = Tensor[4, i32]([0, 0, 0, 0])
            t = t.at[0].set(1)
            t = t.at[1].set(2)
            t = t.at[2].set(3)
            t = t.at[3].set(4)
            return t[0] + t[1] + t[2] + t[3]

        result = chain_inserts()
        assert result == 1 + 2 + 3 + 4  # 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
