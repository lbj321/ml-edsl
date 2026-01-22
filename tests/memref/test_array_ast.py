"""Tests for Array AST nodes (Phase 7 - Step 2)

This test suite validates:
- ArrayLiteral creation with compile-time type checking
- ArrayAccess (arr[i]) with type validation
- ArrayStore (arr[i] = value) with strict type checking
- Type inference for array operations
- Integration with subscript syntax (__getitem__ / __setitem__)
"""

import pytest
from mlir_edsl import Array, i32, f32, i1
from mlir_edsl.ast import ArrayLiteral, ArrayAccess, ArrayStore, Constant
from mlir_edsl.ast.serialization import SerializationContext
from mlir_edsl.types import ArrayType, I32, F32, I1


# ==================== ARRAY LITERAL CREATION ====================

class TestArrayLiteralCreation:
    """Test ArrayLiteral AST node creation"""

    def test_array_literal_i32(self):
        """Test creating array literal with i32 elements"""
        arr_type = Array[4, i32]
        arr = ArrayLiteral([1, 2, 3, 4], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert len(arr.elements) == 4
        assert arr.array_type == arr_type

    def test_array_literal_f32(self):
        """Test creating array literal with f32 elements"""
        arr_type = Array[3, f32]
        arr = ArrayLiteral([1.0, 2.5, 3.14], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert len(arr.elements) == 3

    def test_array_literal_elements_converted_to_ast_nodes(self):
        """Test that Python literals are converted to Constant nodes"""
        arr_type = Array[2, i32]
        arr = ArrayLiteral([10, 20], arr_type)

        # Elements should be Constant AST nodes
        assert all(isinstance(elem, Constant) for elem in arr.elements)
        assert arr.elements[0].value == 10
        assert arr.elements[1].value == 20


# ==================== ARRAY LITERAL TYPE CHECKING ====================

class TestArrayLiteralTypeChecking:
    """Test compile-time type checking in ArrayLiteral"""

    def test_array_size_mismatch(self):
        """Test that size mismatch is caught"""
        arr_type = Array[4, i32]

        with pytest.raises(TypeError, match="Array size mismatch"):
            ArrayLiteral([1, 2, 3], arr_type)  # Only 3 elements, expected 4

    def test_array_element_type_mismatch(self):
        """Test that element type mismatch is caught"""
        arr_type = Array[3, i32]

        with pytest.raises(TypeError, match="Array element type mismatch.*index 1"):
            ArrayLiteral([1, 2.5, 3], arr_type)  # 2.5 is float, expected int

    def test_array_all_elements_must_match(self):
        """Test that all elements must match declared type"""
        arr_type = Array[4, f32]

        # This should fail - int elements in f32 array
        with pytest.raises(TypeError, match="expected f32.*got i32"):
            ArrayLiteral([1, 2, 3, 4], arr_type)

    def test_array_nested_arrays_rejected(self):
        """Test that nested arrays are rejected"""
        inner = ArrayLiteral([1, 2], Array[2, i32])
        outer_type = Array[2, i32]

        with pytest.raises(TypeError, match="cannot be an array"):
            ArrayLiteral([inner, inner], outer_type)

    def test_array_mixed_types_rejected(self):
        """Test that mixed element types are rejected"""
        arr_type = Array[3, i32]

        with pytest.raises(TypeError, match="type mismatch"):
            ArrayLiteral([1, 2.5, 3], arr_type)


# ==================== ARRAY LITERAL TYPE INFERENCE ====================

class TestArrayLiteralTypeInference:
    """Test type inference for ArrayLiteral"""

    def test_array_literal_infer_type_returns_array_type(self):
        """Test that ArrayLiteral.infer_type() returns ArrayType"""
        arr_type = Array[4, i32]
        arr = ArrayLiteral([1, 2, 3, 4], arr_type)

        inferred = arr.infer_type()
        assert isinstance(inferred, ArrayType)
        assert inferred == arr_type

    def test_array_literal_type_preserves_size_and_element_type(self):
        """Test that inferred type has correct size and element type"""
        arr_type = Array[5, f32]
        arr = ArrayLiteral([1.0, 2.0, 3.0, 4.0, 5.0], arr_type)

        inferred = arr.infer_type()
        assert inferred.size == 5
        assert inferred.element_type == f32
        assert inferred.element_enum == F32


# ==================== ARRAY ACCESS ====================

class TestArrayAccess:
    """Test ArrayAccess AST node (arr[i])"""

    def test_array_access_creation(self):
        """Test creating ArrayAccess node"""
        arr = ArrayLiteral([1, 2, 3, 4], Array[4, i32])
        access = ArrayAccess(arr, 2)

        assert isinstance(access, ArrayAccess)
        assert access.array == arr
        assert isinstance(access.indices[0], Constant)
        assert access.indices[0].value == 2

    def test_array_access_with_value_index(self):
        """Test ArrayAccess with Value node as index"""
        arr = ArrayLiteral([10, 20, 30], Array[3, i32])
        idx = Constant(1, I32)
        access = ArrayAccess(arr, idx)

        assert access.indices[0] == idx

    def test_array_access_infer_type_returns_element_type(self):
        """Test that ArrayAccess returns element type, not array type"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        access = ArrayAccess(arr, 0)

        inferred = access.infer_type()
        assert isinstance(inferred, int)  # Returns scalar enum, not ArrayType
        assert inferred == I32

    def test_array_access_f32_array_returns_f32(self):
        """Test accessing f32 array returns f32"""
        arr = ArrayLiteral([1.0, 2.0, 3.0], Array[3, f32])
        access = ArrayAccess(arr, 1)

        assert access.infer_type() == F32


# ==================== ARRAY ACCESS TYPE CHECKING ====================

class TestArrayAccessTypeChecking:
    """Test compile-time type checking for ArrayAccess"""

    def test_array_access_requires_array(self):
        """Test that indexing non-array fails"""
        scalar = Constant(42, I32)

        with pytest.raises(TypeError, match="Cannot index into non-array"):
            ArrayAccess(scalar, 0)

    def test_array_access_index_must_be_i32(self):
        """Test that index must be i32"""
        arr = ArrayLiteral([1.0, 2.0], Array[2, f32])
        float_idx = Constant(1.5, F32)

        with pytest.raises(TypeError, match="Array index .* must be i32"):
            ArrayAccess(arr, float_idx)

    def test_array_access_python_int_index_converted(self):
        """Test that Python int index is converted to i32 Constant"""
        arr = ArrayLiteral([10, 20, 30], Array[3, i32])
        access = ArrayAccess(arr, 1)  # Python int

        assert isinstance(access.indices[0], Constant)
        assert access.indices[0].value_type == I32


# ==================== ARRAY STORE ====================

class TestArrayStore:
    """Test ArrayStore AST node (arr[i] = value)"""

    def test_array_store_creation(self):
        """Test creating ArrayStore node"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        store = ArrayStore(arr, 0, 99)

        assert isinstance(store, ArrayStore)
        assert store.array == arr
        assert isinstance(store.indices[0], Constant)
        assert store.indices[0].value == 0
        assert isinstance(store.value, Constant)
        assert store.value.value == 99

    def test_array_store_with_ast_nodes(self):
        """Test ArrayStore with Value nodes"""
        arr = ArrayLiteral([10, 20], Array[2, i32])
        idx = Constant(1, I32)
        val = Constant(42, I32)
        store = ArrayStore(arr, idx, val)

        assert store.indices[0] == idx
        assert store.value == val

    def test_array_store_infer_type_returns_array_type(self):
        """Test that ArrayStore returns array type"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        store = ArrayStore(arr, 0, 5)

        inferred = store.infer_type()
        assert isinstance(inferred, ArrayType)
        assert inferred == Array[3, i32]


# ==================== ARRAY STORE TYPE CHECKING ====================

class TestArrayStoreTypeChecking:
    """Test strict compile-time type checking for ArrayStore"""

    def test_array_store_requires_array(self):
        """Test that storing to non-array fails"""
        scalar = Constant(42, I32)

        with pytest.raises(TypeError, match="Cannot use \\[\\]= on non-array"):
            ArrayStore(scalar, 0, 99)

    def test_array_store_index_must_be_i32(self):
        """Test that index must be i32"""
        arr = ArrayLiteral([1, 2], Array[2, i32])
        float_idx = Constant(1.5, F32)

        with pytest.raises(TypeError, match="Array index .* must be i32"):
            ArrayStore(arr, float_idx, 99)

    def test_array_store_value_type_must_match(self):
        """Test strict type matching for stored value"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])

        # Try to store float into i32 array - should fail
        with pytest.raises(TypeError, match="Cannot store f32 into Array\\[\\.\\.\\., i32\\]"):
            ArrayStore(arr, 0, 3.14)

    def test_array_store_f32_array_accepts_float(self):
        """Test that f32 array accepts float values"""
        arr = ArrayLiteral([1.0, 2.0], Array[2, f32])
        store = ArrayStore(arr, 0, 5.5)  # Should work

        assert store.value.value == 5.5
        assert store.value.value_type == F32

    def test_array_store_rejects_array_in_element(self):
        """Test that storing array into array element fails"""
        arr = ArrayLiteral([1, 2], Array[2, i32])
        inner = ArrayLiteral([3, 4], Array[2, i32])

        with pytest.raises(TypeError, match="Cannot store array into array element"):
            ArrayStore(arr, 0, inner)


# ==================== SUBSCRIPT SYNTAX INTEGRATION ====================

class TestSubscriptSyntax:
    """Test __getitem__ and __setitem__ integration"""

    def test_getitem_creates_array_access(self):
        """Test that arr[i] creates ArrayAccess node"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        access = arr[1]

        assert isinstance(access, ArrayAccess)
        assert access.array == arr
        assert access.indices[0].value == 1

    def test_setitem_creates_array_store(self):
        """Test that arr[i] = value creates ArrayStore node"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        store = arr.at[0].set(99)

        assert isinstance(store, ArrayStore)
        assert store.array == arr
        assert store.indices[0].value == 0
        assert store.value.value == 99

    def test_subscript_type_checking_works(self):
        """Test that subscript syntax enforces type checking"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])

        # This should work
        access = arr[0]
        assert access.infer_type() == I32

        # This should fail - wrong value type
        with pytest.raises(TypeError, match="Cannot store f32"):
            arr.at[0].set(3.14)


# ==================== GET_CHILDREN FOR SERIALIZATION ====================

class TestArrayGetChildren:
    """Test get_children() for serialization traversal"""

    def test_array_literal_get_children_returns_elements(self):
        """Test that ArrayLiteral.get_children() returns element nodes"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        children = arr.get_children()

        assert len(children) == 3
        assert all(isinstance(child, Constant) for child in children)

    def test_array_access_get_children_returns_array_and_index(self):
        """Test that ArrayAccess.get_children() returns [array, index]"""
        arr = ArrayLiteral([10, 20], Array[2, i32])
        access = ArrayAccess(arr, 1)
        children = access.get_children()

        assert len(children) == 2
        assert children[0] == arr
        assert children[1] == access.indices[0]

    def test_array_store_get_children_returns_all_three(self):
        """Test that ArrayStore.get_children() returns [array, index, value]"""
        arr = ArrayLiteral([1, 2], Array[2, i32])
        store = ArrayStore(arr, 0, 99)
        children = store.get_children()

        assert len(children) == 3
        assert children[0] == arr
        assert children[1] == store.indices[0]
        assert children[2] == store.value


# ==================== PROTOBUF SERIALIZATION ====================

class TestArrayProtobufSerialization:
    """Test protobuf serialization for array operations"""

    def test_array_literal_to_proto(self):
        """Test that ArrayLiteral.to_proto() works"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])
        context = SerializationContext()
        pb = arr.to_proto(context)

        # Should have array_literal field set
        assert pb.HasField("array_literal")

        # Check array type spec (uses new TypeSpec with memref field)
        assert pb.array_literal.type.HasField("memref")
        assert len(pb.array_literal.type.memref.shape) == 1  # 1D array
        assert pb.array_literal.type.memref.shape[0] == 3    # size is 3
        # Element type is nested: type.memref.element_type.scalar.kind
        from mlir_edsl import ast_pb2
        assert pb.array_literal.type.memref.element_type.scalar.kind == ast_pb2.ScalarTypeSpec.I32

        # Check elements
        assert len(pb.array_literal.elements) == 3

    def test_array_access_to_proto(self):
        """Test that ArrayAccess.to_proto() works"""
        arr = ArrayLiteral([1, 2], Array[2, i32])
        access = ArrayAccess(arr, 0)
        context = SerializationContext()
        pb = access.to_proto(context)

        # Should have array_access field set
        assert pb.HasField("array_access")

        # Check that array and indices are serialized (now uses repeated field)
        assert pb.array_access.HasField("array")
        assert len(pb.array_access.indices) == 1  # 1D array has single index

    def test_array_store_to_proto(self):
        """Test that ArrayStore.to_proto() works"""
        arr = ArrayLiteral([1, 2], Array[2, i32])
        store = ArrayStore(arr, 0, 5)
        context = SerializationContext()
        pb = store.to_proto(context)

        # Should have array_store field set
        assert pb.HasField("array_store")

        # Check that array, indices, and value are serialized (now uses repeated field for indices)
        assert pb.array_store.HasField("array")
        assert len(pb.array_store.indices) == 1  # 1D array has single index
        assert pb.array_store.HasField("value")

    def test_array_literal_with_reuse(self):
        """Test ArrayLiteral serialization with SSA value reuse"""
        arr = ArrayLiteral([1, 2, 3], Array[3, i32])

        # Use to_proto_with_reuse (which handles SSA value reuse)
        pb = arr.to_proto_with_reuse()

        assert pb.HasField("array_literal")

    def test_nested_array_operations_to_proto(self):
        """Test serialization of nested array operations"""
        arr = ArrayLiteral([10, 20, 30], Array[3, i32])
        access = ArrayAccess(arr, 1)

        # Serialize the access (which contains the array literal)
        context = SerializationContext()
        pb = access.to_proto(context)

        assert pb.HasField("array_access")
        # The nested array should be serialized
        assert pb.array_access.array.HasField("array_literal")


# ==================== INTEGRATION WITH ARRAY TYPE CONSTRUCTION ====================

class TestArrayTypeIntegration:
    """Test that Array[N, T]([...]) syntax works end-to-end"""

    def test_array_type_call_creates_array_literal(self):
        """Test that Array[4, i32]([1,2,3,4]) creates ArrayLiteral"""
        arr = Array[4, i32]([1, 2, 3, 4])

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type == Array[4, i32]
        assert len(arr.elements) == 4

    def test_array_construction_validates_types(self):
        """Test that Array construction validates element types"""
        # This should fail - float elements in i32 array
        with pytest.raises(TypeError, match="type mismatch"):
            Array[3, i32]([1.0, 2.0, 3.0])

    def test_array_construction_validates_size(self):
        """Test that Array construction validates size"""
        # This should fail - 2 elements when 3 expected
        with pytest.raises(TypeError, match="size mismatch"):
            Array[3, i32]([1, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
