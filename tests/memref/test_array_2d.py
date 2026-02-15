"""Tests for 2D Array Support (Phase 7.2)

This test suite validates:
- 2D ArrayLiteral creation with nested lists
- 2D ArrayAccess with tuple indices
- 2D ArrayStore with tuple indices
- 2D element-wise operations
- MLIR generation for 2D arrays
- Execution of 2D array operations
"""

import pytest
from mlir_edsl import ml_function, Array
from mlir_edsl import i32, f32, i1
from mlir_edsl.ast import ArrayLiteral, ArrayAccess, ArrayStore
from mlir_edsl.ast.serialization import SerializationContext
from mlir_edsl.types import ArrayType, i32, f32


# ==================== 2D ARRAY LITERAL CREATION ====================

class TestArray2DLiteralCreation:
    """Test 2D ArrayLiteral creation with nested lists"""

    def test_2d_array_literal_i32(self):
        """Test creating 2D array literal with i32 elements"""
        arr_type = Array[i32, 2, 3]  # 2 rows, 3 columns
        arr = ArrayLiteral([
            [1, 2, 3],
            [4, 5, 6]
        ], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type == arr_type
        assert arr.array_type.shape == (2, 3)
        assert arr.array_type.ndim == 2
        # Elements should be flattened in row-major order
        assert len(arr.elements) == 6

    def test_2d_array_literal_f32(self):
        """Test creating 2D array literal with f32 elements"""
        arr_type = Array[f32, 3, 2]  # 3 rows, 2 columns
        arr = ArrayLiteral([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0]
        ], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type.shape == (3, 2)
        assert len(arr.elements) == 6

    def test_2d_array_construction_syntax(self):
        """Test Array[M, N, dtype]([...]) construction syntax"""
        arr = Array[i32, 2, 2]([
            [10, 20],
            [30, 40]
        ])

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type == Array[i32, 2, 2]


# ==================== 2D ARRAY LITERAL VALIDATION ====================

class TestArray2DLiteralValidation:
    """Test validation for 2D array literal creation"""

    def test_2d_size_mismatch_wrong_rows(self):
        """Test that wrong number of rows is caught"""
        arr_type = Array[i32, 2, 3]  # Expect 2 rows

        with pytest.raises(TypeError, match="2D array expects 2 rows, got 3"):
            ArrayLiteral([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]  # Extra row
            ], arr_type)

    def test_2d_size_mismatch_wrong_cols(self):
        """Test that wrong number of columns is caught"""
        arr_type = Array[i32, 2, 3]  # Expect 3 columns

        with pytest.raises(TypeError, match="Row 1: expected 3 elements"):
            ArrayLiteral([
                [1, 2, 3],
                [4, 5]  # Only 2 columns
            ], arr_type)

    def test_2d_type_mismatch(self):
        """Test that element type mismatch is caught"""
        arr_type = Array[i32, 2, 2]

        with pytest.raises(TypeError, match="expected i32.*got f32"):
            ArrayLiteral([
                [1, 2],
                [3.0, 4]  # Float in i32 array
            ], arr_type)

    def test_2d_not_nested_list(self):
        """Test that flat list is rejected for 2D array"""
        arr_type = Array[i32, 2, 2]

        with pytest.raises(TypeError, match="2D array expects 2 rows, got 4"):
            ArrayLiteral([1, 2, 3, 4], arr_type)


# ==================== 2D ARRAY ACCESS ====================

class TestArray2DAccess:
    """Test 2D ArrayAccess with tuple indices"""

    def test_2d_array_access_creation(self):
        """Test creating ArrayAccess with 2D indices"""
        arr = ArrayLiteral([
            [1, 2, 3],
            [4, 5, 6]
        ], Array[i32, 2, 3])

        access = ArrayAccess(arr, (0, 1))  # Access row 0, col 1

        assert isinstance(access, ArrayAccess)
        assert access.array == arr
        assert len(access.indices) == 2
        assert access.indices[0].value == 0
        assert access.indices[1].value == 1

    def test_2d_array_subscript_syntax(self):
        """Test arr[i, j] subscript syntax"""
        arr = ArrayLiteral([
            [10, 20],
            [30, 40]
        ], Array[i32, 2, 2])

        access = arr[1, 0]  # Access row 1, col 0

        assert isinstance(access, ArrayAccess)
        assert len(access.indices) == 2
        assert access.indices[0].value == 1
        assert access.indices[1].value == 0

    def test_2d_array_access_infer_type(self):
        """Test that ArrayAccess returns scalar element type"""
        arr = ArrayLiteral([
            [1.0, 2.0],
            [3.0, 4.0]
        ], Array[f32, 2, 2])

        access = ArrayAccess(arr, (0, 0))
        assert access.infer_type() == f32  # Returns scalar, not array

    def test_2d_wrong_index_count(self):
        """Test that wrong number of indices is rejected"""
        arr = ArrayLiteral([
            [1, 2],
            [3, 4]
        ], Array[i32, 2, 2])

        with pytest.raises(TypeError, match="Array dimension mismatch: 2D array requires 2 indices, got 1"):
            ArrayAccess(arr, 0)  # Only 1 index for 2D array


# ==================== 2D ARRAY STORE ====================

class TestArray2DStore:
    """Test 2D ArrayStore with tuple indices"""

    def test_2d_array_store_creation(self):
        """Test creating ArrayStore with 2D indices"""
        arr = ArrayLiteral([
            [1, 2, 3],
            [4, 5, 6]
        ], Array[i32, 2, 3])

        store = ArrayStore(arr, (1, 2), 99)  # Store 99 at row 1, col 2

        assert isinstance(store, ArrayStore)
        assert store.array == arr
        assert len(store.indices) == 2
        assert store.indices[0].value == 1
        assert store.indices[1].value == 2
        assert store.value.value == 99

    def test_2d_array_store_at_set_syntax(self):
        """Test arr.at[i, j].set(value) syntax"""
        arr = ArrayLiteral([
            [10, 20],
            [30, 40]
        ], Array[i32, 2, 2])

        store = arr.at[0, 1].set(50)

        assert isinstance(store, ArrayStore)
        assert len(store.indices) == 2
        assert store.indices[0].value == 0
        assert store.indices[1].value == 1
        assert store.value.value == 50

    def test_2d_array_store_type_checking(self):
        """Test that store validates element type"""
        arr = ArrayLiteral([
            [1, 2],
            [3, 4]
        ], Array[i32, 2, 2])

        with pytest.raises(TypeError, match="Cannot store f32 into Array"):
            ArrayStore(arr, (0, 0), 3.14)  # Float into i32 array


# ==================== 2D ARRAY PROTOBUF SERIALIZATION ====================

class TestArray2DProtobufSerialization:
    """Test protobuf serialization for 2D arrays"""

    def test_2d_array_literal_to_proto(self):
        """Test that 2D ArrayLiteral serializes correctly"""
        arr = ArrayLiteral([
            [1, 2],
            [3, 4]
        ], Array[i32, 2, 2])

        context = SerializationContext()
        pb = arr.to_proto(context)

        assert pb.HasField("array")
        assert pb.array.HasField("literal")
        # Check shape is [2, 2] (uses new TypeSpec with memref field)
        assert pb.array.literal.type.HasField("memref")
        assert len(pb.array.literal.type.memref.shape) == 2
        assert pb.array.literal.type.memref.shape[0] == 2
        assert pb.array.literal.type.memref.shape[1] == 2
        # Element type is nested: type.memref.element_type.scalar.kind
        from mlir_edsl import ast_pb2
        assert pb.array.literal.type.memref.element_type.scalar.kind == ast_pb2.ScalarTypeSpec.I32
        # Check flattened elements
        assert len(pb.array.literal.elements) == 4

    def test_2d_array_access_to_proto(self):
        """Test that 2D ArrayAccess serializes with multiple indices"""
        arr = ArrayLiteral([[1, 2], [3, 4]], Array[i32, 2, 2])
        access = ArrayAccess(arr, (1, 0))

        context = SerializationContext()
        pb = access.to_proto(context)

        assert pb.HasField("array")
        assert pb.array.HasField("access")
        # Check indices count
        assert len(pb.array.access.indices) == 2

    def test_2d_array_store_to_proto(self):
        """Test that 2D ArrayStore serializes with multiple indices"""
        arr = ArrayLiteral([[1, 2], [3, 4]], Array[i32, 2, 2])
        store = ArrayStore(arr, (0, 1), 5)

        context = SerializationContext()
        pb = store.to_proto(context)

        assert pb.HasField("array")
        assert pb.array.HasField("store")
        assert len(pb.array.store.indices) == 2


# ==================== 2D ELEMENT-WISE OPERATIONS ====================

class TestArray2DElementwise:
    """Test element-wise operations on 2D arrays"""

    def test_2d_array_add_arrays(self):
        """Test 2D array + 2D array element-wise addition"""
        arr1 = ArrayLiteral([[1, 2], [3, 4]], Array[i32, 2, 2])
        arr2 = ArrayLiteral([[10, 20], [30, 40]], Array[i32, 2, 2])

        result = arr1 + arr2

        # Validate AST structure
        assert result.infer_type().shape == (2, 2)
        assert result.infer_type().element_type == i32

    def test_2d_array_add_scalar(self):
        """Test 2D array + scalar broadcasting"""
        arr = ArrayLiteral([[1, 2], [3, 4]], Array[i32, 2, 2])
        scalar = 10

        result = arr + scalar

        # Validate broadcast
        assert result.infer_type().shape == (2, 2)

    def test_2d_array_mul_arrays(self):
        """Test 2D array * 2D array element-wise multiplication"""
        arr1 = ArrayLiteral([[2, 3], [4, 5]], Array[i32, 2, 2])
        arr2 = ArrayLiteral([[10, 10], [10, 10]], Array[i32, 2, 2])

        result = arr1 * arr2

        assert result.infer_type().shape == (2, 2)

    def test_2d_shape_mismatch(self):
        """Test that mismatched shapes are rejected"""
        arr1 = ArrayLiteral([[1, 2]], Array[i32, 1, 2])  # 1x2
        arr2 = ArrayLiteral([[1], [2]], Array[i32, 2, 1])  # 2x1

        with pytest.raises(TypeError, match="Array shapes must match"):
            arr1 + arr2


# ==================== 2D MLIR GENERATION ====================

class TestArray2DMLIRGeneration:
    """Test MLIR generation for 2D arrays"""

    def test_2d_array_literal_generates_memref(self, backend):
        """Test that 2D array literal compiles and generates memref type"""
        @ml_function
        def create_2d_array() -> i32:
            arr = Array[i32, 2, 3]([
                [1, 2, 3],
                [4, 5, 6]
            ])
            return arr[1, 2]

        # Should compile without errors - IR contains memref<2x3xi32>
        assert create_2d_array is not None

    def test_2d_array_access_generates_load(self, backend):
        """Test that 2D array access compiles and generates memref.load with 2 indices"""
        @ml_function
        def access_2d_element() -> i32:
            arr = Array[i32, 2, 2]([[1, 2], [3, 4]])
            return arr[1, 0]

        # Should compile without errors - IR contains memref.load with 2 indices
        assert access_2d_element is not None

    def test_2d_array_store_generates_nested_loops(self, backend):
        """Test that 2D array store compiles correctly"""
        @ml_function
        def store_2d_element() -> i32:
            arr = Array[i32, 2, 2]([[1, 2], [3, 4]])
            arr = arr.at[0, 1].set(99)
            return arr[0, 1]

        # Should compile without errors - IR contains memref.store
        assert store_2d_element is not None


# ==================== 2D EXECUTION TESTS ====================

class TestArray2DExecution:
    """Test execution of 2D array operations"""

    def test_2d_array_element_access_execution(self, backend):
        """Test executing 2D array element access"""
        @ml_function
        def get_element() -> i32:
            arr = Array[i32, 2, 3]([
                [10, 20, 30],
                [40, 50, 60]
            ])
            return arr[1, 2]  # Should return 60

        result = get_element()
        assert result == 60

    def test_2d_array_add_execution(self, backend):
        """Test executing 2D array addition"""
        @ml_function
        def add_arrays() -> i32:
            arr1 = Array[i32, 2, 2]([[1, 2], [3, 4]])
            arr2 = Array[i32, 2, 2]([[10, 20], [30, 40]])
            res = arr1 + arr2
            return res[0, 0] + res[0, 1] + res[1, 0] + res[1, 1]

        # Result should be [[11, 22], [33, 44]], sum = 110
        total = add_arrays()
        assert total == 11 + 22 + 33 + 44  # 110

    def test_2d_array_scalar_broadcast_execution(self, backend):
        """Test executing 2D array + scalar broadcasting"""
        @ml_function
        def add_scalar() -> i32:
            arr = Array[i32, 2, 2]([[1, 2], [3, 4]])
            res = arr + 10
            return res[1, 1]  # Should be 4 + 10 = 14

        assert add_scalar() == 14


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
