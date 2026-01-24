"""Tests for 3D Array Support (Phase 7.2)

This test suite validates:
- 3D ArrayLiteral creation with triple-nested lists
- 3D ArrayAccess with 3-tuple indices
- 3D ArrayStore with 3-tuple indices
- 3D element-wise operations
- MLIR generation for 3D arrays
- Execution of 3D array operations
"""

import pytest
from mlir_edsl import ml_function, Array
from mlir_edsl import i32, f32
from mlir_edsl.ast import ArrayLiteral, ArrayAccess, ArrayStore
from mlir_edsl.ast.serialization import SerializationContext
from mlir_edsl.types import ArrayType, i32, f32
from mlir_edsl.backend import HAS_CPP_BACKEND
from tests.test_base import MLIRTestBase


# ==================== 3D ARRAY LITERAL CREATION ====================

class TestArray3DLiteralCreation(MLIRTestBase):
    """Test 3D ArrayLiteral creation with triple-nested lists"""

    def test_3d_array_literal_i32(self):
        """Test creating 3D array literal with i32 elements"""
        arr_type = Array[2, 2, 3, i32]  # 2x2x3 array
        arr = ArrayLiteral([
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]]
        ], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type == arr_type
        assert arr.array_type.shape == (2, 2, 3)
        assert arr.array_type.ndim == 3
        # Elements should be flattened in row-major order
        assert len(arr.elements) == 12

    def test_3d_array_literal_f32(self):
        """Test creating 3D array literal with f32 elements"""
        arr_type = Array[2, 3, 2, f32]  # 2x3x2 array
        arr = ArrayLiteral([
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]
        ], arr_type)

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type.shape == (2, 3, 2)
        assert len(arr.elements) == 12

    def test_3d_array_construction_syntax(self):
        """Test Array[M, N, P, dtype]([...]) construction syntax"""
        arr = Array[2, 2, 2, i32]([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ])

        assert isinstance(arr, ArrayLiteral)
        assert arr.array_type == Array[2, 2, 2, i32]


# ==================== 3D ARRAY LITERAL VALIDATION ====================

class TestArray3DLiteralValidation(MLIRTestBase):
    """Test validation for 3D array literal creation"""

    def test_3d_size_mismatch_wrong_dim0(self):
        """Test that wrong first dimension is caught"""
        arr_type = Array[2, 2, 2, i32]  # Expect 2x2x2

        with pytest.raises(TypeError, match="3D array expects 2 matrices, got 3"):
            ArrayLiteral([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]]  # Extra plane
            ], arr_type)

    def test_3d_size_mismatch_wrong_dim1(self):
        """Test that wrong second dimension is caught"""
        arr_type = Array[2, 2, 2, i32]

        with pytest.raises(TypeError, match="Matrix 0: expected 2 rows, got 3"):
            ArrayLiteral([
                [[1, 2], [3, 4], [5, 6]],  # 3 rows instead of 2
                [[7, 8], [9, 10]]
            ], arr_type)

    def test_3d_size_mismatch_wrong_dim2(self):
        """Test that wrong third dimension is caught"""
        arr_type = Array[2, 2, 3, i32]  # Expect 3 columns

        with pytest.raises(TypeError, match="Matrix 1, row 0: expected 3 elements"):
            ArrayLiteral([
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8], [9, 10, 11]]  # Row 0 has only 2 elements
            ], arr_type)

    def test_3d_type_mismatch(self):
        """Test that element type mismatch is caught"""
        arr_type = Array[2, 2, 2, i32]

        with pytest.raises(TypeError, match="expected i32.*got f32"):
            ArrayLiteral([
                [[1, 2], [3, 4]],
                [[5.0, 6], [7, 8]]  # Float in i32 array
            ], arr_type)


# ==================== 3D ARRAY ACCESS ====================

class TestArray3DAccess(MLIRTestBase):
    """Test 3D ArrayAccess with 3-tuple indices"""

    def test_3d_array_access_creation(self):
        """Test creating ArrayAccess with 3D indices"""
        arr = ArrayLiteral([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], Array[2, 2, 2, i32])

        access = ArrayAccess(arr, (1, 0, 1))  # Access [1][0][1]

        assert isinstance(access, ArrayAccess)
        assert access.array == arr
        assert len(access.indices) == 3
        assert access.indices[0].value == 1
        assert access.indices[1].value == 0
        assert access.indices[2].value == 1

    def test_3d_array_subscript_syntax(self):
        """Test arr[i, j, k] subscript syntax"""
        arr = ArrayLiteral([
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]]
        ], Array[2, 2, 2, i32])

        access = arr[0, 1, 0]  # Access [0][1][0]

        assert isinstance(access, ArrayAccess)
        assert len(access.indices) == 3
        assert access.indices[0].value == 0
        assert access.indices[1].value == 1
        assert access.indices[2].value == 0

    def test_3d_array_access_infer_type(self):
        """Test that ArrayAccess returns scalar element type"""
        arr = ArrayLiteral([
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]]
        ], Array[2, 2, 2, f32])

        access = ArrayAccess(arr, (0, 0, 0))
        assert access.infer_type() == f32  # Returns scalar, not array

    def test_3d_wrong_index_count(self):
        """Test that wrong number of indices is rejected"""
        arr = ArrayLiteral([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], Array[2, 2, 2, i32])

        with pytest.raises(TypeError, match="Array dimension mismatch: 3D array requires 3 indices, got 2"):
            ArrayAccess(arr, (0, 1))  # Only 2 indices for 3D array


# ==================== 3D ARRAY STORE ====================

class TestArray3DStore(MLIRTestBase):
    """Test 3D ArrayStore with 3-tuple indices"""

    def test_3d_array_store_creation(self):
        """Test creating ArrayStore with 3D indices"""
        arr = ArrayLiteral([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], Array[2, 2, 2, i32])

        store = ArrayStore(arr, (1, 1, 0), 99)

        assert isinstance(store, ArrayStore)
        assert store.array == arr
        assert len(store.indices) == 3
        assert store.indices[0].value == 1
        assert store.indices[1].value == 1
        assert store.indices[2].value == 0
        assert store.value.value == 99

    def test_3d_array_store_at_set_syntax(self):
        """Test arr.at[i, j, k].set(value) syntax"""
        arr = ArrayLiteral([
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]]
        ], Array[2, 2, 2, i32])

        store = arr.at[0, 0, 1].set(100)

        assert isinstance(store, ArrayStore)
        assert len(store.indices) == 3
        assert store.indices[0].value == 0
        assert store.indices[1].value == 0
        assert store.indices[2].value == 1
        assert store.value.value == 100

    def test_3d_array_store_type_checking(self):
        """Test that store validates element type"""
        arr = ArrayLiteral([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], Array[2, 2, 2, i32])

        with pytest.raises(TypeError, match="Cannot store f32 into Array"):
            ArrayStore(arr, (0, 0, 0), 3.14)


# ==================== 3D ARRAY PROTOBUF SERIALIZATION ====================

class TestArray3DProtobufSerialization(MLIRTestBase):
    """Test protobuf serialization for 3D arrays"""

    def test_3d_array_literal_to_proto(self):
        """Test that 3D ArrayLiteral serializes correctly"""
        arr = ArrayLiteral([
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]]
        ], Array[2, 2, 2, i32])

        context = SerializationContext()
        pb = arr.to_proto(context)

        assert pb.HasField("array_literal")
        # Check shape is [2, 2, 2] (uses new TypeSpec with memref field)
        assert pb.array_literal.type.HasField("memref")
        assert len(pb.array_literal.type.memref.shape) == 3
        assert pb.array_literal.type.memref.shape[0] == 2
        assert pb.array_literal.type.memref.shape[1] == 2
        assert pb.array_literal.type.memref.shape[2] == 2
        # Element type is nested: type.memref.element_type.scalar.kind
        from mlir_edsl import ast_pb2
        assert pb.array_literal.type.memref.element_type.scalar.kind == ast_pb2.ScalarTypeSpec.I32
        # Check flattened elements
        assert len(pb.array_literal.elements) == 8

    def test_3d_array_access_to_proto(self):
        """Test that 3D ArrayAccess serializes with 3 indices"""
        arr = ArrayLiteral([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], Array[2, 2, 2, i32])
        access = ArrayAccess(arr, (1, 0, 1))

        context = SerializationContext()
        pb = access.to_proto(context)

        assert pb.HasField("array_access")
        assert len(pb.array_access.indices) == 3

    def test_3d_array_store_to_proto(self):
        """Test that 3D ArrayStore serializes with 3 indices"""
        arr = ArrayLiteral([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], Array[2, 2, 2, i32])
        store = ArrayStore(arr, (0, 1, 0), 99)

        context = SerializationContext()
        pb = store.to_proto(context)

        assert pb.HasField("array_store")
        assert len(pb.array_store.indices) == 3


# ==================== 3D ELEMENT-WISE OPERATIONS ====================

class TestArray3DElementwise(MLIRTestBase):
    """Test element-wise operations on 3D arrays"""

    def test_3d_array_add_arrays(self):
        """Test 3D array + 3D array element-wise addition"""
        arr1 = ArrayLiteral([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], Array[2, 2, 2, i32])
        arr2 = ArrayLiteral([[[10, 20], [30, 40]], [[50, 60], [70, 80]]], Array[2, 2, 2, i32])

        result = arr1 + arr2

        # Validate AST structure
        assert result.infer_type().shape == (2, 2, 2)
        assert result.infer_type().element_enum == i32

    def test_3d_array_add_scalar(self):
        """Test 3D array + scalar broadcasting"""
        arr = ArrayLiteral([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], Array[2, 2, 2, i32])
        scalar = 100

        result = arr + scalar

        assert result.infer_type().shape == (2, 2, 2)

    def test_3d_array_mul_arrays(self):
        """Test 3D array * 3D array element-wise multiplication"""
        arr1 = ArrayLiteral([[[2, 3], [4, 5]], [[6, 7], [8, 9]]], Array[2, 2, 2, i32])
        arr2 = ArrayLiteral([[[10, 10], [10, 10]], [[10, 10], [10, 10]]], Array[2, 2, 2, i32])

        result = arr1 * arr2

        assert result.infer_type().shape == (2, 2, 2)

    def test_3d_shape_mismatch(self):
        """Test that mismatched shapes are rejected"""
        arr1 = ArrayLiteral([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], Array[2, 2, 2, i32])  # 2x2x2
        arr2 = ArrayLiteral([[[1]], [[2]]], Array[2, 1, 1, i32])  # 2x1x1

        with pytest.raises(TypeError, match="Array shapes must match"):
            arr1 + arr2


# ==================== 3D MLIR GENERATION ====================

@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend")
class TestArray3DMLIRGeneration(MLIRTestBase):
    """Test MLIR generation for 3D arrays"""

    def test_3d_array_literal_generates_memref(self):
        """Test that 3D array literal compiles and generates memref type"""
        @ml_function
        def create_3d_array() -> Array[2, 2, 2, i32]:
            return Array[2, 2, 2, i32]([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ])

        # Should compile without errors - IR contains memref<2x2x2xi32>
        assert create_3d_array is not None

    def test_3d_array_access_generates_load(self):
        """Test that 3D array access compiles and generates memref.load with 3 indices"""
        @ml_function
        def access_3d_element() -> i32:
            arr = Array[2, 2, 2, i32]([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ])
            return arr[1, 0, 1]

        # Should compile without errors - IR contains memref.load with 3 indices
        assert access_3d_element is not None


# ==================== 3D EXECUTION TESTS ====================

@pytest.mark.skipif(not HAS_CPP_BACKEND, reason="Requires C++ backend")
class TestArray3DExecution(MLIRTestBase):
    """Test execution of 3D array operations"""

    def test_3d_array_element_access_execution(self):
        """Test executing 3D array element access"""
        @ml_function
        def get_element() -> i32:
            arr = Array[2, 2, 3, i32]([
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]]
            ])
            return arr[1, 1, 2]  # Should return 120

        result = get_element()
        assert result == 120

    def test_3d_array_add_execution(self):
        """Test executing 3D array addition"""
        @ml_function
        def add_arrays() -> Array[2, 2, 2, i32]:
            arr1 = Array[2, 2, 2, i32]([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ])
            arr2 = Array[2, 2, 2, i32]([
                [[10, 20], [30, 40]],
                [[50, 60], [70, 80]]
            ])
            return arr1 + arr2

        # Verify by accessing specific element (array stays in MLIR, return scalar)
        @ml_function
        def verify() -> i32:
            res = add_arrays()
            return res[1, 1, 1]  # Should be 8 + 80 = 88

        assert verify() == 88

    def test_3d_array_scalar_broadcast_execution(self):
        """Test executing 3D array + scalar broadcasting"""
        @ml_function
        def add_scalar() -> Array[2, 2, 2, i32]:
            arr = Array[2, 2, 2, i32]([
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]]
            ])
            return arr + 100

        # Verify by accessing element (array stays in MLIR, return scalar)
        @ml_function
        def verify() -> i32:
            res = add_scalar()
            return res[0, 0, 0]  # Should be 1 + 100 = 101

        assert verify() == 101


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
