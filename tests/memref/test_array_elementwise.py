"""Test element-wise array operations with broadcasting

This test validates array arithmetic operations:
- Array + Array (element-wise)
- Array + Scalar (broadcasting)
- Scalar + Array (broadcasting)
- All four operations: add, sub, mul, div
- Type checking and error cases
"""

import pytest
from mlir_edsl import ml_function, Array, i32, f32, cast


# ==================== ARRAY-ARRAY OPERATIONS ====================

class TestArrayArrayOps:
    """Test element-wise operations on two arrays"""

    def test_array_add_array(self, backend):
        """Test element-wise array addition"""
        @ml_function
        def array_add() -> i32:
            arr1 = Array[i32, 4]([1, 2, 3, 4])
            arr2 = Array[i32, 4]([10, 20, 30, 40])
            result = arr1 + arr2
            return result[2]  # Should be 33

        assert array_add() == 33

    def test_array_sub_array(self, backend):
        """Test element-wise array subtraction"""
        @ml_function
        def array_sub() -> i32:
            arr1 = Array[i32, 3]([100, 50, 25])
            arr2 = Array[i32, 3]([10, 20, 5])
            result = arr1 - arr2
            return result[1]  # Should be 30

        assert array_sub() == 30

    def test_array_mul_array(self, backend):
        """Test element-wise array multiplication"""
        @ml_function
        def array_mul() -> i32:
            arr1 = Array[i32, 3]([2, 3, 4])
            arr2 = Array[i32, 3]([5, 6, 7])
            result = arr1 * arr2
            return result[1]  # Should be 18

        assert array_mul() == 18

    def test_array_div_array(self, backend):
        """Test element-wise array division"""
        @ml_function
        def array_div() -> i32:
            arr1 = Array[i32, 3]([100, 50, 20])
            arr2 = Array[i32, 3]([10, 5, 2])
            result = arr1 / arr2
            return result[0]  # Should be 10

        assert array_div() == 10

    def test_array_shape_mismatch_error(self):
        """Test that mismatched array shapes raise TypeError"""
        @ml_function
        def bad_add() -> i32:
            arr1 = Array[i32, 3]([1, 2, 3])
            arr2 = Array[i32, 4]([1, 2, 3, 4])
            return (arr1 + arr2)[0]

        with pytest.raises(TypeError, match="shapes must match"):
            bad_add()

    def test_array_type_mismatch_error(self):
        """Test that mismatched element types raise TypeError"""
        @ml_function
        def bad_types() -> i32:
            arr1 = Array[i32, 3]([1, 2, 3])
            arr2 = Array[f32, 3]([1.0, 2.0, 3.0])
            return (arr1 + arr2)[0]

        with pytest.raises(TypeError, match="element types must match"):
            bad_types()


# ==================== ARRAY-SCALAR OPERATIONS ====================

class TestArrayScalarOps:
    """Test broadcasting with scalar values (array op scalar)"""

    def test_array_add_scalar(self, backend):
        """Test array + scalar broadcasting"""
        @ml_function
        def array_add_scalar() -> i32:
            arr = Array[i32, 4]([10, 20, 30, 40])
            result = arr + 5
            return result[1]  # Should be 25

        assert array_add_scalar() == 25

    def test_array_sub_scalar(self, backend):
        """Test array - scalar broadcasting"""
        @ml_function
        def array_sub_scalar() -> i32:
            arr = Array[i32, 3]([100, 50, 25])
            result = arr - 10
            return result[2]  # Should be 15

        assert array_sub_scalar() == 15

    def test_array_mul_scalar(self, backend):
        """Test array * scalar broadcasting"""
        @ml_function
        def array_mul_scalar() -> i32:
            arr = Array[i32, 3]([2, 3, 4])
            result = arr * 5
            return result[0]  # Should be 10

        assert array_mul_scalar() == 10

    def test_array_div_scalar(self, backend):
        """Test array / scalar broadcasting"""
        @ml_function
        def array_div_scalar() -> i32:
            arr = Array[i32, 4]([100, 50, 20, 10])
            result = arr / 10
            return result[1]  # Should be 5

        assert array_div_scalar() == 5

    def test_array_scalar_type_mismatch_error(self):
        """Test that scalar type must match array element type"""
        @ml_function
        def bad_types() -> i32:
            arr = Array[i32, 3]([1, 2, 3])
            result = arr + 2.5  # i32 array + f32 scalar
            return result[0]

        with pytest.raises(TypeError, match="Scalar type must match"):
            bad_types()


# ==================== SCALAR-ARRAY OPERATIONS ====================

class TestScalarArrayOps:
    """Test broadcasting with scalar values (scalar op array)"""

    def test_scalar_add_array(self, backend):
        """Test scalar + array broadcasting"""
        @ml_function
        def scalar_add_array() -> i32:
            arr = Array[i32, 3]([10, 20, 30])
            result = 5 + arr
            return result[2]  # Should be 35

        assert scalar_add_array() == 35

    def test_scalar_sub_array(self, backend):
        """Test scalar - array broadcasting"""
        @ml_function
        def scalar_sub_array() -> i32:
            arr = Array[i32, 3]([10, 20, 30])
            result = 100 - arr
            return result[1]  # Should be 80

        assert scalar_sub_array() == 80

    def test_scalar_mul_array(self, backend):
        """Test scalar * array broadcasting"""
        @ml_function
        def scalar_mul_array() -> i32:
            arr = Array[i32, 3]([2, 3, 4])
            result = 10 * arr
            return result[2]  # Should be 40

        assert scalar_mul_array() == 40

    def test_scalar_div_array(self, backend):
        """Test scalar / array broadcasting"""
        @ml_function
        def scalar_div_array() -> i32:
            arr = Array[i32, 3]([1, 2, 5])
            result = 100 / arr
            return result[1]  # Should be 50

        assert scalar_div_array() == 50

    def test_scalar_array_type_mismatch_error(self):
        """Test that scalar type must match array element type"""
        @ml_function
        def bad_types() -> f32:
            arr = Array[f32, 3]([1.0, 2.0, 3.0])
            result = 5 + arr  # i32 scalar + f32 array
            return result[0]

        with pytest.raises(TypeError, match="Scalar type must match"):
            bad_types()


# ==================== FLOAT ARRAY OPERATIONS ====================

class TestFloatArrayOps:
    """Test element-wise operations with float arrays"""

    def test_float_array_add_array(self, backend):
        """Test float array element-wise addition"""
        @ml_function
        def float_add() -> f32:
            arr1 = Array[f32, 3]([1.0, 2.0, 3.0])
            arr2 = Array[f32, 3]([0.5, 1.5, 2.5])
            result = arr1 + arr2
            return result[1]  # Should be 3.5

        result = float_add()
        assert abs(result - 3.5) < 0.001

    def test_float_array_mul_scalar(self, backend):
        """Test float array with scalar broadcasting"""
        @ml_function
        def float_broadcast() -> f32:
            arr = Array[f32, 3]([1.0, 2.0, 3.0])
            result = arr * 2.5
            return result[1]  # Should be 5.0

        result = float_broadcast()
        assert abs(result - 5.0) < 0.001

    def test_float_scalar_div_array(self, backend):
        """Test float scalar / array broadcasting"""
        @ml_function
        def float_div() -> f32:
            arr = Array[f32, 3]([2.0, 4.0, 5.0])
            result = 10.0 / arr
            return result[0]  # Should be 5.0

        result = float_div()
        assert abs(result - 5.0) < 0.001


# ==================== COMPLEX EXPRESSIONS ====================

class TestComplexExpressions:
    """Test chained and nested array operations"""

    def test_chained_operations(self, backend):
        """Test multiple operations: (arr1 + arr2) * scalar"""
        @ml_function
        def chained() -> i32:
            arr1 = Array[i32, 3]([1, 2, 3])
            arr2 = Array[i32, 3]([4, 5, 6])
            result = (arr1 + arr2) * 2
            return result[2]  # (3 + 6) * 2 = 18

        assert chained() == 18

    def test_mixed_operations(self, backend):
        """Test mixing array and scalar operations"""
        @ml_function
        def mixed() -> i32:
            arr = Array[i32, 4]([10, 20, 30, 40])
            result = (arr + 5) * 2 - 10
            x = result[0]  # (10 + 5) * 2 - 10 = 20
            y = result[3]  # (40 + 5) * 2 - 10 = 80
            return x + y   # 100

        assert mixed() == 100

    def test_array_operations_with_access(self, backend):
        """Test array operations combined with element access"""
        @ml_function
        def complex_expr() -> i32:
            arr1 = Array[i32, 4]([1, 2, 3, 4])
            arr2 = Array[i32, 4]([10, 20, 30, 40])
            sum_arr = arr1 + arr2
            prod_arr = arr1 * arr2
            return sum_arr[1] + prod_arr[2]  # 22 + 90 = 112

        assert complex_expr() == 112

    def test_nested_broadcast(self, backend):
        """Test nested operations with broadcasting"""
        @ml_function
        def nested() -> i32:
            arr = Array[i32, 3]([5, 10, 15])
            # ((arr * 2) + 10) / 5
            result = ((arr * 2) + 10) / 5
            return result[1]  # ((10 * 2) + 10) / 5 = 6

        assert nested() == 6


# ==================== INTEGRATION WITH EXISTING FEATURES ====================

class TestIntegrationWithExisting:
    """Test element-wise operations work with other features"""

    def test_array_ops_with_store(self, backend):
        """Test element-wise ops combined with array store"""
        @ml_function
        def with_store() -> i32:
            arr1 = Array[i32, 3]([1, 2, 3])
            arr2 = Array[i32, 3]([10, 20, 30])
            result = arr1 + arr2
            # Modify one element
            result = result.at[1].set(99)
            return result[1]  # Should be 99

        assert with_store() == 99

    def test_array_ops_all_four_operations(self, backend):
        """Test all four operations in one function"""
        @ml_function
        def all_ops() -> i32:
            arr = Array[i32, 4]([100, 50, 20, 10])
            add_result = arr + 10
            sub_result = arr - 10
            mul_result = arr * 2
            div_result = arr / 10
            # Sum selected results
            return add_result[0] + sub_result[1] + mul_result[2] + div_result[3]
            # 110 + 40 + 40 + 1 = 191

        assert all_ops() == 191
