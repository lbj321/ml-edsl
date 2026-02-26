"""Test aggregate types as function parameters (Phase 7.4)"""

import pytest
from mlir_edsl import ml_function, Array, For, f32, i32


class TestArrayParameters:
    """Phase 7.4: arrays passed as function arguments"""

    def test_single_array_param_first_element(self, backend):
        """Access first element of an array parameter"""
        @ml_function
        def first_elem(arr: Array[i32, 4]) -> i32:
            return arr[0]

        assert first_elem([10, 20, 30, 40]) == 10

    def test_single_array_param_middle_element(self, backend):
        """Access non-zero index of array parameter"""
        @ml_function
        def third_elem(arr: Array[i32, 4]) -> i32:
            return arr[2]

        assert third_elem([10, 20, 30, 40]) == 30

    def test_two_array_params_scalar_result(self, backend):
        """Sum first elements of two array parameters"""
        @ml_function
        def add_firsts(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return a[0] + b[0]

        result = add_firsts([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
        assert abs(result - 11.0) < 0.001

    def test_dot_product(self, backend):
        """Dot product via lambda For — the primary Phase 7.4 example"""
        @ml_function
        def dot(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return For(start=0, end=4, init=0.0,
                       body=lambda i, acc: acc + a[i] * b[i])

        result = dot([1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0])
        assert abs(result - 10.0) < 0.001

    def test_dot_product_weighted(self, backend):
        """Dot product with non-trivial weights"""
        @ml_function
        def dot_weighted(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return For(start=0, end=4, init=0.0,
                       body=lambda i, acc: acc + a[i] * b[i])

        # 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        result = dot_weighted([1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0])
        assert abs(result - 40.0) < 0.001

    def test_mixed_scalar_and_array_params(self, backend):
        """Function with both array and scalar parameters"""
        @ml_function
        def scaled_first(arr: Array[f32, 4], scale: f32) -> f32:
            return arr[0] * scale

        result = scaled_first([3.0, 0.0, 0.0, 0.0], 5.0)
        assert abs(result - 15.0) < 0.001

    def test_array_param_wrong_length_raises(self, backend):
        """Wrong-length list is rejected at call time"""
        @ml_function
        def takes_four(arr: Array[i32, 4]) -> i32:
            return arr[0]

        with pytest.raises(ValueError, match="4"):
            takes_four([1, 2, 3])

    def test_array_param_non_list_raises(self, backend):
        """Non-list input is rejected"""
        @ml_function
        def takes_array(arr: Array[i32, 4]) -> i32:
            return arr[0]

        with pytest.raises(TypeError, match="list"):
            takes_array((1, 2, 3, 4))
