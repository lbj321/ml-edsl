"""Test tensor types as function parameters (Phase 7.4)

Tensor parameters go through the bufferization boundary:
  tensor<4xf32> param → one-shot-bufferize → memref<4xf32> → LLVM descriptor
The calling convention on the Python side is identical to Array parameters.

Function names are prefixed with tp_ to avoid collisions with the shared
global backend (test_array_params.py uses unprefixed equivalents).
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, For, f32, i32


class TestTensorParameters:
    """Phase 7.4: tensors passed as function arguments"""

    def test_single_tensor_param_first_element(self, backend):
        """Access first element of a tensor parameter"""
        @ml_function
        def tp_first_elem(t: Tensor[f32, 4]) -> f32:
            return t[0]

        assert abs(tp_first_elem(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)) - 1.0) < 0.001

    def test_single_tensor_param_middle_element(self, backend):
        """Access non-zero index of a tensor parameter"""
        @ml_function
        def tp_third_elem(t: Tensor[i32, 4]) -> i32:
            return t[2]

        assert tp_third_elem(np.array([10, 20, 30, 40], dtype=np.int32)) == 30

    def test_two_tensor_params_scalar_result(self, backend):
        """Sum first elements of two tensor parameters"""
        @ml_function
        def tp_add_firsts(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return a[0] + b[0]

        result = tp_add_firsts(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                               np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32))
        assert abs(result - 11.0) < 0.001

    def test_tensor_dot_product(self, backend):
        """Dot product over tensor parameters via lambda For"""
        @ml_function
        def tp_dot(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return For(start=0, end=4, init=0.0,
                       body=lambda i, acc: acc + a[i] * b[i])

        result = tp_dot(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                        np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        assert abs(result - 10.0) < 0.001

    def test_mixed_tensor_and_scalar_params(self, backend):
        """Function with both tensor and scalar parameters"""
        @ml_function
        def tp_scaled_elem(t: Tensor[f32, 4], idx: i32, scale: f32) -> f32:
            return t[idx] * scale

        result = tp_scaled_elem(np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32), 1, 3.0)
        assert abs(result - 12.0) < 0.001

    def test_tensor_param_wrong_length_raises(self, backend):
        """Wrong-length list is rejected at call time"""
        @ml_function
        def tp_takes_four(t: Tensor[f32, 4]) -> f32:
            return t[0]

        with pytest.raises(ValueError, match="4"):
            tp_takes_four(np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_tensor_param_non_list_raises(self, backend):
        """Non-list input is rejected"""
        @ml_function
        def tp_takes_tensor(t: Tensor[f32, 4]) -> f32:
            return t[0]

        with pytest.raises(TypeError, match="ndarray"):
            tp_takes_tensor((1.0, 2.0, 3.0, 4.0))
