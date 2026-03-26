"""Test tensor_map, relu, and leaky_relu execution end-to-end (Phase 8.3)

Validates that element-wise map compiles through the full pipeline:
  Python AST → Protobuf → C++ LinalgBuilder → linalg.generic IR
  → convert-linalg-to-loops → scf.for → LLVM IR → JIT execution
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, tensor_map, relu, leaky_relu


# ==================== EXECUTION TESTS ====================

class TestMapExecution:
    """Test tensor_map execution with various bodies"""

    def test_identity_map(self, backend):
        """tensor_map with identity body returns a copy of the input."""
        @ml_function
        def identity(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda v: v)

        result = identity(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert len(result) == 4
        for got, expected in zip(result, [1.0, 2.0, 3.0, 4.0]):
            assert abs(got - expected) < 1e-5

    def test_scale_map(self, backend):
        """tensor_map scales each element by a constant factor."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda v: v * 2.0)

        result = scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        assert len(result) == 4
        for got, expected in zip(result, [2.0, 4.0, 6.0, 8.0]):
            assert abs(got - expected) < 1e-5

    def test_manual_relu_body(self, backend):
        """tensor_map with explicit if-then-else relu body."""
        from mlir_edsl import If
        from mlir_edsl.ast.helpers import to_value

        @ml_function
        def manual_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda v: If(v > to_value(0.0), v, to_value(0.0)))

        result = manual_relu(np.array([-1.0, 0.0, 2.0, -3.0], dtype=np.float32))
        assert len(result) == 4
        for got, expected in zip(result, [0.0, 0.0, 2.0, 0.0]):
            assert abs(got - expected) < 1e-5

    def test_relu_all_positive(self, backend):
        """relu on all-positive input is identity."""
        @ml_function
        def apply_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return relu(a)

        result = apply_relu(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        for got, expected in zip(result, [1.0, 2.0, 3.0, 4.0]):
            assert abs(got - expected) < 1e-5

    def test_relu_all_negative(self, backend):
        """relu on all-negative input is all zeros."""
        @ml_function
        def apply_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return relu(a)

        result = apply_relu(np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float32))
        for got in result:
            assert abs(got) < 1e-5

    def test_relu_mixed(self, backend):
        """relu clamps negatives to zero and passes positives through."""
        @ml_function
        def apply_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return relu(a)

        result = apply_relu(np.array([-1.0, 2.0, -3.0, 4.0], dtype=np.float32))
        for got, expected in zip(result, [0.0, 2.0, 0.0, 4.0]):
            assert abs(got - expected) < 1e-5

    def test_leaky_relu_positive_passthrough(self, backend):
        """leaky_relu passes positive values through unchanged."""
        @ml_function
        def apply_leaky(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return leaky_relu(a, alpha=0.1)

        result = apply_leaky(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        for got, expected in zip(result, [1.0, 2.0, 3.0, 4.0]):
            assert abs(got - expected) < 1e-5

    def test_leaky_relu_negative_scaling(self, backend):
        """leaky_relu scales negative values by alpha."""
        @ml_function
        def apply_leaky(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return leaky_relu(a, alpha=0.1)

        result = apply_leaky(np.array([-1.0, -2.0, -10.0, -0.5], dtype=np.float32))
        expected = [-0.1, -0.2, -1.0, -0.05]
        for got, exp in zip(result, expected):
            assert abs(got - exp) < 1e-5

    def test_single_element_array(self, backend):
        """tensor_map works on a 1-element array."""
        @ml_function
        def scale_one(a: Tensor[f32, 1]) -> Tensor[f32, 1]:
            return tensor_map(a, lambda v: v * 3.0)

        result = scale_one(np.array([5.0], dtype=np.float32))
        assert abs(result[0] - 15.0) < 1e-5


# ==================== CHAINED MAP ====================

class TestMapChained:
    """Test tensor_map used as a sub-expression (no outParam — tensor.empty path)"""

    def test_map_chained(self, backend):
        """tensor_map(tensor_map(a, f), g) exercises the tensor.empty fallback for the inner map"""
        @ml_function
        def chained(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(tensor_map(a, lambda v: v * 2.0), lambda v: v + 1.0)

        result = chained(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        np.testing.assert_allclose(result, [3.0, 5.0, 7.0, 9.0], rtol=1e-5)


# ==================== TYPE VALIDATION TESTS ====================

class TestMapTypeValidation:
    """Test that invalid inputs are rejected at Python AST construction time"""

    def test_2d_tensor_accepted(self):
        """tensor_map supports ND tensors, not just 1D."""
        from mlir_edsl.ast.nodes.functions import Parameter
        from mlir_edsl.types import TensorType, f32 as f32_type

        arr_2d = Parameter("a", TensorType((2, 2), f32_type))
        result = tensor_map(arr_2d, lambda v: v)
        assert result is not None

    def test_scalar_rejected(self):
        """tensor_map requires a tensor, not a scalar."""
        from mlir_edsl.ast.nodes.scalars import Constant
        from mlir_edsl.types import f32 as f32_type

        scalar = Constant(1.0, f32_type)
        with pytest.raises(TypeError, match="tensor"):
            tensor_map(scalar, lambda v: v)

    def test_body_type_mismatch_rejected(self):
        """tensor_map body must return the same element type as the input."""
        from mlir_edsl.ast.nodes.functions import Parameter
        from mlir_edsl.types import TensorType, f32 as f32_type
        from mlir_edsl.ast.helpers import to_value

        arr = Parameter("a", TensorType(3, f32_type))
        with pytest.raises(TypeError, match="element type"):
            tensor_map(arr, lambda v: to_value(1))
