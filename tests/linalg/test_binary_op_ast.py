"""AST construction tests for LinalgBinaryOp

Validates broadcast mode inference, output type computation,
and error cases — all at Python AST construction time (no backend needed).
"""

import pytest
from mlir_edsl.ast.nodes.linalg import LinalgBinaryOp
from mlir_edsl.ast.nodes.functions import Parameter
from mlir_edsl.ast.nodes.scalars import Constant
from mlir_edsl.types import TensorType, f32, i32
from mlir_edsl import ast_pb2


# ==================== BROADCAST MODE INFERENCE ====================

class TestLinalgBinaryOpBroadcastMode:
    """Verify that the correct BroadcastMode is inferred from operand shapes."""

    def test_same_shape_1d_is_none(self):
        """Two 1D tensors of the same shape → NONE."""
        a = Parameter("a", TensorType((4,), f32))
        b = Parameter("b", TensorType((4,), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, a, b)
        assert node._broadcast_mode == ast_pb2.NONE

    def test_same_shape_2d_is_none(self):
        """Two 2D tensors of the same shape → NONE."""
        a = Parameter("a", TensorType((3, 4), f32))
        b = Parameter("b", TensorType((3, 4), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, a, b)
        assert node._broadcast_mode == ast_pb2.NONE

    def test_2d_plus_1d_is_bias_right(self):
        """Tensor[M,N] + Tensor[N] → TENSOR_BIAS_RIGHT."""
        matrix = Parameter("m", TensorType((4, 8), f32))
        bias   = Parameter("b", TensorType((8,), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, matrix, bias)
        assert node._broadcast_mode == ast_pb2.TENSOR_BIAS_RIGHT

    def test_1d_plus_2d_is_bias_left(self):
        """Tensor[N] + Tensor[M,N] → TENSOR_BIAS_LEFT."""
        bias   = Parameter("b", TensorType((8,), f32))
        matrix = Parameter("m", TensorType((4, 8), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, bias, matrix)
        assert node._broadcast_mode == ast_pb2.TENSOR_BIAS_LEFT

    def test_tensor_plus_scalar_is_scalar_right(self):
        """Tensor[N] + scalar → SCALAR_RIGHT."""
        a = Parameter("a", TensorType((4,), f32))
        s = Constant(2.0, f32)
        node = LinalgBinaryOp(ast_pb2.MUL, a, s)
        assert node._broadcast_mode == ast_pb2.SCALAR_RIGHT

    def test_scalar_plus_tensor_is_scalar_left(self):
        """scalar + Tensor[N] → SCALAR_LEFT."""
        s = Constant(2.0, f32)
        a = Parameter("a", TensorType((4,), f32))
        node = LinalgBinaryOp(ast_pb2.MUL, s, a)
        assert node._broadcast_mode == ast_pb2.SCALAR_LEFT


# ==================== OUTPUT TYPE INFERENCE ====================

class TestLinalgBinaryOpOutputType:
    """Verify infer_type() returns the correct TensorType."""

    def test_same_shape_output_matches_input(self):
        """Same-shape add: output type == input type."""
        t = TensorType((3, 4), f32)
        a = Parameter("a", t)
        b = Parameter("b", t)
        node = LinalgBinaryOp(ast_pb2.ADD, a, b)
        assert node.infer_type() == t

    def test_bias_right_output_is_matrix_shape(self):
        """[M,N] + [N]: output shape is [M,N]."""
        matrix = Parameter("m", TensorType((4, 8), f32))
        bias   = Parameter("b", TensorType((8,), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, matrix, bias)
        assert node.infer_type() == TensorType((4, 8), f32)

    def test_bias_left_output_is_matrix_shape(self):
        """[N] + [M,N]: output shape is [M,N]."""
        bias   = Parameter("b", TensorType((8,), f32))
        matrix = Parameter("m", TensorType((4, 8), f32))
        node = LinalgBinaryOp(ast_pb2.ADD, bias, matrix)
        assert node.infer_type() == TensorType((4, 8), f32)

    def test_scalar_right_output_is_tensor_shape(self):
        """Tensor[N] * scalar: output shape is Tensor[N]."""
        a = Parameter("a", TensorType((4,), f32))
        s = Constant(2.0, f32)
        node = LinalgBinaryOp(ast_pb2.MUL, a, s)
        assert node.infer_type() == TensorType((4,), f32)


# ==================== ERROR CASES ====================

class TestLinalgBinaryOpErrors:
    """Verify type mismatches and invalid shapes are rejected at construction time."""

    def test_element_type_mismatch_same_shape(self):
        """f32 + i32 on same-shape tensors raises TypeError."""
        a = Parameter("a", TensorType((4,), f32))
        b = Parameter("b", TensorType((4,), i32))
        with pytest.raises(TypeError, match="element types must match"):
            LinalgBinaryOp(ast_pb2.ADD, a, b)

    def test_element_type_mismatch_bias(self):
        """[M,N] f32 + [N] i32 raises TypeError."""
        matrix = Parameter("m", TensorType((4, 8), f32))
        bias   = Parameter("b", TensorType((8,), i32))
        with pytest.raises(TypeError, match="element types must match"):
            LinalgBinaryOp(ast_pb2.ADD, matrix, bias)

    def test_inner_dim_mismatch_bias(self):
        """[M,N] + [K] where K != N raises TypeError."""
        matrix = Parameter("m", TensorType((4, 8), f32))
        bias   = Parameter("b", TensorType((5,), f32))
        with pytest.raises(TypeError, match="trailing dims must match"):
            LinalgBinaryOp(ast_pb2.ADD, matrix, bias)

    def test_incompatible_shapes_raises(self):
        """[3,4] + [2,5] raises TypeError (incompatible same-shape)."""
        a = Parameter("a", TensorType((3, 4), f32))
        b = Parameter("b", TensorType((2, 5), f32))
        with pytest.raises(TypeError, match="unsupported tensor shapes"):
            LinalgBinaryOp(ast_pb2.ADD, a, b)

    def test_scalar_type_mismatch_raises(self):
        """Tensor[f32] + i32 scalar raises TypeError."""
        a = Parameter("a", TensorType((4,), f32))
        s = Constant(2, i32)
        with pytest.raises(TypeError, match="must match"):
            LinalgBinaryOp(ast_pb2.ADD, a, s)

    def test_two_scalars_raises(self):
        """Two scalars should not go through LinalgBinaryOp."""
        s1 = Constant(1.0, f32)
        s2 = Constant(2.0, f32)
        with pytest.raises(TypeError, match="at least one tensor"):
            LinalgBinaryOp(ast_pb2.ADD, s1, s2)
