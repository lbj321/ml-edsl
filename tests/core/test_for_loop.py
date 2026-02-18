"""Tests for For loop with lambda body (scf.for with iter_args)

Test suite covers:
- ForIndex/ForIterArg placeholder AST nodes
- ForLoopOp construction with lambda body
- Type checking and validation
- Protobuf serialization
- End-to-end execution with scalar accumulators
- End-to-end execution with tensor accumulators (loop-based tensor init)
"""

import pytest
from mlir_edsl import ml_function, i32, f32, Tensor, For
from mlir_edsl.ast import ForLoopOp, Constant
from mlir_edsl.ast.nodes.control_flow import ForIndex, ForIterArg
from mlir_edsl.ast.serialization import SerializationContext
from mlir_edsl.types import TensorType


# ==================== PLACEHOLDER AST NODES ====================

class TestForIndex:
    """Test ForIndex placeholder node"""

    def test_for_index_type_is_i32(self):
        """ForIndex induction variable is always i32"""
        idx = ForIndex()
        assert idx.infer_type() == i32

    def test_for_index_is_leaf(self):
        """ForIndex has no children"""
        idx = ForIndex()
        assert idx.get_children() == []

    def test_for_index_has_unique_id(self):
        """Each ForIndex gets a unique node_id"""
        a = ForIndex()
        b = ForIndex()
        assert a.node_id != b.node_id

    def test_for_index_serializes(self):
        """ForIndex serializes to control_flow.for_index"""
        idx = ForIndex()
        context = SerializationContext()
        pb = idx.to_proto(context)
        assert pb.HasField("control_flow")
        assert pb.control_flow.HasField("for_index")
        assert pb.control_flow.for_index.node_id == idx.node_id


class TestForIterArg:
    """Test ForIterArg placeholder node"""

    def test_for_iter_arg_type_matches_init(self):
        """ForIterArg type matches the declared value_type"""
        arg = ForIterArg(i32)
        assert arg.infer_type() == i32

    def test_for_iter_arg_tensor_type(self):
        """ForIterArg can have tensor type"""
        tensor_type = TensorType(4, i32)
        arg = ForIterArg(tensor_type)
        assert arg.infer_type() == tensor_type

    def test_for_iter_arg_is_leaf(self):
        """ForIterArg has no children"""
        arg = ForIterArg(i32)
        assert arg.get_children() == []

    def test_for_iter_arg_serializes(self):
        """ForIterArg serializes to control_flow.for_iter_arg"""
        arg = ForIterArg(f32, arg_index=0)
        context = SerializationContext()
        pb = arg.to_proto(context)
        assert pb.HasField("control_flow")
        assert pb.control_flow.HasField("for_iter_arg")
        assert pb.control_flow.for_iter_arg.node_id == arg.node_id
        assert pb.control_flow.for_iter_arg.arg_index == 0


# ==================== FOR LOOP CONSTRUCTION ====================

class TestForLoopOpConstruction:
    """Test ForLoopOp AST node construction with lambda body"""

    def test_for_loop_scalar_body(self):
        """Test constructing a for loop with scalar accumulator"""
        result = For(0, 10, init=0, body=lambda i, acc: acc + i)
        assert isinstance(result, ForLoopOp)
        assert result.infer_type() == i32

    def test_for_loop_tensor_body(self):
        """Test constructing a for loop with tensor accumulator"""
        t = Tensor[i32, 4]([0, 0, 0, 0])
        result = For(0, 4, init=t, body=lambda i, acc: acc.at[i].set(i * 2))
        assert isinstance(result, ForLoopOp)
        assert isinstance(result.infer_type(), TensorType)

    def test_for_loop_body_type_mismatch_rejected(self):
        """Body return type must match init type"""
        with pytest.raises(TypeError, match="does not match"):
            For(0, 10, init=0, body=lambda i, acc: Constant(1.0, f32))

    def test_for_loop_float_bounds_rejected(self):
        """Loop bounds must be integer"""
        with pytest.raises(TypeError, match="integer loop bounds"):
            For(Constant(0.0, f32), Constant(10.0, f32),
                init=0, body=lambda i, acc: acc + i,
                step=Constant(1.0, f32))

    def test_for_loop_get_children(self):
        """ForLoopOp.get_children includes body"""
        result = For(0, 10, init=0, body=lambda i, acc: acc + i)
        children = result.get_children()
        assert len(children) == 5  # start, end, step, init, body


# ==================== PROTOBUF SERIALIZATION ====================

class TestForLoopSerialization:
    """Test protobuf serialization of ForLoopOp"""

    def test_for_loop_serializes(self):
        """ForLoopOp serializes with body and loop_id"""
        result = For(0, 4, init=0, body=lambda i, acc: acc + i)
        context = SerializationContext()
        context.count_uses(result)
        pb = result.to_proto(context)

        assert pb.HasField("control_flow")
        fl = pb.control_flow.for_loop
        assert fl.HasField("start")
        assert fl.HasField("end")
        assert fl.HasField("step")
        assert fl.HasField("init_value")
        assert fl.HasField("body")
        assert fl.loop_id == result.id
        assert fl.HasField("result_type")


# ==================== SCALAR ACCUMULATOR EXECUTION ====================

class TestForLoopScalarExecution:
    """Test for loop execution with scalar accumulators"""

    def test_sum_0_to_9(self, backend):
        """Sum integers 0..9"""
        @ml_function
        def sum_loop() -> i32:
            return For(0, 10, init=0, body=lambda i, acc: acc + i)

        assert sum_loop() == 45  # 0+1+2+...+9

    def test_sum_with_step(self, backend):
        """Sum even numbers 0, 2, 4, 6, 8"""
        @ml_function
        def sum_evens() -> i32:
            return For(0, 10, init=0, body=lambda i, acc: acc + i, step=2)

        assert sum_evens() == 20  # 0+2+4+6+8

    def test_product(self, backend):
        """Factorial-like: 1*1*2*3*4"""
        @ml_function
        def product_loop() -> i32:
            return For(1, 5, init=1, body=lambda i, acc: acc * i)

        assert product_loop() == 24  # 1*2*3*4

    def test_parametric_sum(self, backend):
        """Sum with parametric bound"""
        @ml_function
        def param_sum(n: int) -> int:
            return For(0, n, init=0, body=lambda i, acc: acc + i)

        assert param_sum(5) == 10   # 0+1+2+3+4
        assert param_sum(10) == 45  # 0+1+2+...+9


# ==================== TENSOR ACCUMULATOR EXECUTION ====================

class TestForLoopTensorExecution:
    """Test for loop execution with tensor accumulators"""

    def test_tensor_fill_loop(self, backend):
        """Fill tensor with i*2 using loop"""
        @ml_function
        def fill_tensor() -> i32:
            t = Tensor.empty(i32, 4)
            t = For(0, 4, init=t, body=lambda i, acc: acc.at[i].set(i * 2))
            return t[0] + t[1] + t[2] + t[3]

        result = fill_tensor()
        assert result == 0 + 2 + 4 + 6  # 12

    def test_tensor_fill_read_single(self, backend):
        """Fill tensor and read single element"""
        @ml_function
        def fill_and_read() -> i32:
            t = Tensor.empty(i32, 4)
            t = For(0, 4, init=t, body=lambda i, acc: acc.at[i].set(i * 10))
            return t[2]

        assert fill_and_read() == 20

    def test_tensor_fill_constants(self, backend):
        """Fill tensor with constant value"""
        @ml_function
        def fill_constant() -> i32:
            t = Tensor.empty(i32, 3)
            t = For(0, 3, init=t, body=lambda i, acc: acc.at[i].set(42))
            return t[0] + t[1] + t[2]

        assert fill_constant() == 126  # 42*3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
