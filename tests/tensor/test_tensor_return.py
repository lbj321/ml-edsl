"""Test tensor return types (Phase 8.7)

Functions returning Tensor[dtype, shape] compile to a void function with a
hidden memref out-param, identical to the Array return calling convention.
At the Python boundary the caller gets a numpy array back.
"""

import numpy as np
import pytest
from mlir_edsl import ml_function, Tensor, f32, i32


class TestTensorReturnLiteral:
    """Return a tensor literal from a function body."""

    def test_return_f32_tensor_literal(self, backend):
        """Return a 1D f32 tensor literal"""
        @ml_function
        def tr_f32_literal() -> Tensor[f32, 4]:
            return Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])

        result = tr_f32_literal()
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0], rtol=1e-5)

    def test_return_i32_tensor_literal(self, backend):
        """Return a 1D i32 tensor literal"""
        @ml_function
        def tr_i32_literal() -> Tensor[i32, 3]:
            return Tensor[i32, 3]([10, 20, 30])

        result = tr_i32_literal()
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestTensorReturnPassthrough:
    """Return a tensor that was passed in as a parameter."""

    def test_tensor_param_passthrough(self, backend):
        """Pass-through: tensor parameter as return value"""
        @ml_function
        def tr_passthrough(t: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return t

        inp = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        result = tr_passthrough(inp)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_allclose(result, inp, rtol=1e-5)

    def test_i32_tensor_param_passthrough(self, backend):
        """Pass-through: i32 tensor parameter as return value"""
        @ml_function
        def tr_i32_passthrough(t: Tensor[i32, 3]) -> Tensor[i32, 3]:
            return t

        inp = np.array([100, 200, 300], dtype=np.int32)
        result = tr_i32_passthrough(inp)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, inp)


class TestTensorReturnInsertChain:
    """Return the result of a tensor.insert chain."""

    def test_return_tensor_after_insert(self, backend):
        """Return a tensor modified by insert operations"""
        @ml_function
        def tr_insert() -> Tensor[i32, 3]:
            t = Tensor[i32, 3]([0, 0, 0])
            t2 = t.at[0].set(10)
            t3 = t2.at[1].set(20)
            t4 = t3.at[2].set(30)
            return t4

        result = tr_insert()
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        np.testing.assert_array_equal(result, [10, 20, 30])


class TestTensorReturnMultipleParams:
    """Return a tensor when multiple tensor params are present."""

    def test_return_first_of_two_tensor_params(self, backend):
        """Two tensor params: return the first — exercises to_tensor for each param"""
        @ml_function
        def tr_first(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a

        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        result = tr_first(a, b)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, a, rtol=1e-5)

    def test_return_second_of_two_tensor_params(self, backend):
        """Two tensor params: return the second"""
        @ml_function
        def tr_second(a: Tensor[i32, 3], b: Tensor[i32, 3]) -> Tensor[i32, 3]:
            return b

        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([10, 20, 30], dtype=np.int32)
        result = tr_second(a, b)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, b)


class TestTensorReturnMixedParams:
    """Return a tensor from a function that also has scalar params."""

    def test_insert_scalar_into_tensor_param(self, backend):
        """Scalar params used to insert into a tensor param and return it"""
        @ml_function
        def tr_set_first(t: Tensor[i32, 4], val: int) -> Tensor[i32, 4]:
            return t.at[0].set(val)

        inp = np.array([0, 2, 3, 4], dtype=np.int32)
        result = tr_set_first(inp, 99)
        assert isinstance(result, np.ndarray)
        assert result.shape == (4,)
        np.testing.assert_array_equal(result, [99, 2, 3, 4])

    def test_return_tensor_ignoring_scalar_param(self, backend):
        """Scalar param present but tensor passed through unchanged"""
        @ml_function
        def tr_ignore_scalar(t: Tensor[f32, 3], _unused: float) -> Tensor[f32, 3]:
            return t

        inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = tr_ignore_scalar(inp, 42.0)
        assert isinstance(result, np.ndarray)
        np.testing.assert_allclose(result, inp, rtol=1e-5)


class TestTensorReturnIR:
    """IR structure tests for tensor return functions."""

    def test_tensor_return_has_to_tensor_for_param(self, check_ir):
        """Tensor param should appear as to_tensor in the function body"""
        @ml_function
        def tr_ir_param(t: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return t

        tr_ir_param(np.zeros(4, dtype=np.float32))

        check_ir("""
        // CHECK: func.func @tr_ir_param
        // CHECK-SAME: memref<4xf32>
        // CHECK: bufferization.to_tensor
        // CHECK-SAME: restrict writable
        """)

    def test_tensor_return_has_materialize_in_destination(self, check_ir):
        """Tensor return should use materialize_in_destination restrict writable"""
        @ml_function
        def tr_ir_mat() -> Tensor[f32, 4]:
            return Tensor[f32, 4]([1.0, 2.0, 3.0, 4.0])

        tr_ir_mat()

        check_ir("""
        // CHECK: func.func @tr_ir_mat
        // CHECK: bufferization.materialize_in_destination
        // CHECK-SAME: restrict writable
        """)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
