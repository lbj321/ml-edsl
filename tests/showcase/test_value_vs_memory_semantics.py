"""Showcase: value semantics (tensor) vs memory semantics (array/memref).

Three angles, same Python expression, completely different IR:

  1. Element-wise add — a + b looks identical in both, but:
       Tensor → linalg.map  (high-level op, compiler owns the loops)
       Array  → scf.for + memref.load/store  (loops emitted by MemRefBuilder)

  2. Point update — insert/store:
       Tensor → tensor.insert returns a NEW tensor SSA value
       Array  → memref.store mutates the buffer in place (including the input!)

  3. Aliasing — same code, different result:
       updated = b.at[0].set(99); return updated[0] + b[0]
       Tensor → 99 + 10 = 109  (updated and b are independent values)
       Array  → 99 + 99 = 198  (updated and b alias the same buffer)

Run with SAVE_IR=1 to compare the generated IR:
  SAVE_IR=1 python3 -m pytest tests/showcase/test_value_vs_memory_semantics.py -v
"""

import numpy as np
import pytest

from mlir_edsl import ml_function, Tensor, Array, f32, i32


N = 8


# ==================== 1. Element-wise add ====================

class TestTensorAdd:
    """a + b on tensors emits linalg.map — one high-level op."""

    def test_tensor_add(self, backend):
        @ml_function
        def tensor_add(a: Tensor[f32, N], b: Tensor[f32, N]) -> Tensor[f32, N]:
            return a + b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        b = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(tensor_add(a, b), a + b, atol=1e-6)

    def test_tensor_add_negatives(self, backend):
        @ml_function
        def tensor_add_neg(a: Tensor[f32, N], b: Tensor[f32, N]) -> Tensor[f32, N]:
            return a + b

        a = np.full(N, -3.0, dtype=np.float32)
        b = np.full(N,  1.0, dtype=np.float32)
        np.testing.assert_allclose(tensor_add_neg(a, b), np.full(N, -2.0, dtype=np.float32), atol=1e-6)


class TestArrayAdd:
    """a + b on arrays emits scf.for + memref.load/store — MemRefBuilder owns the loops."""

    def test_array_add(self, backend):
        @ml_function
        def array_add(a: Array[f32, N], b: Array[f32, N]) -> Array[f32, N]:
            return a + b

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        b = np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(array_add(a, b), a + b, atol=1e-6)

    def test_array_add_negatives(self, backend):
        @ml_function
        def array_add_neg(a: Array[f32, N], b: Array[f32, N]) -> Array[f32, N]:
            return a + b

        a = np.full(N, -3.0, dtype=np.float32)
        b = np.full(N,  1.0, dtype=np.float32)
        np.testing.assert_allclose(array_add_neg(a, b), np.full(N, -2.0, dtype=np.float32), atol=1e-6)


# ==================== 2. Point update ====================

class TestTensorInsert:
    """tensor.insert returns a new SSA value — original tensor is unchanged."""

    def test_insert_returns_new_value(self, backend):
        @ml_function
        def tensor_update(t: Tensor[i32, 4]) -> Tensor[i32, 4]:
            return t.at[1].set(99)

        t = np.array([10, 20, 30, 40], dtype=np.int32)
        result = tensor_update(t)
        np.testing.assert_array_equal(result, [10, 99, 30, 40])

    def test_chain_updates(self, backend):
        """Each update is an independent new value — no aliasing between steps."""
        @ml_function
        def tensor_chain(t: Tensor[i32, 4]) -> Tensor[i32, 4]:
            t1 = t.at[0].set(100)
            t2 = t1.at[2].set(300)
            return t2

        t = np.array([1, 2, 3, 4], dtype=np.int32)
        result = tensor_chain(t)
        np.testing.assert_array_equal(result, [100, 2, 300, 4])


class TestArrayStore:
    """memref.store mutates the buffer in place — no new allocation."""

    def test_store_mutates(self, backend):
        @ml_function
        def array_update(a: Array[i32, 4]) -> Array[i32, 4]:
            return a.at[1].set(99)

        a = np.array([10, 20, 30, 40], dtype=np.int32)
        result = array_update(a)
        np.testing.assert_array_equal(result, [10, 99, 30, 40])

    def test_chain_stores(self, backend):
        """Each .at[].set() is a store to the same underlying buffer."""
        @ml_function
        def array_chain(a: Array[i32, 4]) -> Array[i32, 4]:
            a = a.at[0].set(100)
            a = a.at[2].set(300)
            return a

        a = np.array([1, 2, 3, 4], dtype=np.int32)
        result = array_chain(a)
        np.testing.assert_array_equal(result, [100, 2, 300, 4])


# ==================== 3. Aliasing ====================

class TestTensorAliasing:
    """updated and b are independent SSA values — no aliasing possible."""

    def test_no_alias(self, backend):
        @ml_function
        def tensor_alias(b: Tensor[i32, 4]) -> i32:
            updated = b.at[0].set(99)
            return updated[0] + b[0]  # 99 + 10 = 109 — two distinct values

        b = np.array([10, 20, 30, 40], dtype=np.int32)
        assert tensor_alias(b) == 109


class TestArrayAliasing:
    """updated and b alias the same buffer — store is immediately visible through both names."""

    def test_alias(self, backend):
        @ml_function
        def array_alias(b: Array[i32, 4]) -> i32:
            updated = b.at[0].set(99)
            return updated[0] + b[0]  # 99 + 99 = 198 — same memory

        b = np.array([10, 20, 30, 40], dtype=np.int32)
        assert array_alias(b) == 198


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
