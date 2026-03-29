"""Test linalg IR structure (Phase 8.2)

Pre-lowering: verify linalg.dot, linalg.matmul, linalg.reduce appear in MLIR IR.
Post-lowering: verify linalg ops are replaced by scf.for after convert-linalg-to-loops.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Tensor, f32, dot, matmul, tensor_map, tensor_sum


class TestLinalgDotIR:
    """IR structure tests for linalg.dot"""

    def test_dot_emits_linalg_dot(self, check_ir):
        """Pre-lowering IR contains linalg.dot"""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_ir("""
        // CHECK: linalg.dot
        """)

    def test_dot_result_is_scalar(self, check_ir):
        """Pre-lowering IR: tensor inputs used directly, dot extracts scalar"""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @dot_fn(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>)
        // CHECK-NOT: bufferization.to_tensor
        // CHECK: tensor.empty
        // CHECK: linalg.fill
        // CHECK: linalg.dot
        // CHECK: tensor.extract
        // CHECK: return {{.*}} : f32
        """)

    def test_dot_lowered_to_loops(self, check_lowered_ir):
        """After linalg-vectorize, linalg.dot is replaced by vector ops"""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.dot
        """, after="linalg-vectorize")


class TestLinalgMatmulIR:
    """IR structure tests for linalg.matmul"""

    def test_matmul_emits_linalg_matmul(self, check_ir):
        """Pre-lowering IR contains linalg.matmul"""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_ir("""
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        """)

    def test_matmul_lowered_to_loops(self, check_lowered_ir):
        """After linalg-vectorize, linalg.matmul is replaced by vector ops"""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.matmul
        """, after="linalg-vectorize")


class TestDirectOutputBuffer:
    """IR tests for Phase 8.6: array-returning ops write directly into the
    Python-allocated out-param — no intermediate alloca+copy."""

    def test_map_writes_into_out_param(self, check_ir):
        """linalg.map outs(...) is the hidden out-param tensor, not an alloca."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @scale(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>
        // CHECK-SAME: bufferization.writable = true
        // CHECK-NOT: bufferization.to_tensor
        // CHECK: linalg.map
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.alloca() : memref<4xf32>
        """)

    def test_matmul_writes_into_out_param(self, check_ir):
        """linalg.matmul outs(...) is the hidden out-param tensor, not an alloca."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @mm(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>
        // CHECK-SAME: bufferization.writable = true
        // CHECK-NOT: bufferization.to_tensor
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.alloca() : memref<2x2xf32>
        """)

    def test_map_materializes_into_out_param(self, check_ir):
        """linalg.map result is materialized into the writable tensor out-param."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: bufferization.materialize_in_destination {{.*}} in %arg1
        // CHECK-NOT: restrict writable
        """)

    def test_matmul_materializes_into_out_param(self, check_ir):
        """linalg.matmul result is materialized into the writable tensor out-param."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: bufferization.materialize_in_destination {{.*}} in %arg2
        // CHECK-NOT: restrict writable
        """)


class TestLinalgReduceIR:
    """IR structure tests for linalg.reduce (via tensor_sum)"""

    def test_reduce_uses_0d_tensor_accumulator(self, check_ir):
        """Pre-lowering IR: tensor_sum uses tensor.from_elements init, linalg.reduce, tensor.extract."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: tensor.from_elements
        // CHECK: linalg.reduce
        // CHECK: tensor.extract
        // CHECK: return {{.*}} : f32
        """)

    def test_reduce_no_alloca(self, check_ir):
        """Pre-lowering IR: no memref.alloca for the scalar accumulator."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK-NOT: memref.alloca
        // CHECK: linalg.reduce
        """)

    def test_reduce_lowered_to_loops(self, check_lowered_ir):
        """After linalg-vectorize, linalg.reduce is replaced by vector ops."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.reduce
        """, after="linalg-vectorize")

    def test_reduce_vectorized(self, check_lowered_ir):
        """After linalg-vectorize, linalg.reduce is replaced by vector ops."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.reduce
        """, after="linalg-vectorize")


class TestLinalgDotVectorizationIR:
    """IR tests for linalg.dot vectorization (Phase 9.1)"""

    def test_dot_vectorized(self, check_lowered_ir):
        """After linalg-vectorize, linalg.dot is replaced by vector ops."""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.dot
        """, after="linalg-vectorize")


class TestLinalgMatmulVectorizationIR:
    """IR tests for linalg.matmul vectorization (Phase 9.1)"""

    def test_matmul_vectorized(self, check_lowered_ir):
        """After linalg-vectorize, linalg.matmul is replaced by vector ops."""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.transfer_read
        // CHECK-NOT: linalg.matmul
        """, after="linalg-vectorize")


class TestLinalgMatmulLargeIR:
    """IR tests for larger matmul vectorization"""

    def test_matmul_8x8_vectorized(self, check_lowered_ir):
        """8x8 matmul vectorizes directly without tiling."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.matmul
        """, after="linalg-vectorize")


class TestLinalgMatmulTilingPass:
    """IR tests for LinalgMatmulTilingPass (linalg-tile-matmul).

    Matrices with all dims > 8 are tiled into scf.for loops over 8x8 tiles.
    Matrices with any dim <= 8 are left untouched for direct vectorization.
    """

    def test_large_matmul_tiled_to_scf_for(self, check_lowered_ir):
        """16x16 matmul is replaced by two nested scf.for loops over 8x8 tiles."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: scf.for
        // CHECK: linalg.matmul
        """, after="linalg-tile-matmul")

    def test_large_matmul_tile_step_is_8(self, check_lowered_ir):
        """Tiling uses step size 8 for both M and N dimensions."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: arith.constant 8 : index
        // CHECK: scf.for
        """, after="linalg-tile-matmul")

    def test_large_matmul_produces_subviews(self, check_lowered_ir):
        """Tiled matmul slices inputs into 8x16, 16x8 and output into 8x8 subviews."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: memref.subview {{.*}} [8, 16] [1, 1]
        // CHECK: memref.subview {{.*}} [16, 8] [1, 1]
        // CHECK: memref.subview {{.*}} [8, 8] [1, 1]
        """, after="linalg-tile-matmul")

    def test_boundary_8x8_matmul_tiled(self, check_lowered_ir):
        """8x8 matmul is tiled into a single 8x8 tile (one-iteration scf.for loops)."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: linalg.matmul
        """, after="linalg-tile-matmul")

    def test_small_matmul_tiled(self, check_lowered_ir):
        """2x2 matmul is tiled — produces scf.for loops with a partial tile."""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: linalg.matmul
        """, after="linalg-tile-matmul")


class TestVectorCleanupPass:
    """IR tests for VectorCleanupPass (vector-cleanup).

    Fuses the mulf + multi_reduction pattern emitted by linalg::vectorize
    into vector.contract, giving the LLVM backend explicit contraction semantics.
    """

    def test_matmul_fused_to_vector_contract(self, check_lowered_ir):
        """After vector-cleanup, mulf+multi_reduction is fused into vector.contract."""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.contract
        // CHECK-NOT: vector.multi_reduction
        """, after="vector-cleanup")

    def test_matmul_contract_has_reduction_iterator(self, check_lowered_ir):
        """vector.contract encodes matmul semantics: parallel+parallel+reduction."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.contract {{.*}} iterator_types = ["parallel", "parallel", "reduction"]
        """, after="vector-cleanup")

    def test_large_tiled_matmul_fused_to_contract(self, check_lowered_ir):
        """After tiling + cleanup, each tile's matmul becomes a vector.contract inside scf.for."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: vector.contract
        // CHECK-NOT: linalg.matmul
        """, after="vector-cleanup")


class TestLinalgBinaryOpIR:
    """IR structure tests for LinalgBinaryOp (same-shape, scalar, bias add)."""

    def test_same_shape_add_emits_linalg_map(self, check_ir):
        """Tensor + Tensor (same shape) emits linalg.map with two inputs."""
        @ml_function
        def add_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a + b

        add_fn(np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        check_ir("""
        // CHECK: linalg.map
        // CHECK-NOT: linalg.broadcast
        """)

    def test_scalar_broadcast_emits_linalg_map(self, check_ir):
        """Tensor * scalar emits a single linalg.map (no broadcast op needed)."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return a * 2.0

        scale(np.ones(4, dtype=np.float32))
        check_ir("""
        // CHECK: linalg.map
        // CHECK-NOT: linalg.broadcast
        """)

    def test_bias_add_emits_broadcast_then_map(self, check_ir):
        """[M,N] + [N] emits linalg.broadcast to expand bias, then linalg.map for add."""
        @ml_function
        def bias_add(x: Tensor[f32, 2, 4], b: Tensor[f32, 4]) -> Tensor[f32, 2, 4]:
            return x + b

        bias_add(np.zeros((2, 4), dtype=np.float32), np.zeros(4, dtype=np.float32))
        check_ir("""
        // CHECK: linalg.broadcast
        // CHECK: linalg.map
        """)
