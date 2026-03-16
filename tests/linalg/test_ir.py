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
        """Pre-lowering IR: linalg.dot result is loaded as f32"""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_ir("""
        // CHECK: tensor.empty
        // CHECK: linalg.fill
        // CHECK: linalg.dot
        // CHECK: tensor.extract
        // CHECK: return {{.*}} : f32
        """)

    def test_dot_lowered_to_loops(self, check_lowered_ir):
        """After convert-linalg-to-loops, linalg.dot is gone and scf.for appears"""
        @ml_function
        def dot_fn(a: Tensor[f32, 4], b: Tensor[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK-NOT: linalg.dot
        """, after="convert-linalg-to-loops")


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
        """After convert-linalg-to-loops, linalg.matmul is gone"""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK-NOT: linalg.matmul
        """, after="convert-linalg-to-loops")


class TestDirectOutputBuffer:
    """IR tests for Phase 8.6: array-returning ops write directly into the
    Python-allocated out-param — no intermediate alloca+copy."""

    def test_map_writes_into_out_param(self, check_ir):
        """linalg.map outs(...) is the hidden out-param, not an alloca."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @scale(%arg0: memref<4xf32>, %arg1: memref<4xf32>)
        // CHECK: bufferization.to_tensor %arg1
        // CHECK: linalg.map
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.alloca() : memref<4xf32>
        """)

    def test_matmul_writes_into_out_param(self, check_ir):
        """linalg.matmul outs(...) is the hidden out-param, not an alloca."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @mm(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>)
        // CHECK: bufferization.to_tensor %arg2
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.alloca() : memref<2x2xf32>
        """)

    def test_map_materializes_into_out_param(self, check_ir):
        """linalg.map result is materialized into the out-param via materialize_in_destination."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: bufferization.materialize_in_destination {{.*}} in restrict writable %arg1
        """)

    def test_matmul_materializes_into_out_param(self, check_ir):
        """linalg.matmul result is materialized into the out-param via materialize_in_destination."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: bufferization.materialize_in_destination {{.*}} in restrict writable %arg2
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
        """After convert-linalg-to-loops, linalg.reduce is gone and scf.for appears."""
        @ml_function
        def my_sum(a: Tensor[f32, 4]) -> f32:
            return tensor_sum(a)

        my_sum(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK-NOT: linalg.reduce
        """, after="convert-linalg-to-loops")
