"""Test linalg IR structure (Phase 8.2)

Pre-lowering: verify linalg.dot and linalg.matmul appear in MLIR IR.
Post-lowering: verify linalg ops are replaced by scf.for after convert-linalg-to-loops.
"""

import pytest
import numpy as np
from mlir_edsl import ml_function, Array, f32, dot, matmul


class TestLinalgDotIR:
    """IR structure tests for linalg.dot"""

    def test_dot_emits_linalg_dot(self, check_ir):
        """Pre-lowering IR contains linalg.dot"""
        @ml_function
        def dot_fn(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_ir("""
        // CHECK: linalg.dot
        """)

    def test_dot_result_is_scalar(self, check_ir):
        """Pre-lowering IR: linalg.dot result is loaded as f32"""
        @ml_function
        def dot_fn(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
            return dot(a, b)

        dot_fn(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
               np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))
        check_ir("""
        // CHECK: memref.alloca() : memref<f32>
        // CHECK: linalg.dot
        // CHECK: memref.load
        // CHECK: return {{.*}} : f32
        """)

    def test_dot_lowered_to_loops(self, check_lowered_ir):
        """After convert-linalg-to-loops, linalg.dot is gone and scf.for appears"""
        @ml_function
        def dot_fn(a: Array[f32, 4], b: Array[f32, 4]) -> f32:
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
        def mm_fn(A: Array[f32, 2, 2], B: Array[f32, 2, 2]) -> Array[f32, 2, 2]:
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
        def mm_fn(A: Array[f32, 2, 2], B: Array[f32, 2, 2]) -> Array[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
              np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK-NOT: linalg.matmul
        """, after="convert-linalg-to-loops")
