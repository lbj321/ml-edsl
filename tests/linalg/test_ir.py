"""Pre-lowering IR structure tests for linalg ops.

Verifies what MLIRBuilder emits before any lowering passes run — dialect choice,
op structure, type representation, and structural invariants invisible to runtime tests.

Post-lowering / pass-level IR tests live in test_lowering_ir.py.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, dot, matmul, tensor_map, tensor_sum, relu, leaky_relu


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


class TestEmittedIR:
    """Verify the pre-lowering IR emitted by MLIRBuilder — tensor-returning
    form with tensor.empty init, before any lowering passes run."""

    def test_matmul_emits_tensor_return(self, check_ir):
        """Frontend emits tensor-returning function (no out-param yet)."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @mm(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32>
        // CHECK: tensor.empty
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK: return {{.*}} : tensor<2x2xf32>
        """)

    def test_map_emits_tensor_return(self, check_ir):
        """Frontend emits tensor-returning function with tensor.empty init."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @scale(%arg0: tensor<4xf32>) -> tensor<4xf32>
        // CHECK: tensor.empty
        // CHECK: linalg.map
        // CHECK: return {{.*}} : tensor<4xf32>
        """)

    def test_emitted_ir_has_no_out_param(self, check_ir):
        """Frontend does not emit an out-param — that is added by TensorReturnToOutParamPass."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_ir("""
        // CHECK: func.func @mm(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32>
        // CHECK-NOT: bufferization.writable
        // CHECK-NOT: bufferization.materialize_in_destination
        """)


class TestDirectOutputBuffer:
    """IR tests for array-returning ops writing directly into the Python-allocated
    out-param — no intermediate alloca+copy."""

    def test_map_writes_into_out_param(self, check_lowered_ir):
        """After TensorReturnToOutParamPass: void + writable out-param, linalg.map outs is the out-param."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: func.func @scale(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>
        // CHECK-SAME: bufferization.writable = true
        // CHECK: linalg.map
        // CHECK-NOT: bufferization.materialize_in_destination
        """, after="tensor-return-to-out-param")

    def test_matmul_writes_into_out_param(self, check_lowered_ir):
        """After TensorReturnToOutParamPass: void + writable out-param, linalg.matmul outs is the out-param."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: func.func @mm(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>
        // CHECK-SAME: bufferization.writable = true
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK-NOT: bufferization.materialize_in_destination
        """, after="tensor-return-to-out-param")

    def test_map_no_copy_after_bufferization(self, check_lowered_ir):
        """After bufferization: linalg.map writes directly into out-param, no memref.copy."""
        @ml_function
        def scale(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return tensor_map(a, lambda x: x * 2.0)

        scale(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: func.func @scale(%arg0: memref<4xf32>, %arg1: memref<4xf32>
        // CHECK: linalg.map
        // CHECK-NOT: memref.copy
        // CHECK-NOT: memref.alloc
        """, after="one-shot-bufferize")

    def test_matmul_no_copy_after_bufferization(self, check_lowered_ir):
        """After bufferization: linalg.matmul writes directly into out-param, no memref.copy."""
        @ml_function
        def mm(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
           np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))
        check_lowered_ir("""
        // CHECK: func.func @mm(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>
        // CHECK: linalg.fill
        // CHECK: linalg.matmul
        // CHECK-NOT: memref.copy
        """, after="one-shot-bufferize")

    def test_bias_relu_fused_and_no_copy(self, check_lowered_ir):
        """bias+relu generics are fused into one linalg.generic that writes directly into out-param."""
        @ml_function
        def dense_relu(W: Tensor[f32, 2, 4], x: Tensor[f32, 4, 3], b: Tensor[f32, 3]) -> Tensor[f32, 2, 3]:
            return relu(matmul(W, x) + b)

        W = np.zeros((2, 4), dtype=np.float32)
        x = np.zeros((4, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)
        dense_relu(W, x, b)

        check_lowered_ir("""
        // CHECK: func.func @dense_relu
        // CHECK: linalg.matmul
        // CHECK: linalg.generic
        // CHECK-NOT: linalg.generic
        // CHECK-NOT: memref.copy
        // CHECK: return
        """, after="one-shot-bufferize")


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

    def test_bias_add_emits_generic_with_broadcast_map(self, check_ir):
        """[M,N] + [N] emits a single linalg.generic with broadcast indexing map — no intermediate broadcast op."""
        @ml_function
        def bias_add(x: Tensor[f32, 2, 4], b: Tensor[f32, 4]) -> Tensor[f32, 2, 4]:
            return x + b

        bias_add(np.zeros((2, 4), dtype=np.float32), np.zeros(4, dtype=np.float32))
        check_ir("""
        // CHECK: linalg.generic
        // CHECK-NOT: linalg.broadcast
        """)


class TestLinalgActivationIR:
    """IR structure tests for LinalgActivation (relu, leaky_relu).

    Verifies that activations emit linalg.generic with pure arith ops,
    not linalg.map with scf.if — which matters for vectorization and fusion.
    """

    def test_relu_emits_linalg_generic(self, check_ir):
        """relu emits linalg.generic, not linalg.map."""
        @ml_function
        def apply_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return relu(a)

        apply_relu(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        check_ir("""
        // CHECK: linalg.generic
        // CHECK-NOT: linalg.map
        """)

    def test_relu_uses_maximumf(self, check_ir):
        """relu body uses arith.maximumf — no scf.if or control flow."""
        @ml_function
        def apply_relu(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return relu(a)

        apply_relu(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        check_ir("""
        // CHECK: arith.maximumf
        // CHECK-NOT: scf.if
        """)

    def test_leaky_relu_emits_linalg_generic(self, check_ir):
        """leaky_relu emits linalg.generic, not linalg.map."""
        @ml_function
        def apply_leaky(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return leaky_relu(a, alpha=0.1)

        apply_leaky(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        check_ir("""
        // CHECK: linalg.generic
        // CHECK-NOT: linalg.map
        """)

    def test_leaky_relu_no_control_flow(self, check_ir):
        """leaky_relu body uses arith ops only — no scf.if or arith.select."""
        @ml_function
        def apply_leaky(a: Tensor[f32, 4]) -> Tensor[f32, 4]:
            return leaky_relu(a, alpha=0.1)

        apply_leaky(np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        check_ir("""
        // CHECK-NOT: scf.if
        // CHECK-NOT: arith.select
        // CHECK: arith.maximumf
        // CHECK: arith.minimumf
        """)
