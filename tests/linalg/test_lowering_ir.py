"""Post-lowering IR structure tests for linalg ops.

Verifies what the lowering pipeline transforms — that high-level ops are correctly
replaced by lower-level ops at specific pipeline stages.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, dot, matmul, tensor_sum, relu


class TestLinalgDotLoweringIR:
    """Post-lowering IR tests for linalg.dot"""

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


class TestLinalgMatmulLoweringIR:
    """Post-lowering IR tests for linalg.matmul"""

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


class TestLinalgReduceLoweringIR:
    """Post-lowering IR tests for linalg.reduce"""

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
    """IR tests for linalg.dot vectorization"""

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
    """IR tests for linalg.matmul vectorization"""

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
        """16x16 matmul is replaced by three nested scf.for loops (M, N, K tiled to 8)."""
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
        """Tiled matmul slices all three operands into 8x8 subviews (M, N, K all tiled)."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: memref.subview {{.*}} [8, 8] [1, 1]
        // CHECK: memref.subview {{.*}} [8, 8] [1, 1]
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


class TestLinalgMatmulToContractPass:
    """IR tests for LinalgMatmulToContractPass (linalg-matmul-to-contract).

    Bypasses the linalg vectorizer (which produces a 3D double-broadcast
    contract) by lowering static 8x8 matmul tiles directly to vector.contract
    with standard 2D indexing maps {(m,k),(k,n),(m,n)}.
    """

    def test_8x8_matmul_emits_vector_contract(self, check_lowered_ir):
        """8x8 linalg.matmul is replaced by vector.contract after this pass."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.contract
        // CHECK-NOT: linalg.matmul
        """, after="linalg-matmul-to-contract")

    def test_contract_has_standard_2d_matmul_maps(self, check_lowered_ir):
        """vector.contract uses standard (m,k)x(k,n)->(m,n) indexing maps, not the
        3D double-broadcast form the linalg vectorizer emits."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: affine_map<(d0, d1, d2) -> (d0, d2)>
        // CHECK: affine_map<(d0, d1, d2) -> (d2, d1)>
        // CHECK: affine_map<(d0, d1, d2) -> (d0, d1)>
        """, after="linalg-matmul-to-contract")

    def test_non_8x8_matmul_not_lowered(self, check_lowered_ir):
        """Non-8x8 matmuls are left for the linalg vectorizer — only exact 8x8
        tiles are handled by this pass."""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 2], B: Tensor[f32, 2, 2]) -> Tensor[f32, 2, 2]:
            return matmul(A, B)

        mm_fn(np.ones((2, 2), dtype=np.float32),
              np.ones((2, 2), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: linalg.matmul
        // CHECK-NOT: vector.contract
        """, after="linalg-matmul-to-contract")


class TestVectorContractToOuterProductPass:
    """IR tests for VectorContractToOuterProductPass (vector-contract-to-outerproduct).

    Lowers vector.contract with standard 2D matmul maps to vector.fma via the
    OuterProduct strategy (contract → outerproduct → fma in one pass).
    """

    def test_8x8_contract_lowered_to_fma(self, check_lowered_ir):
        """vector.contract on an 8x8 matmul is fully lowered to vector.fma."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.fma
        // CHECK-NOT: vector.contract
        """, after="vector-contract-to-outerproduct")

    def test_large_tiled_matmul_lowered_to_fma(self, check_lowered_ir):
        """Each 8x8 tile of a tiled matmul is lowered to vector.fma inside scf.for."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: vector.fma
        // CHECK-NOT: vector.contract
        """, after="vector-contract-to-outerproduct")


class TestLinalgGenericTilingPass:
    """IR tests for LinalgGenericTilingPass (linalg-tile-generic).

    Tiles linalg.generic ops along the innermost dimension to strips of 8,
    preventing LLVM O3 from seeing large vector<NxNxf32> types that cause
    combinatorial explosion in its analysis passes (e.g. bias_add/relu at 512x512).
    """

    def test_2d_elementwise_tiled_to_scf_for(self, check_lowered_ir):
        """2D elementwise op is tiled into scf.for over innermost-dim strips."""
        @ml_function
        def bias_fn(X: Tensor[f32, 16, 16], b: Tensor[f32, 16]) -> Tensor[f32, 16, 16]:
            return X + b

        bias_fn(np.ones((16, 16), dtype=np.float32),
                np.ones(16, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: linalg.generic
        """, after="linalg-tile-generic")

    def test_tile_step_is_8(self, check_lowered_ir):
        """Innermost dimension is tiled with step size 8."""
        @ml_function
        def bias_fn(X: Tensor[f32, 16, 16], b: Tensor[f32, 16]) -> Tensor[f32, 16, 16]:
            return X + b

        bias_fn(np.ones((16, 16), dtype=np.float32),
                np.ones(16, dtype=np.float32))
        check_lowered_ir("""
        // CHECK: arith.constant 8 : index
        // CHECK: scf.for
        """, after="linalg-tile-generic")

    def test_generic_removed_after_vectorization(self, check_lowered_ir):
        """After vectorization, tiled linalg.generic strips are fully lowered to vector ops."""
        @ml_function
        def relu_fn(X: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return relu(X)

        relu_fn(np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.
        // CHECK-NOT: linalg.generic
        """, after="linalg-vectorize")
