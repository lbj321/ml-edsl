"""Post-lowering IR structure tests for linalg ops.

Verifies what the lowering pipeline transforms — that high-level ops are correctly
replaced by lower-level ops at specific pipeline stages.
"""

import numpy as np
from mlir_edsl import ml_function, Tensor, f32, i32, dot, matmul, tensor_sum, relu


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

class TestVectorCleanupPass:
    """IR tests for VectorCleanupPass (vector-cleanup).

    Fuses the mulf + multi_reduction pattern emitted by linalg::vectorize
    into vector.contract, giving the LLVM backend explicit contraction semantics.
    Also splits broadcasting transfer_reads so the broadcasts fold into the
    contract's indexing maps, leaving 2D operands.
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

    def test_non_square_matmul_fused_to_contract(self, check_lowered_ir):
        """Non-square M/K/N (2x4 @ 4x3) also becomes a vector.contract."""
        @ml_function
        def mm_fn(A: Tensor[f32, 2, 4], B: Tensor[f32, 4, 3]) -> Tensor[f32, 2, 3]:
            return matmul(A, B)

        mm_fn(np.ones((2, 4), dtype=np.float32),
              np.ones((4, 3), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.contract
        // CHECK-NOT: linalg.matmul
        """, after="vector-cleanup")

    def test_contract_operands_are_2d(self, check_lowered_ir):
        """The contract takes 2D operands, not the 3D double-broadcast form
        linalg::vectorize emits, which the OuterProduct lowering cannot turn
        into FMAs. Checks shapes rather than maps: packed A is k-major, so its
        transpose is folded into a (k, m) map."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.contract {{.*}} : vector<{{[0-9]+}}x{{[0-9]+}}xf32>, vector<{{[0-9]+}}x{{[0-9]+}}xf32> into vector<{{[0-9]+}}x{{[0-9]+}}xf32>
        """, after="vector-cleanup")

class TestVectorContractToOuterProductPass:
    """IR tests for VectorContractToOuterProductPass (vector-contract-to-outerproduct).

    Lowers vector.contract with standard 2D matmul maps to vector.outerproduct
    via the OuterProduct strategy, with disableOuterProductLowering=true so the
    outerproduct→fma decomposition is left for convert-vector-to-llvm later.
    """

    def test_8x8_contract_lowered_to_outerproduct(self, check_lowered_ir):
        """vector.contract on an 8x8 matmul is fully lowered to vector.outerproduct."""
        @ml_function
        def mm_fn(A: Tensor[f32, 8, 8], B: Tensor[f32, 8, 8]) -> Tensor[f32, 8, 8]:
            return matmul(A, B)

        mm_fn(np.ones((8, 8), dtype=np.float32),
              np.ones((8, 8), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: vector.outerproduct
        // CHECK-NOT: vector.contract
        // CHECK-NOT: vector.fma
        """, after="vector-contract-to-outerproduct")

    def test_large_tiled_matmul_lowered_to_outerproduct(self, check_lowered_ir):
        """Each 8x8 tile of a tiled matmul is lowered to vector.outerproduct inside scf.for."""
        @ml_function
        def mm_fn(A: Tensor[f32, 16, 16], B: Tensor[f32, 16, 16]) -> Tensor[f32, 16, 16]:
            return matmul(A, B)

        mm_fn(np.ones((16, 16), dtype=np.float32),
              np.ones((16, 16), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: scf.for
        // CHECK: vector.outerproduct
        // CHECK-NOT: vector.contract
        // CHECK-NOT: vector.fma
        """, after="vector-contract-to-outerproduct")


class TestAllocaScopeCleanupPass:
    """IR tests for AllocaScopeCleanupPass (alloca-scope-cleanup).

    ConvertSCFToOpenMPPass unconditionally wraps the omp.parallel loop body in a
    memref.alloca_scope. This pass inlines that scope away when it contains no
    memref.alloca, since it must be gone before scf-to-cf runs: AllocaScopeOp
    requires a single-block body, and scf-to-cf introduces branches for any
    scf.for still inside it. Left un-inlined, this crashes on large matmuls
    (e.g. 1024x1024) with 'expects region #0 to have 0 or 1 blocks'.
    """

    def test_omp_loop_body_has_no_alloca_scope(self, check_lowered_ir):
        """256x256 matmul: the blocked ic x jc forall has two iterations (MC =
        128), so it survives canonicalize as a real omp.parallel. Its loop body
        must not retain the vacuous memref.alloca_scope ConvertSCFToOpenMPPass
        wraps it in."""
        @ml_function
        def mm_fn(A: Tensor[f32, 256, 256], B: Tensor[f32, 256, 256]) -> Tensor[f32, 256, 256]:
            return matmul(A, B)

        mm_fn(np.ones((256, 256), dtype=np.float32),
              np.ones((256, 256), dtype=np.float32))
        check_lowered_ir("""
        // CHECK: omp.parallel
        // CHECK-NOT: memref.alloca_scope
        """, after="alloca-scope-cleanup")


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
