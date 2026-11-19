#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <array>

namespace {

// Tiles a fusion root and greedily fuses its linalg producers upward into
// the generated scf.forall loops. tileConsumerAndFuseProducersUsingSCF walks
// the use-def chain transitively, so this pulls in a whole producer chain in
// one call without needing them pre-merged into a single generic.
//
// Two trigger paths, tried in order:
//   1. Epilogue fusion: the relu linalg.generic (found via its "relu"
//      library_call attribute, see LinalgBuilder.cpp) is the root, pulling in
//      the bias_add → matmul → fill chain above it.
//   2. Bare matmul: when there is no relu epilogue, a plain linalg.matmul is
//      used as the root instead, pulling in just its fill producer. Without
//      this, an unfused fill is left as a separate top-level op that (on GPU)
//      becomes its own gpu.launch_func, or (on CPU) is lowered to a plain
//      scf.for outside the matmul's omp.parallel region and runs single-
//      threaded ahead of the parallel matmul.
//
// Running in tensor land (before bufferization) enables this fusion: the
// producer chain all executes on the same [tileM×tileN] tile, keeping the
// matmul output in L2 cache instead of writing it to DRAM first.
//
// After this pass the body of the scf.forall contains either:
//   linalg.fill (tile) → linalg.matmul (tile, full K) → linalg.generic (tile)
// or, for the bare-matmul path:
//   linalg.fill (tile) → linalg.matmul (tile, full K)
// The scf.forall is subsequently converted to omp.parallel by the existing
// ForallToParallelLoop + ConvertSCFToOpenMP pass sequence.
//
// Note: both walks below keep the *last* matching op found rather than the
// first, so a function with multiple independent relu epilogues or multiple
// independent bare matmuls will only have one of them fused here; the others
// are left unfused with no diagnostic emitted.
struct LinalgOuterTileAndFusePass
    : public mlir::PassWrapper<LinalgOuterTileAndFusePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgOuterTileAndFusePass)

  int64_t tileM, tileN;
  explicit LinalgOuterTileAndFusePass(int64_t m, int64_t n)
      : tileM(m), tileN(n) {}

  // tileConsumerAndFuseProducersUsingSCF constructs a new scf.forall, which
  // needs SCFDialect loaded even when the input IR contains no scf ops yet
  // (e.g. this pass running standalone on pre-tiling IR, as in
  // cpp/tools/mlir-edsl-opt).
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-outer-tile-and-fuse";
  }
  llvm::StringRef getDescription() const override {
    return "Tile relu epilogue (or bare matmul) and fuse matmul+fill "
           "producers into scf.forall loops (epilogue fusion, "
           "pre-bufferization)";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    // Find the relu generic by its library_call attribute (set in
    // LinalgBuilder.cpp). Producer fusion below walks backward from here.
    mlir::Operation *consumer = nullptr;
    func.walk([&](mlir::linalg::GenericOp op) {
      auto libCall = op->getAttrOfType<mlir::StringAttr>("library_call");
      if (libCall && libCall.getValue() == "relu")
        consumer = op;
    });

    if (!consumer) {
      // No relu epilogue: fuse fill directly into a bare matmul instead.
      func.walk([&](mlir::linalg::MatmulOp op) { consumer = op; });
    }

    if (!consumer)
      return;

    llvm::SmallVector<mlir::OpFoldResult> tileSizes =
        mlir::getAsIndexOpFoldResult(func->getContext(), {tileM, tileN});
    mlir::scf::SCFTileAndFuseOptions opts;
    opts.setTilingOptions(
        mlir::scf::SCFTilingOptions()
            .setTileSizes(tileSizes)
            .setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForallOp));

    rewriter.setInsertionPoint(consumer);
    auto fuseResult = mlir::scf::tileConsumerAndFuseProducersUsingSCF(
        rewriter, mlir::cast<mlir::TilingInterface>(consumer), opts);
    if (mlir::failed(fuseResult)) {
      consumer->emitWarning(
          "linalg-outer-tile-and-fuse: tiling failed, skipping");
      return;
    }

    // Replace the consumer with the forall result; fused producers (matmul,
    // fill) are now dead and will be cleaned up by the subsequent canonicalizer.
    llvm::SmallVector<mlir::Value> repls;
    for (mlir::Value res : consumer->getResults())
      repls.push_back(fuseResult->replacements.lookup(res));
    rewriter.replaceOp(consumer, repls);
  }
};

// True when `op` is a linalg.fill whose result initializes a matmul that is
// left for scalar lowering (see mlir_edsl::isLeftForScalarLowering).
static bool initializesScalarMatmul(mlir::Operation *op) {
  if (!llvm::isa<mlir::linalg::FillOp>(op) || op->getNumResults() != 1)
    return false;
  return llvm::any_of(op->getResult(0).getUsers(), [](mlir::Operation *user) {
    return mlir_edsl::isLeftForScalarLowering(user);
  });
}

struct LinalgVectorizationPass
    : public mlir::PassWrapper<LinalgVectorizationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgVectorizationPass)

  // linalg::vectorize constructs new vector.* ops.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  llvm::StringRef getArgument() const override { return "linalg-vectorize"; }
  llvm::StringRef getDescription() const override {
    return "Vectorize linalg structured ops to vector dialect";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::Operation *> linalgOps;
    func.walk([&](mlir::linalg::LinalgOp op) {
      linalgOps.push_back(op.getOperation());
    });

    for (mlir::Operation *op : linalgOps) {
      if (!op->getBlock())
        continue; // erased by a prior iteration (e.g. nested op inside
                  // vectorized outer)
      if (!mlir::linalg::hasVectorizationImpl(op))
        continue;
      // Vectorizing such a matmul whole would build one matrix-sized vector.
      // The fill that initializes it is untiled too.
      if (mlir_edsl::isLeftForScalarLowering(op) || initializesScalarMatmul(op))
        continue;
      rewriter.setInsertionPoint(op);
      if (mlir::failed(mlir::linalg::vectorize(rewriter, op)))
        op->emitWarning("linalg-vectorize: vectorization failed, skipping op");
      // Do NOT signalPassFailure — leave op for fallback loop lowering
    }
  }
};

// Fuses the mulf + multi_reduction pattern emitted by linalg::vectorize into
// vector.contract, giving the LLVM backend a clear contraction semantic.
// The permutation-map patterns split the vectorizer's broadcasting
// transfer_reads into a plain read plus vector.broadcast/transpose, which the
// contract patterns then fold into its indexing maps. Without them a matmul
// keeps the 3D double-broadcast form, which the OuterProduct lowering cannot
// turn into FMAs.
struct VectorCleanupPass
    : public mlir::PassWrapper<VectorCleanupPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorCleanupPass)
  llvm::StringRef getArgument() const override { return "vector-cleanup"; }
  llvm::StringRef getDescription() const override {
    return "Fuse mulf+multi_reduction into vector.contract";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(func->getContext());
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    mlir::vector::populateVectorReductionToContractPatterns(patterns);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

// Rewrites permuting vector.transfer ops into a plain transfer plus an
// explicit vector.transpose, then lowers that transpose with shuffles.
// Without this a transposing transfer_read (as the A-packing copy produces)
// reaches convert-vector-to-llvm and degenerates into a vinsertps chain.
struct VectorTransposeLoweringPass
    : public mlir::PassWrapper<VectorTransposeLoweringPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorTransposeLoweringPass)
  llvm::StringRef getArgument() const override {
    return "vector-transpose-lowering";
  }
  llvm::StringRef getDescription() const override {
    return "Lower permuting vector transfers and vector.transpose to shuffles";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(func->getContext());
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    mlir::vector::populateVectorTransposeLoweringPatterns(
        patterns, mlir::vector::VectorTransposeLowering::Shuffle16x16);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

// Inlines memref.alloca_scope ops whose body contains no memref.alloca —
// i.e. scopes wrapped by ConvertSCFToOpenMPPass "just in case" that never
// actually need stack scoping. Must run before scf-to-cf: AllocaScopeOp
// requires a single-block body, and scf-to-cf introduces branches for any
// scf.for still inside it. Deliberately narrow (only AllocaScopeOp's own
// canonicalization patterns) instead of a blanket canonicalizer pass, so
// this can't be silently defeated by unrelated pattern/pass changes
// elsewhere in the pipeline — see addCPUPasses call site for the incident
// that motivated this.
struct AllocaScopeCleanupPass
    : public mlir::PassWrapper<AllocaScopeCleanupPass,
                                mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AllocaScopeCleanupPass)
  llvm::StringRef getArgument() const override {
    return "alloca-scope-cleanup";
  }
  llvm::StringRef getDescription() const override {
    return "Inline memref.alloca_scope ops that contain no memref.alloca";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(func->getContext());
    mlir::memref::AllocaScopeOp::getCanonicalizationPatterns(
        patterns, func->getContext());
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

// Lowers vector.contract to vector.outerproduct on rank-1 vector slices.
// Must run before convert-vector-to-scf so that the 3D transfer_reads
// produced by linalg-vectorize are not expanded into broadcast+transpose+alloca
// loops — those only arise when a rank-3 contract is still present at that pass.
struct VectorContractToOuterProductPass
    : public mlir::PassWrapper<VectorContractToOuterProductPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VectorContractToOuterProductPass)
  llvm::StringRef getArgument() const override {
    return "vector-contract-to-outerproduct";
  }
  llvm::StringRef getDescription() const override {
    return "Lower vector.contract to vector.outerproduct (OuterProduct strategy)";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::RewritePatternSet patterns(func->getContext());
    mlir::vector::populateVectorContractLoweringPatterns(
        patterns, mlir::vector::VectorContractLowering::OuterProduct,
        /*benefit=*/1, /*disableOuterProductLowering=*/true);
    if (mlir::failed(mlir::applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

// Tiles linalg.matmul into scf loops with configurable tile sizes and loop
// type. Only the GPU pipeline uses it (createLinalgGPUMatmulTilingPass:
// {32,32,0} ForallOp, K untiled to avoid reduction races); the CPU pipeline
// tiles matmuls with the blocked passes.
struct LinalgMatmulTilingPass
    : public mlir::PassWrapper<LinalgMatmulTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulTilingPass)

  using LoopType = mlir::scf::SCFTilingOptions::LoopType;

  int64_t tileM, tileN, tileK;
  LoopType loopType;

  explicit LinalgMatmulTilingPass(int64_t m, int64_t n, int64_t k,
                                  LoopType lt = LoopType::ForOp)
      : tileM(m), tileN(n), tileK(k), loopType(lt) {}

  // tileUsingSCF constructs a new scf.for or scf.forall.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  llvm::StringRef getArgument() const override {
    return loopType == LoopType::ForallOp ? "linalg-tile-matmul-forall"
                                          : "linalg-tile-matmul";
  }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.matmul into scf loops over configurable tiles";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) {
      // In the ForallOp (outer parallel) configuration, skip matmuls already
      // nested inside an scf.forall: those were placed and correctly sized
      // (tileM x tileN, full K) by LinalgOuterTileAndFusePass's producer
      // fusion, and retiling them here would double-tile a already-tiled op.
      // The ForOp (inner K-tiling) configuration deliberately runs on such
      // nested matmuls, so this guard only applies to the ForallOp case.
      if (loopType == LoopType::ForallOp &&
          op->getParentOfType<mlir::scf::ForallOp>())
        return;
      matmuls.push_back(op);
    });

    for (mlir::linalg::MatmulOp op : matmuls) {
      // Named variable required — setTileSizes captures a non-owning ArrayRef.
      llvm::SmallVector<mlir::OpFoldResult> tileSizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), {tileM, tileN, tileK});
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(tileSizes);
      opts.setLoopType(loopType);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op.getOperation()), opts);
      if (mlir::failed(result)) {
        op->emitWarning("linalg-tile-matmul: tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
    }
  }
};

// Tiles linalg.generic ops along the innermost loop dimension to `tileSize`.
// All outer dims are left untiled (size 0). This keeps vectorization from
// seeing the full tensor as a single vector (e.g. vector<512x512xf32>), which
// causes LLVM O3 to hang on large shapes. After tiling, the vectorizer only
// sees vector<tileSizexf32> strips that O3 can handle trivially.
struct LinalgGenericTilingPass
    : public mlir::PassWrapper<LinalgGenericTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgGenericTilingPass)

  int64_t tileSize;
  explicit LinalgGenericTilingPass(int64_t tile) : tileSize(tile) {}

  // tileUsingSCF constructs a new scf.for.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  llvm::StringRef getArgument() const override { return "linalg-tile-generic"; }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.generic ops along the innermost dimension";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::LinalgOp> ops;
    func.walk([&](mlir::linalg::LinalgOp op) {
      if (mlir::isa<mlir::linalg::GenericOp>(op))
        ops.push_back(op);
    });

    for (mlir::linalg::LinalgOp op : ops) {
      unsigned rank = op.getNumLoops();
      if (rank == 0)
        continue;

      // Tile all loops to tileSize. Tiling only the innermost loop leaves
      // outer dims untiled (e.g. 1024x8 for a 1024x1024 op), which the
      // vectorizer promotes to vector<1024x8xf32> — large enough to overflow
      // OMP worker thread stacks at sizes >= 512.
      llvm::SmallVector<int64_t> sizes(rank, tileSize);

      llvm::SmallVector<mlir::OpFoldResult> tileSizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), sizes);
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(tileSizes);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op.getOperation()), opts);
      if (mlir::failed(result)) {
        op->emitWarning("linalg-tile-generic: tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgOuterTileAndFusePass(int64_t tileM,
                                                              int64_t tileN) {
  return std::make_unique<LinalgOuterTileAndFusePass>(tileM, tileN);
}
std::unique_ptr<mlir::Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
std::unique_ptr<mlir::Pass> createVectorCleanupPass() {
  return std::make_unique<VectorCleanupPass>();
}
std::unique_ptr<mlir::Pass> createVectorTransposeLoweringPass() {
  return std::make_unique<VectorTransposeLoweringPass>();
}
std::unique_ptr<mlir::Pass> createAllocaScopeCleanupPass() {
  return std::make_unique<AllocaScopeCleanupPass>();
}
std::unique_ptr<mlir::Pass> createVectorContractToOuterProductPass() {
  return std::make_unique<VectorContractToOuterProductPass>();
}
std::unique_ptr<mlir::Pass> createLinalgGenericTilingPass() {
  return std::make_unique<LinalgGenericTilingPass>(8);
}
// The createLinalgMatmulBlocked*Pass factories live in
// passes/LinalgMatmulBlocked{Distribute,Tile,Pack,Kernel}.cpp.

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulTilingPass>(32, 32, 0, LoopType::ForallOp);
}
#endif

} // namespace mlir_edsl
