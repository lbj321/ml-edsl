#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
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
      // Blocked tiles are skipped: LinalgMatmulBlockedPass already placed
      // them inside its own loop nest and tiled their fill, so wrapping one in
      // another scf.forall here would double-tile it.
      func.walk([&](mlir::linalg::MatmulOp op) {
        if (op->hasAttr(mlir_edsl::kBlockedAttrName))
          return;
        consumer = op;
      });
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

// Lowers any static-shape linalg.matmul tile directly to vector.contract
// using standard 2D indexing maps {(m,k),(k,n),(m,n)}. This bypasses the
// linalg vectorizer which always produces a 3D double-broadcast form
// {(d0,d1,d2),(d0,d1,d2),(d0,d1)} that the OuterProduct lowering strategy
// cannot decompose into vector.outerproduct → vector.fma — instead it falls
// back to a per-output-element vector.extract -> arith.mulf ->
// vector.reduction -> vector.insert chain, with no FMA at all.
//
// Deliberately shape-agnostic: this pass's only job is to avoid ever
// producing the broken 3D-broadcast form. How efficiently a given (M, K, N)
// then lowers to hardware (e.g. whether it's a clean full-width AVX2 FMA, or
// a narrower/masked op for a shape that isn't a multiple of the target
// vector width) is left to VectorContractToOuterProductPass and the
// standard vector-to-llvm legalization further down the pipeline — not
// this pass's concern.
//
// Runs pre-bufferize (tensor semantics), so operands are tensors, not
// memref. vector.transfer_write on a tensor is functional — it returns a new
// tensor rather than mutating C in place — so the matmul's tensor result is
// replaced with that value instead of being erased as a pure side effect.
struct LinalgMatmulToContractPass
    : public mlir::PassWrapper<LinalgMatmulToContractPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulToContractPass)

  // Constructs new vector.transfer_read/write and vector.contraction ops.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-to-contract";
  }
  llvm::StringRef getDescription() const override {
    return "Lower static-shape linalg.matmul tiles to vector.contract with "
           "standard (m,k)x(k,n)->(m,n) indexing maps";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());
    mlir::MLIRContext *ctx = func->getContext();

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp matmul : matmuls) {
      // LinalgMatmulBlockedPass's vectorize=false toggle: leave the register
      // tile for convert-linalg-to-loops so a scalar baseline can be measured
      // against the vectorized one.
      if (matmul->hasAttr(mlir_edsl::kNoVectorizeAttrName))
        continue;

      mlir::Value A = matmul.getInputs()[0];
      mlir::Value B = matmul.getInputs()[1];
      mlir::Value C = matmul.getOutputs()[0];

      auto aType = mlir::dyn_cast<mlir::RankedTensorType>(A.getType());
      auto bType = mlir::dyn_cast<mlir::RankedTensorType>(B.getType());
      auto cType = mlir::dyn_cast<mlir::RankedTensorType>(C.getType());
      if (!aType || !bType || !cType)
        continue;

      // Dynamic-shape tiles (boundary tiles under dynamic tiling) fall
      // through to convert-linalg-to-loops for scalar lowering.
      if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
          !cType.hasStaticShape())
        continue;

      int64_t M = aType.getShape()[0];
      int64_t K = aType.getShape()[1];
      int64_t N = bType.getShape()[1];
      if (bType.getShape()[0] != K || cType.getShape() != llvm::ArrayRef<int64_t>{M, N})
        continue;

      mlir::Type elemType = aType.getElementType();
      if (bType.getElementType() != elemType || cType.getElementType() != elemType)
        continue;

      // vector.transfer_read's padding value must be a zero of elemType;
      // only float/integer scalars are supported here.
      auto floatType = mlir::dyn_cast<mlir::FloatType>(elemType);
      auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType);
      if (!floatType && !intType)
        continue;

      auto vecTypeA = mlir::VectorType::get({M, K}, elemType);
      auto vecTypeB = mlir::VectorType::get({K, N}, elemType);
      auto vecTypeC = mlir::VectorType::get({M, N}, elemType);
      mlir::Location loc = matmul.getLoc();
      rewriter.setInsertionPoint(matmul);

      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value pad =
          floatType ? rewriter.create<mlir::arith::ConstantOp>(
                          loc, elemType, mlir::FloatAttr::get(floatType, 0.0))
                    : rewriter.create<mlir::arith::ConstantOp>(
                          loc, elemType, mlir::IntegerAttr::get(intType, 0));

      llvm::SmallVector<bool> inBounds = {true, true};
      mlir::Value vA = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecTypeA, A, mlir::ValueRange{zero, zero}, pad, inBounds);
      mlir::Value vB = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecTypeB, B, mlir::ValueRange{zero, zero}, pad, inBounds);
      mlir::Value vC = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecTypeC, C, mlir::ValueRange{zero, zero}, pad, inBounds);

      // Standard matmul indexing: (m,n,k) -> (m,k) for A, (k,n) for B, (m,n) for C
      mlir::AffineExpr m, n, k;
      mlir::bindDims(ctx, m, n, k);
      auto indexingMaps = rewriter.getAffineMapArrayAttr({
          mlir::AffineMap::get(3, 0, {m, k}, ctx),
          mlir::AffineMap::get(3, 0, {k, n}, ctx),
          mlir::AffineMap::get(3, 0, {m, n}, ctx),
      });

      auto par = mlir::vector::IteratorType::parallel;
      auto red = mlir::vector::IteratorType::reduction;
      auto iterTypes = rewriter.getArrayAttr({
          mlir::vector::IteratorTypeAttr::get(ctx, par),
          mlir::vector::IteratorTypeAttr::get(ctx, par),
          mlir::vector::IteratorTypeAttr::get(ctx, red),
      });

      mlir::Value result = rewriter.create<mlir::vector::ContractionOp>(
          loc, vA, vB, vC, indexingMaps, iterTypes);

      auto newC = rewriter.create<mlir::vector::TransferWriteOp>(
          loc, result, C, mlir::ValueRange{zero, zero}, inBounds);

      rewriter.replaceOp(matmul, newC.getResult());
    }
  }
};

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
      // See the matching skip in LinalgMatmulToContractPass — the blocked
      // pass's vectorize=false toggle.
      if (op->hasAttr(mlir_edsl::kNoVectorizeAttrName))
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
// This is the key optimization for larger matmuls.
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
    mlir::vector::populateVectorReductionToContractPatterns(patterns);
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

// Tiles linalg.matmul into scf loops with configurable tile sizes and loop type.
//
// Inner vectorization (createLinalgMatmulTilingPass):
//   {8,8,8} ForOp — K tiling required for square vector<8x8x8> contracts that
//   the OuterProduct strategy decomposes into vector.outerproduct → vector.fma.
//
// Outer parallelism (createLinalgMatmulParallelTilingPass / createLinalgGPUMatmulTilingPass):
//   {64,64,0} or {32,32,0} ForallOp — K untiled to avoid reduction races.
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
      // Blocked tiles are already at the register tile size chosen by
      // MatmulStrategy; the 8x8x8 ForOp configuration would split a 4x16 tile
      // into 4x8 halves, and the 64x64 ForallOp one would re-tile it entirely.
      if (op->hasAttr(mlir_edsl::kBlockedAttrName))
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

// Cache-blocks the K (reduction) dimension of linalg.matmul ops whose K is
// large. Runs after the outer 64x64 parallel tiling (LinalgOuterTileAndFuse-
// Pass's fusion path and LinalgMatmulTilingPass's ForallOp fallback both
// leave K untiled/full-length on the matmul they place inside the
// scf.forall, to avoid reduction races) and before the 8x8x8 inner
// vectorization tiling. Without this, the inner loop streams the *entire*
// K-length A-row-panel and B-column-panel through every 64x64 output tile;
// once K is large those panels no longer fit L1/L2, and each tile re-fetches
// from L3/memory on every pass instead of reusing cache — this is what turns
// into a >400ms 2048x2048 matmul (17x slower than a naive NumPy baseline)
// despite the vectorized inner kernel.
//
// kKcTileSize=256 comes directly from this project's dev-machine cache sizes
// (32KiB L1d / core): with the existing 8-wide (Mr=Nr=8) register tile, a
// Kc x 8 panel of each operand is (Kc*8 + Kc*8)*4 bytes, which stays within
// half of L1d up to Kc=256.
struct LinalgMatmulKTilingPass
    : public mlir::PassWrapper<LinalgMatmulKTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulKTilingPass)

  static constexpr int64_t kKcTileSize = 256;

  llvm::StringRef getArgument() const override {
    return "linalg-tile-matmul-k";
  }
  llvm::StringRef getDescription() const override {
    return "Cache-block the K reduction dimension of large-K linalg.matmul ops";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) {
      // Only touch matmuls already placed inside an outer parallel tile (by
      // the epilogue-fusion pass or the parallel-tiling fallback) — those
      // are exactly the ones left with full-length K. A matmul with no
      // forall parent hasn't reached outer tiling yet and will shortly, so
      // skip it here rather than cache-block a shape that's about to change.
      if (!op->getParentOfType<mlir::scf::ForallOp>())
        return;

      // A blocked tile has K=1 and so is already below the threshold below;
      // the explicit skip is insurance for Step 3, where jc becomes an
      // scf.forall and the parent check above starts passing.
      if (op->hasAttr(mlir_edsl::kBlockedAttrName))
        return;

      auto lhsType = llvm::cast<mlir::ShapedType>(op.getInputs()[0].getType());
      int64_t k = lhsType.getShape().back();
      if (mlir::ShapedType::isDynamic(k) || k <= kKcTileSize)
        return;

      matmuls.push_back(op);
    });

    for (mlir::linalg::MatmulOp op : matmuls) {
      // Named variable required — setTileSizes captures a non-owning ArrayRef.
      llvm::SmallVector<mlir::OpFoldResult> tileSizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), {0, 0, kKcTileSize});
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(tileSizes);
      opts.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForOp);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op.getOperation()), opts);
      if (mlir::failed(result)) {
        op->emitWarning("linalg-tile-matmul-k: tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
    }
  }
};

// Row height used when strip-mining the C-initializing linalg.fill in
// blockMatmul. Any small constant works — the point is only to keep the
// vectorizer from seeing the whole MxN fill as one vector; the column extent
// is the register tile's NR so the stores line up with the microkernel.
constexpr int64_t kFillTileRows = 8;

// Tiles one level of a loop nest with scf.for, replacing `op` with the tiled
// result and returning the newly created tile op. Exactly one entry of `sizes`
// is expected to be non-zero per call: issuing one call per level is what
// fixes the resulting loop order, which a single multi-dimensional
// tile_using_for would not (see PLAN.md Stage 2 — [6,16,0] gives ir-outer/
// jr-inner, the opposite of what the macro-kernel wants).
static mlir::FailureOr<mlir::Operation *>
tileOneLevel(mlir::IRRewriter &rewriter, mlir::Operation *op,
             llvm::ArrayRef<int64_t> sizes,
             mlir::scf::SCFTilingOptions::LoopType loopType =
                 mlir::scf::SCFTilingOptions::LoopType::ForOp) {
  auto tilingOp = llvm::dyn_cast<mlir::TilingInterface>(op);
  if (!tilingOp)
    return mlir::failure();

  // Named variable required — setTileSizes captures a non-owning ArrayRef.
  llvm::SmallVector<mlir::OpFoldResult> tileSizes =
      mlir::getAsIndexOpFoldResult(op->getContext(), sizes);
  mlir::scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);
  opts.setLoopType(loopType);

  rewriter.setInsertionPoint(op);
  auto result = mlir::scf::tileUsingSCF(rewriter, tilingOp, opts);
  if (mlir::failed(result))
    return mlir::failure();
  if (result->tiledOps.size() != 1)
    return mlir::failure();

  rewriter.replaceOp(op, result->mergeResult.replacements);
  return result->tiledOps.front();
}

// BLIS-style cache and register blocking for a single linalg.matmul, on
// tensor semantics. Produces the loop nest
//
//   jc (N/NC) → pc (K/KC) → ic (M/MC) → jr (NC/NR) → ir (MC/MR) → k (KC/1)
//
// leaving an MR x NR x 1 linalg.matmul tile at the bottom. That tile is the
// microkernel, but this pass does not build it: LinalgMatmulToContractPass
// turns it into a vector.contract, LinalgVectorizationPass handles the fill,
// LoopInvariantSubsetHoisting lifts the C accumulator into the k-loop's
// iter_args as a vector<MRxNRxf32>, and VectorContractToOuterProductPass
// lowers the contract to vector.outerproduct → FMA. All four already exist in
// buildCPUPipeline.
//
// No operand packing yet (INTEGRATION.md Step 2): A is read as MR scalars from
// MR different rows and B has stride N between k steps, which is the bulk of
// the remaining gap to the experiment's ~100 GFLOPS.
static mlir::LogicalResult blockMatmul(mlir::IRRewriter &rewriter,
                                       mlir::linalg::MatmulOp op,
                                       const mlir_edsl::MatmulStrategy &s) {
  // The fill initializing C, captured before tiling rewrites the operand.
  // Handling it is not optional: a bare matmul's fill is normally fused into
  // the 64x64 forall by LinalgOuterTileAndFusePass, but that pass skips
  // blocked matmuls, finds no other consumer and returns — leaving the fill
  // untiled for LinalgVectorizationPass to turn into a single
  // vector<1024x1024xf32>, which convert-vector-to-scf makes a 4 MB stack
  // temporary.
  auto fill = op.getOutputs()[0].getDefiningOp<mlir::linalg::FillOp>();

  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  const std::array<std::pair<std::array<int64_t, 3>, LoopType>, 4> levels = {{
      // jc and ic fused into one 2-D scf.forall over MC x NC tiles of C. Both
      // are parallel dimensions in BLIS, and splitting only N makes every
      // thread sweep the whole of A, so total A traffic grows with the thread
      // count — measured at 1024^3, 8 bands of 128 was *slower* than 4 bands
      // of 256. Tiling both keeps each thread's working set to an MC-row band
      // of A and an NC-column band of B, which is what the 64x64 forall in the
      // old pipeline did.
      //
      // ForallToParallelLoop + ConvertSCFToOpenMP turn this into omp.parallel.
      // The rest stay serial: K must not be split across threads (it is the
      // reduction), and the register tiles are per-thread work.
      {{s.mc, s.nc, 0}, LoopType::ForallOp}, // ic x jc: C cache tile
      {{0, 0, s.kc}, LoopType::ForOp},       // pc: K cache block
      {{0, s.nr, 0}, LoopType::ForOp},       // jr: N register tile
      {{s.mr, 0, 0}, LoopType::ForOp},       // ir: M register tile
  }};

  mlir::Operation *current = op.getOperation();
  for (const auto &[sizes, loopType] : levels) {
    auto tiled = tileOneLevel(rewriter, current, sizes, loopType);
    if (mlir::failed(tiled))
      return mlir::failure();
    current = *tiled;
  }

  // Reduce the MR x NR x KC tile to the MR x NR x 1 register tile. With no
  // packing there is no un-transpose to fuse into this loop, so a plain tiling
  // suffices here.
  auto kernel = tileOneLevel(rewriter, current, {0, 0, 1});
  if (mlir::failed(kernel))
    return mlir::failure();

  (*kernel)->setAttr(mlir_edsl::kBlockedAttrName, rewriter.getUnitAttr());
  if (!s.vectorize)
    (*kernel)->setAttr(mlir_edsl::kNoVectorizeAttrName, rewriter.getUnitAttr());

  // Strip-mine the fill so it vectorizes into small row stores rather than one
  // whole-matrix vector. Step 3 replaces this with fusion into the jc forall,
  // which also parallelizes it.
  if (fill) {
    auto tiledFill = tileOneLevel(rewriter, fill.getOperation(),
                                  {kFillTileRows, s.nr});
    if (mlir::failed(tiledFill))
      return mlir::failure();
  }

  return mlir::success();
}

// BLIS-style blocking driver. See blockMatmul above for the loop nest and
// mlir_edsl::chooseStrategy for the fast-path guard; a matmul the guard
// rejects is left untouched for the existing 64x64 / 8x8x8 passes.
//
// Not part of buildCPUPipeline — it is reachable only from mlir-edsl-opt,
// where it composes with the rest of the pipeline as
//
//   mlir-edsl-opt in.mlir -linalg-matmul-blocked=mr=6,nr=16 -cpu-pipeline
//
// because the mlir_edsl.blocked marker keeps the superseded passes off its
// tiles.
struct LinalgMatmulBlockedPass
    : public mlir::PassWrapper<LinalgMatmulBlockedPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulBlockedPass)

  LinalgMatmulBlockedPass() = default;
  // Pass::Option is not copy-constructible, so the compiler cannot generate
  // the copy constructor clonePass() needs; copying the PassWrapper base
  // preserves the option values.
  LinalgMatmulBlockedPass(const LinalgMatmulBlockedPass &other)
      : mlir::PassWrapper<LinalgMatmulBlockedPass,
                          mlir::OperationPass<mlir::func::FuncOp>>(other) {}

  Option<int64_t> mr{*this, "mr",
                     llvm::cl::desc("Register tile rows (microkernel M)"),
                     llvm::cl::init(4)};
  Option<int64_t> nr{*this, "nr",
                     llvm::cl::desc("Register tile columns (microkernel N)"),
                     llvm::cl::init(16)};
  Option<int64_t> mc{*this, "mc",
                     llvm::cl::desc("Upper bound on the M cache block"),
                     llvm::cl::init(128)};
  Option<int64_t> nc{*this, "nc",
                     llvm::cl::desc("Upper bound on the N cache block"),
                     llvm::cl::init(256)};
  Option<int64_t> kc{*this, "kc",
                     llvm::cl::desc("Upper bound on the K cache block"),
                     llvm::cl::init(256)};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Let the register tile reach the vectorizing passes"),
      llvm::cl::init(true)};

  // tileUsingSCF constructs new scf.for loops.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked";
  }
  llvm::StringRef getDescription() const override {
    return "BLIS-style cache and register blocking of linalg.matmul into a "
           "jc/pc/ic/jr/ir loop nest over an MR x NR register tile";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    mlir_edsl::StrategyOverrides ov;
    ov.mr = mr;
    ov.nr = nr;
    ov.mcTarget = mc;
    ov.ncTarget = nc;
    ov.kcTarget = kc;
    ov.vectorize = vectorize;

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp op : matmuls) {
      auto strategy = mlir_edsl::chooseStrategy(op, ov);
      if (mlir::failed(strategy))
        continue; // Guard rejected it: leave it for the existing pipeline.
      if (mlir::failed(blockMatmul(rewriter, op, *strategy))) {
        // Do NOT signalPassFailure — a partially tiled matmul is still
        // correct, and the remaining passes can lower whatever is left.
        op->emitWarning("linalg-matmul-blocked: tiling failed, skipping op");
      }
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
std::unique_ptr<mlir::Pass> createLinalgMatmulToContractPass() {
  return std::make_unique<LinalgMatmulToContractPass>();
}
std::unique_ptr<mlir::Pass> createLinalgVectorizationPass() {
  return std::make_unique<LinalgVectorizationPass>();
}
std::unique_ptr<mlir::Pass> createVectorCleanupPass() {
  return std::make_unique<VectorCleanupPass>();
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
std::unique_ptr<mlir::Pass> createLinalgMatmulTilingPass() {
  return std::make_unique<LinalgMatmulTilingPass>(8, 8, 8);
}
std::unique_ptr<mlir::Pass> createLinalgMatmulParallelTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulTilingPass>(64, 64, 0, LoopType::ForallOp);
}
std::unique_ptr<mlir::Pass> createLinalgMatmulKTilingPass() {
  return std::make_unique<LinalgMatmulKTilingPass>();
}
std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPass() {
  return std::make_unique<LinalgMatmulBlockedPass>();
}

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulTilingPass>(32, 32, 0, LoopType::ForallOp);
}
#endif

} // namespace mlir_edsl
