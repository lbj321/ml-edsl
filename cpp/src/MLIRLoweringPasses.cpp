#include "mlir_edsl/MLIRLoweringPasses.h"

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
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

// Marks the linalg.generic that linalg::pack rewrites a linalg.matmul into,
// so later passes in the experimental packed pipeline (tiling, vectorization)
// can re-find exactly that op across separate PassManager stages without
// walking module-wide state or guessing from shape/iterator-type signatures
// alone — same idea as LinalgBuilder.cpp's "relu" library_call marker (see
// LinalgOuterTileAndFusePass above). Each stage that further rewrites the
// marked op must re-attach this attribute to its replacement so the next
// stage can still find it; LinalgMatmulPackPass sets it once here, the
// tiling pass below re-attaches it after every tileUsingSCF call.
constexpr llvm::StringLiteral kPackedMatmulMarker = "mlir_edsl.packed_matmul";

// Experimental: packs each linalg.matmul into 32x32x32 blocks via
// linalg.pack/unpack, the first stage of the linalg.pack-based lowering
// pipeline being ported from experiments/matmul-per-tile-packing/pack.mlir
// (see that file for the transform-dialect version this mirrors). Not yet
// wired into addCPUPasses — this pass is being built and verified stage by
// stage before it replaces any part of the default pipeline.
//
// Runs pre-bufferize (tensor semantics), like LinalgMatmulToContractPass
// above. linalg::pack rewrites the matched matmul (and its A/B/acc operands)
// in place: linalg.pack ops materialize the packed A/B, the matmul itself
// becomes a 6-loop linalg.generic over the packed blocks, and a trailing
// linalg.unpack restores the original (unpacked) output shape.
struct LinalgMatmulPackPass
    : public mlir::PassWrapper<LinalgMatmulPackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulPackPass)

  int64_t packM, packN, packK;

  explicit LinalgMatmulPackPass(int64_t m = 32, int64_t n = 32, int64_t k = 32)
      : packM(m), packN(n), packK(k) {}

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-pack";
  }
  llvm::StringRef getDescription() const override {
    return "Experimental: pack linalg.matmul into blocked layout via "
           "linalg.pack/unpack";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());
    mlir::MLIRContext *ctx = func->getContext();

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp matmulOp : matmuls) {
      llvm::SmallVector<mlir::OpFoldResult> packedSizes =
          mlir::getAsIndexOpFoldResult(ctx, {packM, packN, packK});
      rewriter.setInsertionPoint(matmulOp);
      auto packResult = mlir::linalg::pack(
          rewriter, llvm::cast<mlir::linalg::LinalgOp>(matmulOp.getOperation()),
          packedSizes);
      if (mlir::failed(packResult)) {
        matmulOp->emitWarning("linalg-matmul-pack: packing failed, skipping op");
        continue;
      }
      packResult->packedLinalgOp->setAttr(kPackedMatmulMarker,
                                          rewriter.getUnitAttr());
    }
  }
};

// Experimental: tiles the linalg.generic LinalgMatmulPackPass produced
// (found via kPackedMatmulMarker) by one configurable level, ported from
// experiments/matmul-per-tile-packing/tile.mlir. Reused for all three of
// tile.mlir's nested tiling levels via the create*() factories below — same
// "one struct, several instantiations" idiom LinalgMatmulTilingPass already
// uses for the non-packed pipeline's outer/inner tiling. Each level is its
// own separate PassManager stage (not folded into one pass), sandwiched with
// the standard canonicalize/cse passes at the addCPUPasses call site once
// wired in, matching every other stage in this pipeline.
//
// Re-attaches kPackedMatmulMarker to the newly tiled (smaller) op after each
// tileUsingSCF call, so the next tiling-level pass (a fresh func.walk, since
// this is a separate PassManager stage) can still find it.
//
// Each tileUsingSCF call on a tile-by-one dim leaves unit-extent dims behind
// (this is a 6D op), so `foldUnitDims` folds them via ReassociativeReshape.
// tile.mlir also runs its own tiling_canonicalization patterns right after
// each tile call (narrower than the general canonicalizer, and not
// something createCanonicalizerPass applies on its own) — both pattern sets
// applied directly here via applyPatternsGreedily, same mechanism
// VectorCleanupPass and VectorContractToOuterProductPass already use above,
// rather than pulling in a nested PassManager for the standard canonicalizer
// (that stays a sibling pass at the call site instead).
struct LinalgMatmulPackedTilingPass
    : public mlir::PassWrapper<LinalgMatmulPackedTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulPackedTilingPass)
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;

  llvm::SmallVector<int64_t, 6> tileSizes;
  LoopType loopType;
  bool foldUnitDims;
  std::string argument;
  std::string description;

  LinalgMatmulPackedTilingPass(llvm::ArrayRef<int64_t> sizes, LoopType lt,
                               bool foldUnits, llvm::StringRef arg,
                               llvm::StringRef desc)
      : tileSizes(sizes.begin(), sizes.end()), loopType(lt),
        foldUnitDims(foldUnits), argument(arg), description(desc) {}

  llvm::StringRef getArgument() const override { return argument; }
  llvm::StringRef getDescription() const override { return description; }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::Operation *> targets;
    func.walk([&](mlir::linalg::GenericOp op) {
      if (op->hasAttr(kPackedMatmulMarker))
        targets.push_back(op.getOperation());
    });

    for (mlir::Operation *op : targets) {
      llvm::SmallVector<mlir::OpFoldResult> ofrSizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), tileSizes);
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(ofrSizes);
      opts.setLoopType(loopType);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op), opts);
      if (mlir::failed(result)) {
        op->emitWarning(getArgument() + ": tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
      mlir::Operation *tiledOp = result->tiledOps.back();

      // Tiling-by-one leaves unit-extent dims on `tiledOp` (this is a 6D
      // op). Fold them via a direct dropUnitDims() call, scoped to just this
      // op, rather than a blanket applyPatternsGreedily: dropUnitDims
      // *replaces* the op with a new one (a collapsed reshape of it), and a
      // greedy rewrite gives no handle back to that replacement — the
      // kPackedMatmulMarker tag below would be lost the moment the pattern
      // fires, leaving the next tiling-level pass's func.walk with nothing
      // to find. Calling dropUnitDims ourselves gets the replacement handle
      // directly instead.
      //
      // dropUnitDims() itself only builds the replacement op (stealing
      // genericOp's region via inlineRegionBefore) and returns it — unlike
      // tileUsingSCF above, it does NOT call replaceOp on our behalf (only
      // its OpRewritePattern wrapper, used by the greedy-pattern path, does
      // that). Skipping the explicit replaceOp here would leave genericOp
      // alive in the IR with its region already stolen — an ill-formed op
      // that segfaults the next pass that walks over it.
      if (foldUnitDims) {
        if (auto genericOp =
                llvm::dyn_cast<mlir::linalg::GenericOp>(tiledOp)) {
          rewriter.setInsertionPoint(genericOp);
          mlir::linalg::ControlDropUnitDims options;
          auto dropResult =
              mlir::linalg::dropUnitDims(rewriter, genericOp, options);
          if (mlir::succeeded(dropResult)) {
            rewriter.replaceOp(genericOp, dropResult->replacements);
            tiledOp = dropResult->resultOp.getOperation();
          }
        }
      }

      tiledOp->setAttr(kPackedMatmulMarker, rewriter.getUnitAttr());
    }

    // tiling_canonicalization patterns only target affine/memref/scf/
    // tensor.cast ops (see populateLinalgTilingCanonicalizationPatterns),
    // never linalg.generic, so a blanket applyPatternsGreedily here can't
    // touch — or drop the marker from — the ops tagged above.
    mlir::RewritePatternSet tilingPatterns(func->getContext());
    mlir::linalg::populateLinalgTilingCanonicalizationPatterns(tilingPatterns);
    (void)mlir::applyPatternsGreedily(func, std::move(tilingPatterns));
  }
};

// Experimental: vectorizes the linalg.generic LinalgMatmulPackedTilingPass's
// final (8x8x8) tiling level produced (found via kPackedMatmulMarker),
// ported from experiments/matmul-per-tile-packing/vectorize.mlir's first
// sub-step (transform.structured.vectorize).
//
// Scoped to the marked op only, unlike the existing (shared)
// LinalgVectorizationPass below, which is deliberately NOT reused here: it
// walks every remaining linalg op in the function, and running it this
// early would also vectorize not-yet-tiled epilogue generics (relu/bias),
// causing the SSA blowup LinalgGenericTilingPass exists to prevent (see its
// own comment). By the time the *shared* LinalgVectorizationPass runs later
// in the pipeline, this op is already gone (replaced by vector ops below),
// so it never sees — or re-vectorizes — it.
//
// Once vectorized, the op is a sequence of vector.transfer_read/
// arith.mulf/vector.multi_reduction/transfer_write ops, not a single op
// anymore, so there is nothing left to re-tag with kPackedMatmulMarker —
// the two passes below operate function-wide on whatever vector ops exist
// at that point, which at this stage in the pipeline can only be the ones
// this pass just created (every other linalg op is still untouched linalg
// form; the shared vectorization stage hasn't run yet).
struct LinalgMatmulPackedVectorizePass
    : public mlir::PassWrapper<LinalgMatmulPackedVectorizePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulPackedVectorizePass)
  llvm::StringRef getArgument() const override {
    return "linalg-matmul-packed-vectorize";
  }
  llvm::StringRef getDescription() const override {
    return "Experimental: vectorize the packed matmul's 8x8x8 tiled generic";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::Operation *> targets;
    func.walk([&](mlir::linalg::GenericOp op) {
      if (op->hasAttr(kPackedMatmulMarker))
        targets.push_back(op.getOperation());
    });

    for (mlir::Operation *op : targets) {
      rewriter.setInsertionPoint(op);
      if (mlir::failed(mlir::linalg::vectorize(rewriter, op)))
        op->emitWarning(
            "linalg-matmul-packed-vectorize: vectorization failed, "
            "skipping op");
    }
  }
};

// Experimental: paired with LinalgMatmulPackedVectorizePass above, ported
// from vectorize.mlir's second sub-step. linalg::vectorize always lowers a
// reduction to arith.mulf + vector.multi_reduction with BROADCAST-shaped
// operands (see LinalgMatmulToContractPass's own comment for why the
// non-packed pipeline avoids ever sending linalg.matmul through
// linalg::vectorize in the first place — this pass exists precisely because
// the packed path does). transfer_permutation_patterns strips the broadcast
// dim from each transfer_read into an explicit vector.broadcast;
// reduction_to_contract then folds mulf+multi_reduction into vector.contract
// AND reduces its rank using that explicit broadcast — both needed
// together, in the same greedy pattern application, or the contract keeps
// its broadcast shape and VectorContractToOuterProductPass's OuterProduct
// lowering (reused as-is right after this pass, see addCPUPasses) silently
// falls back to a scalar expansion instead of vector.fma.
//
// Unlike the existing (shared) VectorCleanupPass below — which only needs
// reduction_to_contract, since production never routes matmul through
// linalg::vectorize — this pass is not a reuse of it, it is a distinct
// pattern set required specifically because this path does.
struct LinalgMatmulPackedReductionToContractPass
    : public mlir::PassWrapper<LinalgMatmulPackedReductionToContractPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulPackedReductionToContractPass)
  llvm::StringRef getArgument() const override {
    return "linalg-matmul-packed-reduction-to-contract";
  }
  llvm::StringRef getDescription() const override {
    return "Experimental: fold mulf+multi_reduction into vector.contract "
           "for the packed matmul path (with transfer_permutation "
           "lowering)";
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

// Experimental: rewrites every linalg.pack op into pad + expand_shape +
// transpose, ported from experiments/matmul-per-tile-packing/
// pack_lowering.mlir's first sub-step (transform.structured.lower_pack).
// Runs pre-bufferize, like every other pass in this experimental path.
//
// Unlike LinalgMatmulPackedTilingPass's dropUnitDims usage above,
// linalg::lowerPack sets its own insertion point and calls
// rewriter.replaceOp on packOp internally (see Transforms.cpp) — safe to
// call directly, no extra bookkeeping needed here.
//
// Walks the whole function for linalg.pack ops directly by type, not via
// kPackedMatmulMarker: pack ops are a type only LinalgMatmulPackPass
// produces in this pipeline, so there is no ambiguity to resolve the way
// there was for linalg.generic (which many unrelated ops can also be).
struct LinalgMatmulPackedLowerPackPass
    : public mlir::PassWrapper<LinalgMatmulPackedLowerPackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulPackedLowerPackPass)
  llvm::StringRef getArgument() const override {
    return "linalg-matmul-packed-lower-pack";
  }
  llvm::StringRef getDescription() const override {
    return "Experimental: lower linalg.pack to pad+expand_shape+transpose";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::PackOp> packOps;
    func.walk([&](mlir::linalg::PackOp op) { packOps.push_back(op); });

    for (mlir::linalg::PackOp packOp : packOps) {
      if (mlir::failed(mlir::linalg::lowerPack(rewriter, packOp)))
        packOp->emitWarning(
            "linalg-matmul-packed-lower-pack: lowering failed, skipping op");
    }
  }
};

// Experimental: rewrites every linalg.unpack op into empty + transpose +
// collapse_shape + extract_slice, ported from pack_lowering.mlir's second
// sub-step (transform.structured.lower_unpack). Same reasoning as
// LinalgMatmulPackedLowerPackPass above (direct call is safe, no marker
// needed — linalg.unpack is likewise unambiguous by type).
//
// This is the last experimental pass in the pipeline: after this, only
// pad/expand_shape/transpose/collapse_shape/extract_slice and the vector
// ops from the earlier vectorization stage remain, which is exactly the
// shape the *shared* bufferize -> dealloc -> forall-to-omp -> LLVM-dialect
// tail (already in addCPUPasses, unchanged) expects to consume.
struct LinalgMatmulPackedLowerUnpackPass
    : public mlir::PassWrapper<LinalgMatmulPackedLowerUnpackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulPackedLowerUnpackPass)
  llvm::StringRef getArgument() const override {
    return "linalg-matmul-packed-lower-unpack";
  }
  llvm::StringRef getDescription() const override {
    return "Experimental: lower linalg.unpack to "
           "empty+transpose+collapse_shape+extract_slice";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::UnPackOp> unpackOps;
    func.walk([&](mlir::linalg::UnPackOp op) { unpackOps.push_back(op); });

    for (mlir::linalg::UnPackOp unpackOp : unpackOps) {
      if (mlir::failed(mlir::linalg::lowerUnPack(rewriter, unpackOp)))
        unpackOp->emitWarning("linalg-matmul-packed-lower-unpack: lowering "
                              "failed, skipping op");
    }
  }
};

struct LinalgVectorizationPass
    : public mlir::PassWrapper<LinalgVectorizationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgVectorizationPass)
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
        patterns, mlir::vector::VectorContractLowering::OuterProduct);
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
std::unique_ptr<mlir::Pass> createLinalgMatmulPackPass(int64_t packM,
                                                       int64_t packN,
                                                       int64_t packK) {
  return std::make_unique<LinalgMatmulPackPass>(packM, packN, packK);
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedForallTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulPackedTilingPass>(
      llvm::ArrayRef<int64_t>{1, 1, 0, 0, 0, 0}, LoopType::ForallOp,
      /*foldUnits=*/true, "linalg-matmul-packed-tile-forall",
      "Experimental: tile packed matmul M/N block-grid dims via scf.forall");
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedKTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulPackedTilingPass>(
      llvm::ArrayRef<int64_t>{1, 0, 0, 0}, LoopType::ForOp,
      /*foldUnits=*/true, "linalg-matmul-packed-tile-k",
      "Experimental: tile packed matmul K block-grid dim via scf.for");
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedInnerTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulPackedTilingPass>(
      llvm::ArrayRef<int64_t>{8, 8, 8}, LoopType::ForOp,
      /*foldUnits=*/false, "linalg-matmul-packed-tile-inner",
      "Experimental: tile packed matmul remaining 32x32x32 block to 8x8x8");
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedVectorizePass() {
  return std::make_unique<LinalgMatmulPackedVectorizePass>();
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedReductionToContractPass() {
  return std::make_unique<LinalgMatmulPackedReductionToContractPass>();
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedLowerPackPass() {
  return std::make_unique<LinalgMatmulPackedLowerPackPass>();
}
std::unique_ptr<mlir::Pass> createLinalgMatmulPackedLowerUnpackPass() {
  return std::make_unique<LinalgMatmulPackedLowerUnpackPass>();
}

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulTilingPass>(32, 32, 0, LoopType::ForallOp);
}
#endif

} // namespace mlir_edsl
