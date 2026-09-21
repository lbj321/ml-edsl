//===- LinalgMatmulBlockedPass.cpp - BLIS-style matmul blocking -----------===//
//
// LinalgMatmulBlockedPass: the fast path for f32 matmuls whose shape
// chooseStrategy accepts. Tiles one linalg.matmul into a BLIS loop nest over
// an MR x NR register tile, optionally packing the A and B operands into
// contiguous panels, and leaves the register tile for the passes that build
// the microkernel (LinalgMatmulToContractPass, LinalgVectorizationPass,
// LoopInvariantSubsetHoisting, VectorContractToOuterProductPass — all in
// MLIRLoweringPasses.cpp and buildCPUPipeline).
//
// Split out of MLIRLoweringPasses.cpp, which holds the older matmul tiling,
// vectorization and fusion passes this one supersedes for accepted shapes.
// The shape guard and the marker attributes those passes check live in
// MatmulStrategy.h.
//
//===----------------------------------------------------------------------===//

#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <array>

namespace {

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

// Temporary id attached to the register tile so the packing phases can
// re-find it after applying function-wide patterns. Removed again once the
// microkernel is built; unlike mlir_edsl.blocked it means nothing to any
// other pass.
constexpr llvm::StringLiteral kTileIdAttrName = "mlir_edsl.blocked_tile_id";

// Rows of B packed per iteration of the B~ packing loop. 8 is one f32 ymm
// register: each iteration copies an 8 x NR block with two vector transfers.
constexpr int64_t kPackTileRows = 8;

// The register tile carrying `tileId`, or null. Raw Operation* handles do not
// survive a greedy pattern application, so every step that applies patterns
// re-acquires the tile through this.
static mlir::Operation *findTileById(mlir::func::FuncOp func, int64_t tileId) {
  mlir::Operation *found = nullptr;
  func.walk([&](mlir::linalg::MatmulOp op) {
    auto id = op->getAttrOfType<mlir::IntegerAttr>(kTileIdAttrName);
    if (id && id.getInt() == tileId) {
      found = op.getOperation();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return found;
}

// Runs one pattern set greedily over `func`, the same way VectorCleanupPass
// and AllocaScopeCleanupPass do. Kept deliberately narrow — no canonicalizer
// — because canonicalization at the wrong moment undoes two of the rewrites
// below (see the comments in packOperands).
static mlir::LogicalResult applyPatternSet(mlir::func::FuncOp func,
                                           mlir::RewritePatternSet &&patterns) {
  return mlir::applyPatternsGreedily(func, std::move(patterns));
}

// BLIS-style operand packing for one MR x NR x KC register tile.
//
// Pads the tile's A and/or B operand and hoists the pad out of the ir and jr
// loops, which turns it into a packing loop nest at the pc level plus a cheap
// slice at the use site:
//
//   B~ : tensor<(NC/NR) x KC x NR>     contiguous NR-wide rows
//   A~ : tensor<(MC/MR) x KC x MR>     k-major, via a [1, 0] transpose
//
// numLoops is 2, not the 3 the integration notes assumed: this pass fuses ic
// and jc into a single 2-D scf.forall, and hoistPaddingOnTensors only hoists
// across scf.for, so the loops between the tile and the pc body are just ir
// and jr. Both panels therefore land in the pc body, which is where BLIS
// wants them — B~ reused across ir, A~ across jr and ir.
//
// Returns the (re-found) register tile, whose operands now read from the
// packed panels.
static mlir::FailureOr<mlir::Operation *>
packOperands(mlir::IRRewriter &rewriter, mlir::func::FuncOp func,
             mlir::Operation *tile, const mlir_edsl::MatmulStrategy &s,
             int64_t tileId) {
  mlir::MLIRContext *ctx = func->getContext();

  // Every tiling level slices the previous level's slice, so the tile's
  // operands are defined *inside* the loops we want to hoist out of, and
  // hoistPaddingOnTensors refuses with "Source not defined outside of loops".
  // Collapsing the chains to a single slice of the original tensor is what
  // makes the hoist legal.
  {
    mlir::RewritePatternSet patterns(ctx);
    mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    if (mlir::failed(applyPatternSet(func, std::move(patterns))))
      return mlir::failure();
  }
  tile = findTileById(func, tileId);
  if (!tile)
    return mlir::failure();

  // Pad every iteration dimension. The slices are already exactly MR x NR x KC
  // so no element is actually added; the nofold flag is what forces the pad to
  // survive as a real copy, which *is* the packing. C is never padded — it is
  // accumulated in place across pc.
  auto linalgTile = llvm::dyn_cast<mlir::linalg::LinalgOp>(tile);
  if (!linalgTile)
    return mlir::failure();

  mlir::Attribute zero = rewriter.getZeroAttr(rewriter.getF32Type());
  mlir::linalg::LinalgPaddingOptions padOpts;
  padOpts.setPaddingValues({zero, zero, zero});
  padOpts.setPaddingDimensions({0, 1, 2});
  padOpts.setNofoldFlags({s.packA, s.packB, false});
  padOpts.setCopyBackOp(mlir::linalg::LinalgPaddingOptions::CopyBackOp::None);

  mlir::linalg::LinalgOp paddedOp;
  llvm::SmallVector<mlir::Value> replacements;
  llvm::SmallVector<mlir::tensor::PadOp> padOps;
  rewriter.setInsertionPoint(tile);
  if (mlir::failed(mlir::linalg::rewriteAsPaddedOp(
          rewriter, linalgTile, padOpts, paddedOp, replacements, padOps)))
    return mlir::failure();
  // rewriteAsPaddedOp clones the op onto the padded operands (carrying its
  // attributes, so the marker and id survive) but leaves the original in
  // place for the caller to replace.
  rewriter.replaceOp(tile, replacements);
  tile = paddedOp.getOperation();

  // B~: hoist across ir and jr with no transpose. The pad is a plain row copy,
  // so the packed panel keeps B's (k, n) order.
  if (s.packB) {
    auto padB = tile->getOperand(1).getDefiningOp<mlir::tensor::PadOp>();
    if (!padB)
      return mlir::failure();

    mlir::tensor::PadOp hoistedPad;
    llvm::SmallVector<mlir::linalg::TransposeOp> transposeOps;
    auto packed = mlir::linalg::hoistPaddingOnTensors(
        rewriter, padB, /*numLoops=*/2, /*transposeVector=*/{}, hoistedPad,
        transposeOps);
    if (mlir::failed(packed))
      return mlir::failure();
    rewriter.replaceOp(padB, *packed);

    // Tile the packing copy to 8 rows and vectorize each 8 x NR block.
    // Untiled, it vectorizes to one whole-panel vector that
    // convert-vector-to-llvm can only lower by scalar-unrolling it.
    // tensor.pad needs explicit vector sizes; it fails to vectorize without.
    auto padTile =
        tileOneLevel(rewriter, hoistedPad.getOperation(), {kPackTileRows, 0});
    if (mlir::failed(padTile))
      return mlir::failure();
    rewriter.setInsertionPoint(*padTile);
    // inputScalableVecDims must be given whenever inputVectorSizes is (the
    // vectorizer asserts on a length mismatch); nothing here is scalable.
    const llvm::SmallVector<bool> notScalable = {false, false};
    if (mlir::failed(mlir::linalg::vectorize(
            rewriter, *padTile, /*inputVectorSizes=*/{kPackTileRows, s.nr},
            notScalable)))
      return mlir::failure();

    // Fold the insert_slice into the transfer_write *now*. Left alone, the
    // transfer_read/transfer_write round trip through a fresh tensor.empty
    // folds back to insert_slice(extract_slice(B)), which bufferizes to a
    // strided memref.copy — a call to the memrefCopy runtime helper, which is
    // an undefined symbol in the JIT.
    mlir::RewritePatternSet patterns(ctx);
    mlir::tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
    if (mlir::failed(applyPatternSet(func, std::move(patterns))))
      return mlir::failure();

    tile = findTileById(func, tileId);
    if (!tile)
      return mlir::failure();
  }

  return tile;
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
// With packB, B's operand slice is padded and the pad hoisted out of the ir
// and jr loops, producing B~ = (NC/NR) x KC x NR built once per pc iteration,
// so the microkernel reads contiguous NR-wide rows instead of striding by N.
// packA does the same with a [1, 0] transpose, giving k-major A~ panels.
static mlir::LogicalResult blockMatmul(mlir::IRRewriter &rewriter,
                                       mlir::func::FuncOp func,
                                       mlir::linalg::MatmulOp op,
                                       const mlir_edsl::MatmulStrategy &s,
                                       int64_t tileId) {
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

  // Mark the tile now rather than after k-tiling: every later tiling clones
  // the op and carries the attributes along, the marker keeps the outer walk
  // in runOnOperation from picking this matmul up a second time, and the id
  // lets the packing phases below re-find the tile after they have applied
  // function-wide patterns (which invalidate raw Operation* handles).
  current->setAttr(mlir_edsl::kBlockedAttrName, rewriter.getUnitAttr());
  current->setAttr(kTileIdAttrName, rewriter.getI64IntegerAttr(tileId));
  if (!s.vectorize)
    current->setAttr(mlir_edsl::kNoVectorizeAttrName, rewriter.getUnitAttr());

  // Strip-mine the fill so it vectorizes into small row stores rather than one
  // whole-matrix vector. Done before packing, because packing applies
  // function-wide patterns that would invalidate the captured fill handle.
  // Step 3 replaces this with fusion into the jc forall, which also
  // parallelizes it.
  if (fill) {
    auto tiledFill = tileOneLevel(rewriter, fill.getOperation(),
                                  {kFillTileRows, s.nr});
    if (mlir::failed(tiledFill))
      return mlir::failure();
  }

  if (s.packA || s.packB) {
    auto packed = packOperands(rewriter, func, current, s, tileId);
    if (mlir::failed(packed))
      return mlir::failure();
    current = *packed;
  }

  // Reduce the MR x NR x KC tile to the MR x NR x 1 register tile.
  //
  // With packA there is an un-transpose in front of the tile (hoisting with a
  // transpose leaves one behind, see packOperands) which must move inside this
  // loop, or every microtile pays a full MR x KC copy. Fusing the producer
  // into the generated k-loop makes each k step transpose a 1 x MR row of A~
  // instead — i.e. exactly the contiguous read the packing was for.
  mlir::Operation *kernel = nullptr;
  if (s.packA) {
    auto tilingOp = llvm::dyn_cast<mlir::TilingInterface>(current);
    if (!tilingOp)
      return mlir::failure();

    llvm::SmallVector<mlir::OpFoldResult> tileSizes =
        mlir::getAsIndexOpFoldResult(current->getContext(), {0, 0, 1});
    mlir::scf::SCFTileAndFuseOptions opts;
    opts.tilingOptions.setTileSizes(tileSizes);
    rewriter.setInsertionPoint(current);
    auto result =
        mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp, opts);
    if (mlir::failed(result))
      return mlir::failure();

    llvm::SmallVector<mlir::Value> replacements;
    for (mlir::Value res : current->getResults())
      replacements.push_back(result->replacements.lookup(res));
    rewriter.replaceOp(current, replacements);

    kernel = findTileById(func, tileId);
    if (!kernel)
      return mlir::failure();
  } else {
    auto tiled = tileOneLevel(rewriter, current, {0, 0, 1});
    if (mlir::failed(tiled))
      return mlir::failure();
    kernel = *tiled;
  }

  kernel->removeAttr(kTileIdAttrName);
  return mlir::success();
}

// BLIS-style blocking driver. See blockMatmul above for the loop nest and
// mlir_edsl::chooseStrategy for the fast-path guard; a matmul the guard
// rejects is left untouched for the existing 64x64 / 8x8x8 passes.
//
// Runs first in buildCPUPipeline with its default options. A non-default
// strategy can be composed ahead of the pipeline from mlir-edsl-opt as
//
//   mlir-edsl-opt in.mlir -linalg-matmul-blocked=mr=6,nr=16 -cpu-pipeline
//
// because the mlir_edsl.blocked marker keeps the superseded passes — and the
// pipeline's own second run of this pass — off its tiles.
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
  Option<bool> packA{*this, "pack_a",
                     llvm::cl::desc("Pack A into k-major MR-wide panels"),
                     llvm::cl::init(false)};
  Option<bool> packB{*this, "pack_b",
                     llvm::cl::desc("Pack B into contiguous NR-wide panels"),
                     llvm::cl::init(true)};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Let the register tile reach the vectorizing passes"),
      llvm::cl::init(true)};

  // tileUsingSCF constructs new scf.for loops; packing additionally builds
  // tensor.pad/empty, linalg.transpose and vector transfer ops.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect>();
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
    ov.packA = packA;
    ov.packB = packB;
    ov.vectorize = vectorize;

    // Re-walk for each matmul rather than collecting them up front: the
    // packing phases apply function-wide patterns, which can invalidate any
    // Operation* handle held across them. Ops that have been handled carry
    // mlir_edsl.blocked (which chooseStrategy also rejects); ops the guard
    // rejected get a temporary skip marker so this loop terminates.
    constexpr llvm::StringLiteral kSkipAttrName = "mlir_edsl.blocked_skip";
    int64_t tileId = 0;

    while (true) {
      mlir::linalg::MatmulOp next;
      func.walk([&](mlir::linalg::MatmulOp op) {
        if (op->hasAttr(mlir_edsl::kBlockedAttrName) ||
            op->hasAttr(kSkipAttrName))
          return mlir::WalkResult::advance();
        next = op;
        return mlir::WalkResult::interrupt();
      });
      if (!next)
        break;

      auto strategy = mlir_edsl::chooseStrategy(next, ov);
      if (mlir::failed(strategy)) {
        // Guard rejected it: leave it for the existing pipeline.
        next->setAttr(kSkipAttrName, rewriter.getUnitAttr());
        continue;
      }
      if (mlir::failed(blockMatmul(rewriter, func, next, *strategy, tileId))) {
        // Do NOT signalPassFailure — a partially tiled matmul is still
        // correct, and the remaining passes can lower whatever is left. The
        // op may be gone by now (a failure part-way through packing), so
        // diagnose against the function.
        func->emitWarning(
            "linalg-matmul-blocked: blocking failed, leaving matmul as-is");
      }
      ++tileId;
    }

    func.walk([&](mlir::linalg::MatmulOp op) { op->removeAttr(kSkipAttrName); });
  }
};
} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPass() {
  return std::make_unique<LinalgMatmulBlockedPass>();
}

} // namespace mlir_edsl
