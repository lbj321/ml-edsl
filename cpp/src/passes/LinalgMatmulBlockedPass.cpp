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

// The tile op one tiling level produced and the loops generated around it,
// outermost first.
struct TiledLevel {
  mlir::Operation *op;
  llvm::SmallVector<mlir::LoopLikeOpInterface> loops;
};

// Tiles one level of a loop nest with scf.for, replacing `op` with the tiled
// result. Exactly one entry of `sizes` is expected to be non-zero per call:
// issuing one call per level is what fixes the resulting loop order, which a
// single multi-dimensional tile_using_for would not (see PLAN.md Stage 2 —
// [6,16,0] gives ir-outer/jr-inner, the opposite of what the macro-kernel
// wants).
static mlir::FailureOr<TiledLevel>
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
  return TiledLevel{result->tiledOps.front(), std::move(result->loops)};
}

// The BLIS loop nest around one register tile. The loop handles stay valid
// through packing because C is never padded: padding it would make
// hoistPaddingOnTensors rebuild the loops to carry the pad through iter_args.
struct BlockedNest {
  mlir::scf::ForallOp forall; // ic x jc
  mlir::scf::ForOp pc, jr, ir;
  mlir::Operation *tile; // MR x NR x KC, before k-tiling
};

// Number of scf.for loops enclosing `op` strictly inside `outer`.
static int64_t loopsBetween(mlir::Operation *op, mlir::Operation *outer) {
  int64_t count = 0;
  for (mlir::Operation *p = op->getParentOp(); p && p != outer;
       p = p->getParentOp())
    if (llvm::isa<mlir::scf::ForOp>(p))
      ++count;
  return count;
}
// Rows of B packed per iteration of the B~ packing loop. 8 is one f32 ymm
// register: each iteration copies an 8 x NR block with two vector transfers.
constexpr int64_t kPackTileRows = 8;

// Runs one pattern set greedily over `ops` only, plus the ops the patterns
// create. Scoping is what lets the caller keep holding handles: ops outside
// the set are never folded, rewritten or erased as dead. Kept deliberately
// narrow — no canonicalizer — because canonicalization at the wrong moment
// undoes two of the rewrites below (see packB).
static mlir::LogicalResult applyPatternSetTo(llvm::ArrayRef<mlir::Operation *> ops,
                                             mlir::RewritePatternSet &&patterns) {
  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
  return mlir::applyOpPatternsGreedily(ops, std::move(patterns), config);
}

// Tiles `op` into the BLIS loop nest
//
//   jc (N/NC) → pc (K/KC) → ic (M/MC) → jr (NC/NR) → ir (MC/MR)
//
// leaving an MR x NR x KC tile, and strip-mines the fill initializing C.
static mlir::FailureOr<BlockedNest>
tileNest(mlir::IRRewriter &rewriter, mlir::linalg::MatmulOp op,
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
  std::array<mlir::Operation *, 4> levelLoops{};
  for (auto [i, level] : llvm::enumerate(levels)) {
    const auto &[sizes, loopType] = level;
    auto tiled = tileOneLevel(rewriter, current, sizes, loopType);
    if (mlir::failed(tiled) || tiled->loops.size() != 1)
      return mlir::failure();
    current = tiled->op;
    levelLoops[i] = tiled->loops.front().getOperation();
  }

  BlockedNest nest;
  nest.forall = llvm::dyn_cast<mlir::scf::ForallOp>(levelLoops[0]);
  nest.pc = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[1]);
  nest.jr = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[2]);
  nest.ir = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[3]);
  nest.tile = current;
  if (!nest.forall || !nest.pc || !nest.jr || !nest.ir)
    return mlir::failure();

  // Marked now rather than after k-tiling: every later tiling clones the op
  // and carries the attributes along.
  current->setAttr(mlir_edsl::kBlockedAttrName, rewriter.getUnitAttr());
  if (!s.vectorize)
    current->setAttr(mlir_edsl::kNoVectorizeAttrName, rewriter.getUnitAttr());

  // Strip-mine the fill so it vectorizes into small row stores rather than one
  // whole-matrix vector. Step 3 replaces this with fusion into the jc forall,
  // which also parallelizes it.
  if (fill) {
    auto tiledFill = tileOneLevel(rewriter, fill.getOperation(),
                                  {kFillTileRows, s.nr});
    if (mlir::failed(tiledFill))
      return mlir::failure();
  }
  return nest;
}

// Collapses the slice chains feeding the tile into single slices of the
// original tensors. Every tiling level slices the previous level's slice, so
// the tile's operands are defined *inside* the loops we want to hoist out of,
// and hoistPaddingOnTensors refuses with "Source not defined outside of
// loops". Seeding the whole chain also lets the driver erase the outer slices
// once they are dead.
static mlir::LogicalResult mergeSliceChains(BlockedNest &nest) {
  llvm::SmallVector<mlir::Operation *> slices;
  nest.forall.walk(
      [&](mlir::tensor::ExtractSliceOp op) { slices.push_back(op); });
  mlir::RewritePatternSet patterns(nest.forall->getContext());
  mlir::tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
  return applyPatternSetTo(slices, std::move(patterns));
}

// Pads every iteration dimension of the tile. The slices are already exactly
// MR x NR x KC so no element is actually added; the nofold flag is what forces
// the pad to survive as a real copy, which *is* the packing. C is never padded
// — it is accumulated in place across pc.
static mlir::LogicalResult padTile(mlir::IRRewriter &rewriter,
                                   BlockedNest &nest,
                                   const mlir_edsl::MatmulStrategy &s) {
  auto linalgTile = llvm::dyn_cast<mlir::linalg::LinalgOp>(nest.tile);
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
  rewriter.setInsertionPoint(nest.tile);
  if (mlir::failed(mlir::linalg::rewriteAsPaddedOp(
          rewriter, linalgTile, padOpts, paddedOp, replacements, padOps)))
    return mlir::failure();
  // rewriteAsPaddedOp clones the op onto the padded operands (carrying its
  // attributes, so the marker survives) but leaves the original in place for
  // the caller to replace.
  rewriter.replaceOp(nest.tile, replacements);
  nest.tile = paddedOp.getOperation();
  return mlir::success();
}

// B~ = (NC/NR) x KC x NR: hoists B's pad to the pc body with no transpose.
// The pad is a plain row copy, so the packed panel keeps B's (k, n) order and
// the microkernel reads contiguous NR-wide rows instead of striding by N.
static mlir::LogicalResult packB(mlir::IRRewriter &rewriter, BlockedNest &nest,
                                 const mlir_edsl::MatmulStrategy &s) {
  auto padB = nest.tile->getOperand(1).getDefiningOp<mlir::tensor::PadOp>();
  if (!padB)
    return mlir::failure();

  mlir::tensor::PadOp hoistedPad;
  llvm::SmallVector<mlir::linalg::TransposeOp> transposeOps;
  auto packed = mlir::linalg::hoistPaddingOnTensors(
      rewriter, padB, loopsBetween(nest.tile, nest.pc),
      /*transposeVector=*/{}, hoistedPad, transposeOps);
  if (mlir::failed(packed))
    return mlir::failure();
  rewriter.replaceOp(padB, *packed);

  // Tile the packing copy to 8 rows and vectorize each 8 x NR block.
  // Untiled, it vectorizes to one whole-panel vector that
  // convert-vector-to-llvm can only lower by scalar-unrolling it.
  // tensor.pad needs explicit vector sizes; it fails to vectorize without.
  auto padTile =
      tileOneLevel(rewriter, hoistedPad.getOperation(), {kPackTileRows, 0});
  if (mlir::failed(padTile) || padTile->loops.size() != 1)
    return mlir::failure();
  rewriter.setInsertionPoint(padTile->op);
  // inputScalableVecDims must be given whenever inputVectorSizes is (the
  // vectorizer asserts on a length mismatch); nothing here is scalable.
  const llvm::SmallVector<bool> notScalable = {false, false};
  if (mlir::failed(mlir::linalg::vectorize(
          rewriter, padTile->op, /*inputVectorSizes=*/{kPackTileRows, s.nr},
          notScalable)))
    return mlir::failure();

  // Fold the insert_slice into the transfer_write *now*. Left alone, the
  // transfer_read/transfer_write round trip through a fresh tensor.empty
  // folds back to insert_slice(extract_slice(B)), which bufferizes to a
  // strided memref.copy — a call to the memrefCopy runtime helper, which is
  // an undefined symbol in the JIT.
  llvm::SmallVector<mlir::Operation *> packLoopOps;
  padTile->loops.front()->walk([&](mlir::Operation *op) {
    if (op != padTile->loops.front().getOperation())
      packLoopOps.push_back(op);
  });
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::tensor::populateFoldTensorSubsetIntoVectorTransferPatterns(patterns);
  return applyPatternSetTo(packLoopOps, std::move(patterns));
}

// A~ = (MC/MR) x KC x MR: hoists A's pad to the pc body with a [1, 0]
// transpose, so each k step reads a contiguous MR-element row instead of MR
// scalars from MR different rows.
static mlir::LogicalResult packA(mlir::IRRewriter &rewriter, BlockedNest &nest) {
  auto padA = nest.tile->getOperand(0).getDefiningOp<mlir::tensor::PadOp>();
  if (!padA)
    return mlir::failure();

  mlir::tensor::PadOp hoistedPad;
  llvm::SmallVector<mlir::linalg::TransposeOp> transposeOps;
  auto packed = mlir::linalg::hoistPaddingOnTensors(
      rewriter, padA, loopsBetween(nest.tile, nest.pc),
      /*transposeVector=*/{1, 0}, hoistedPad, transposeOps);
  if (mlir::failed(packed))
    return mlir::failure();
  rewriter.replaceOp(padA, *packed);

  // transposeOps[0] is the packing transpose, [1] the un-transpose put back
  // in front of the tile.
  if (transposeOps.empty())
    return mlir::failure();
  mlir::linalg::TransposeOp packTranspose = transposeOps.front();

  // A's pad is zero-width — the transpose is the packing copy — and
  // tensor.pad is not destination-style, so it cannot be fused with the
  // fusion utilities. Decomposing it lets the transpose read A directly.
  mlir::RewritePatternSet patterns(rewriter.getContext());
  mlir::linalg::populateDecomposePadPatterns(patterns);
  if (mlir::failed(
          applyPatternSetTo({hoistedPad.getOperation()}, std::move(patterns))))
    return mlir::failure();

  // Tiled to 8 rows for the same reason as B's copy; LinalgVectorizationPass
  // vectorizes the tiles later.
  auto trTile = tileOneLevel(rewriter, packTranspose.getOperation(),
                             {kPackTileRows, 0});
  return mlir::success(mlir::succeeded(trTile));
}

// Reduces the MR x NR x KC tile to the MR x NR x 1 register tile.
//
// With packA there is an un-transpose in front of the tile (hoisting with a
// transpose leaves one behind, see packA) which must move inside this loop,
// or every microtile pays a full MR x KC copy. Fusing the producer into the
// generated k-loop makes each k step transpose a 1 x MR row of A~ instead —
// i.e. exactly the contiguous read the packing was for.
static mlir::LogicalResult tileK(mlir::IRRewriter &rewriter, BlockedNest &nest,
                                 const mlir_edsl::MatmulStrategy &s) {
  if (!s.packA) {
    auto tiled = tileOneLevel(rewriter, nest.tile, {0, 0, 1});
    if (mlir::failed(tiled))
      return mlir::failure();
    nest.tile = tiled->op;
    return mlir::success();
  }

  auto tilingOp = llvm::dyn_cast<mlir::TilingInterface>(nest.tile);
  if (!tilingOp)
    return mlir::failure();

  llvm::SmallVector<mlir::OpFoldResult> tileSizes =
      mlir::getAsIndexOpFoldResult(rewriter.getContext(), {0, 0, 1});
  mlir::scf::SCFTileAndFuseOptions opts;
  opts.tilingOptions.setTileSizes(tileSizes);
  rewriter.setInsertionPoint(nest.tile);
  auto result =
      mlir::scf::tileConsumerAndFuseProducersUsingSCF(rewriter, tilingOp, opts);
  if (mlir::failed(result))
    return mlir::failure();

  llvm::SmallVector<mlir::Value> replacements;
  for (mlir::Value res : nest.tile->getResults())
    replacements.push_back(result->replacements.lookup(res));
  rewriter.replaceOp(nest.tile, replacements);

  // tiledAndFusedOps also holds the fused un-transpose.
  auto kernel = llvm::find_if(result->tiledAndFusedOps, [](mlir::Operation *op) {
    return llvm::isa<mlir::linalg::MatmulOp>(op);
  });
  if (kernel == result->tiledAndFusedOps.end())
    return mlir::failure();
  nest.tile = *kernel;
  return mlir::success();
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
// With packB and/or packA, the operand is padded and the pad hoisted out of
// the ir and jr loops, turning it into a packing loop nest in the pc body
// plus a cheap slice at the use site (see packB and packA).
static mlir::LogicalResult blockMatmul(mlir::IRRewriter &rewriter,
                                       mlir::linalg::MatmulOp op,
                                       const mlir_edsl::MatmulStrategy &s) {
  auto nest = tileNest(rewriter, op, s);
  if (mlir::failed(nest))
    return mlir::failure();

  if (s.packA || s.packB) {
    if (mlir::failed(mergeSliceChains(*nest)) ||
        mlir::failed(padTile(rewriter, *nest, s)))
      return mlir::failure();
    if (s.packB && mlir::failed(packB(rewriter, *nest, s)))
      return mlir::failure();
    if (s.packA && mlir::failed(packA(rewriter, *nest)))
      return mlir::failure();
  }

  return tileK(rewriter, *nest, s);
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
                     llvm::cl::init(true)};
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

    // Collected up front: blocking one matmul only rewrites ops inside its
    // own loop nest, so the other handles stay valid.
    llvm::SmallVector<mlir::linalg::MatmulOp> candidates;
    func.walk([&](mlir::linalg::MatmulOp op) { candidates.push_back(op); });

    for (mlir::linalg::MatmulOp op : candidates) {
      auto strategy = mlir_edsl::chooseStrategy(op, ov);
      if (mlir::failed(strategy))
        continue; // Guard rejected it: leave it for the existing pipeline.
      if (mlir::failed(blockMatmul(rewriter, op, *strategy))) {
        // Do NOT signalPassFailure — a partially tiled matmul is still
        // correct, and the remaining passes can lower whatever is left. The
        // op may be gone by now (a failure part-way through packing), so
        // diagnose against the function.
        func->emitWarning(
            "linalg-matmul-blocked: blocking failed, leaving matmul as-is");
      }
    }
  }
};
} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedPass() {
  return std::make_unique<LinalgMatmulBlockedPass>();
}

} // namespace mlir_edsl
