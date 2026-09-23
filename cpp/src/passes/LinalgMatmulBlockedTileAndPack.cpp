//===- LinalgMatmulBlockedTileAndPack.cpp - cache/register tiles + packing ===//
//
// linalg-matmul-blocked-tile-and-pack: the second of the three blocked
// matmul passes (see BlockedStage in MatmulStrategy.h). Tiles each
// Distributed tile into pc → jr → ir loops over an MR x NR x KC tile, packs
// its A and B operands into contiguous panels, and leaves it at stage Tiled.
//
// Tiling and packing share a pass because packing hoists to the pc body and
// needs pc, jr and ir by identity; canonicalize between passes would remove
// any of them that has a single iteration.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <array>

namespace {

using mlir_edsl::BlockedStage;
using mlir_edsl::tileOneLevel;

// Rows of B packed per iteration of the B~ packing loop. 8 is one f32 ymm
// register: each iteration copies an 8 x NR block with two vector transfers.
constexpr int64_t kPackTileRows = 8;

// Runs one pattern set greedily over `ops` only, plus the ops the patterns
// create. Scoping is what lets the caller keep holding handles: ops outside
// the set are never folded, rewritten or erased as dead. Kept deliberately
// narrow — no canonicalizer — because canonicalization at the wrong moment
// undoes two of the rewrites below (see packB).
static mlir::LogicalResult
applyPatternSetTo(llvm::ArrayRef<mlir::Operation *> ops,
                  mlir::RewritePatternSet &&patterns) {
  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingAndNewOps);
  return mlir::applyOpPatternsGreedily(ops, std::move(patterns), config);
}

// The loops this pass creates around one tile. The handles stay valid
// through packing because C is never padded: padding it would make
// hoistPaddingOnTensors rebuild the loops to carry the pad through iter_args.
struct BlockedNest {
  mlir::scf::ForOp pc, jr, ir;
  mlir::Operation *tile; // MR x NR x KC
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

// Tiles a distributed MC x NC x K tile into
//
//   pc (K/KC) → jr (NC/NR) → ir (MC/MR)
//
// leaving an MR x NR x KC tile. The rest stay serial: K must not be split
// across threads (it is the reduction), and the register tiles are
// per-thread work.
static mlir::FailureOr<BlockedNest>
tileCacheAndRegisterLevels(mlir::IRRewriter &rewriter, mlir::Operation *tile,
                           const mlir_edsl::MatmulStrategy &s) {
  const std::array<std::array<int64_t, 3>, 3> levels = {{
      {0, 0, s.kc}, // pc: K cache block
      {0, s.nr, 0}, // jr: N register tile
      {s.mr, 0, 0}, // ir: M register tile
  }};

  std::array<mlir::Operation *, 3> levelLoops{};
  for (auto [i, sizes] : llvm::enumerate(levels)) {
    auto tiled = tileOneLevel(rewriter, tile, sizes);
    if (mlir::failed(tiled) || tiled->loops.size() != 1)
      return mlir::failure();
    tile = tiled->op;
    levelLoops[i] = tiled->loops.front().getOperation();
  }

  BlockedNest nest;
  nest.pc = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[0]);
  nest.jr = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[1]);
  nest.ir = llvm::dyn_cast<mlir::scf::ForOp>(levelLoops[2]);
  nest.tile = tile;
  if (!nest.pc || !nest.jr || !nest.ir)
    return mlir::failure();
  return nest;
}

// Collapses the slice chains feeding the tile's A and B into single slices of
// the original tensors. Every tiling level slices the previous level's slice,
// so the tile's operands are defined *inside* the loops we want to hoist out
// of, and hoistPaddingOnTensors refuses with "Source not defined outside of
// loops". The chains are followed back from the operands rather than found
// under the forall, which canonicalize may have removed. Seeding the whole
// chain also lets the driver erase the outer slices once they are dead.
static mlir::LogicalResult mergeSliceChains(BlockedNest &nest) {
  llvm::SmallVector<mlir::Operation *> slices;
  for (mlir::Value operand : nest.tile->getOperands().take_front(2))
    for (auto slice = operand.getDefiningOp<mlir::tensor::ExtractSliceOp>();
         slice;
         slice = slice.getSource().getDefiningOp<mlir::tensor::ExtractSliceOp>())
      slices.push_back(slice);
  mlir::RewritePatternSet patterns(nest.tile->getContext());
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
// scalars from MR different rows. Leaves an un-transpose in front of the tile
// for the kernel pass to fuse into its k-loop.
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

// Tiles one distributed tile down to MR x NR x KC and packs its operands.
static mlir::LogicalResult tileAndPack(mlir::IRRewriter &rewriter,
                                       mlir::Operation *tile,
                                       const mlir_edsl::MatmulStrategy &s) {
  auto nest = tileCacheAndRegisterLevels(rewriter, tile, s);
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

  mlir_edsl::setBlockedStage(nest->tile, BlockedStage::Tiled);
  return mlir::success();
}

struct LinalgMatmulBlockedTileAndPackPass
    : public mlir::PassWrapper<LinalgMatmulBlockedTileAndPackPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulBlockedTileAndPackPass)

  // Packing builds tensor.pad/empty, linalg.transpose and vector transfers.
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                    mlir::linalg::LinalgDialect, mlir::vector::VectorDialect,
                    mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-tile-and-pack";
  }
  llvm::StringRef getDescription() const override {
    return "Tile distributed blocked matmuls into pc/jr/ir loops over an "
           "MR x NR x KC tile and pack their A and B operands";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<
        std::pair<mlir::linalg::MatmulOp, mlir_edsl::MatmulStrategy>>
        tiles;
    if (mlir::failed(mlir_edsl::collectBlockedTilesAtStage(
            func, BlockedStage::Distributed, tiles)))
      return signalPassFailure();

    // A tile left half-done here still carries mlir_edsl.blocked, which keeps
    // the older passes off it, so a failure cannot fall back to them.
    for (auto &[tile, strategy] : tiles) {
      if (mlir::failed(tileAndPack(rewriter, tile, strategy))) {
        func->emitError("linalg-matmul-blocked-tile-and-pack failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedTileAndPackPass() {
  return std::make_unique<LinalgMatmulBlockedTileAndPackPass>();
}

} // namespace mlir_edsl
