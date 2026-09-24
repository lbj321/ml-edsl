//===- LinalgMatmulBlockedDistribute.cpp - ic x jc forall -----------------===//
//
// linalg-matmul-blocked-distribute: the first of the four blocked matmul
// passes (see BlockedStage in MatmulStrategy.h). Chooses the strategy for
// every matmul chooseStrategy accepts, distributes it over an ic x jc
// scf.forall, and leaves the tile at stage Distributed.
//
//===----------------------------------------------------------------------===//

#include "TilingUtils.h"

#include "mlir_edsl/MLIRLoweringPasses.h"
#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace {

using mlir_edsl::BlockedStage;

// Row height used when strip-mining the C-initializing linalg.fill. Any small
// constant works — the point is only to keep the vectorizer from seeing the
// whole MxN fill as one vector; the column extent is the register tile's NR
// so the stores line up with the microkernel.
constexpr int64_t kFillTileRows = 8;

// Tiles `op` into one 2-D scf.forall over MC x NC tiles of C, marks the tile
// with the strategy, and strip-mines the fill initializing C.
//
// jc and ic are fused because both are parallel dimensions in BLIS, and
// splitting only N makes every thread sweep the whole of A, so total A
// traffic grows with the thread count — measured at 1024^3, 8 bands of 128
// was *slower* than 4 bands of 256. Tiling both keeps each thread's working
// set to an MC-row band of A and an NC-column band of B.
//
// Fails only before `op` is rewritten, leaving it for the older passes; a
// failure once it is inside the forall is reported through `committed`.
static mlir::LogicalResult distribute(mlir::IRRewriter &rewriter,
                                      mlir::linalg::MatmulOp op,
                                      const mlir_edsl::MatmulStrategy &s,
                                      bool &committed) {
  committed = false;
  // Captured before tiling rewrites the operand. Handling it is not
  // optional: no other pass tiles a blocked matmul's fill, so an untiled
  // fill would reach LinalgVectorizationPass as one MxN vector, which
  // convert-vector-to-scf makes a 4 MB stack temporary at 1024^2. Fusing it
  // into the forall was measured and rejected (-9% at 1024^3, 1 thread).
  auto fill = op.getOutputs()[0].getDefiningOp<mlir::linalg::FillOp>();

  auto tiled = mlir_edsl::tileOneLevel(
      rewriter, op, {s.mc, s.nc, 0},
      mlir::scf::SCFTilingOptions::LoopType::ForallOp);
  if (mlir::failed(tiled))
    return mlir::failure();
  committed = true;

  // Set before any further tiling: every later tiling clones the op and
  // carries the attribute along.
  tiled->op->setAttr(mlir_edsl::kBlockedAttrName,
                     mlir_edsl::buildBlockedConfig(rewriter.getContext(), s,
                                                   BlockedStage::Distributed));

  if (fill && mlir::failed(mlir_edsl::tileOneLevel(
                  rewriter, fill.getOperation(), {kFillTileRows, s.nr})))
    return mlir::failure();
  return mlir::success();
}

// A matmul the guard rejects is left untouched for the existing 64x64 /
// 8x8x8 passes.
//
// Runs first in buildCPUPipeline with its default options. A non-default
// strategy can be composed ahead of the pipeline from mlir-edsl-opt as
//
//   mlir-edsl-opt in.mlir '-linalg-matmul-blocked-distribute=mr=6 nr=16' \
//       -cpu-pipeline
//
// because the mlir_edsl.blocked attribute keeps the superseded passes — and
// the pipeline's own run of this pass — off its tiles.
struct LinalgMatmulBlockedDistributePass
    : public mlir::PassWrapper<LinalgMatmulBlockedDistributePass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      LinalgMatmulBlockedDistributePass)

  LinalgMatmulBlockedDistributePass() = default;
  // Pass::Option is not copy-constructible, so the compiler cannot generate
  // the copy constructor clonePass() needs; copying the PassWrapper base
  // preserves the option values.
  LinalgMatmulBlockedDistributePass(
      const LinalgMatmulBlockedDistributePass &other)
      : mlir::PassWrapper<LinalgMatmulBlockedDistributePass,
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

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  }

  llvm::StringRef getArgument() const override {
    return "linalg-matmul-blocked-distribute";
  }
  llvm::StringRef getDescription() const override {
    return "Choose a BLIS blocking strategy for each accepted linalg.matmul "
           "and distribute it over an ic x jc scf.forall";
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

    // Collected up front: distributing one matmul only rewrites that matmul
    // and its fill, so the other handles stay valid.
    llvm::SmallVector<mlir::linalg::MatmulOp> candidates;
    func.walk([&](mlir::linalg::MatmulOp op) { candidates.push_back(op); });

    for (mlir::linalg::MatmulOp op : candidates) {
      auto strategy = mlir_edsl::chooseStrategy(op, ov);
      if (mlir::failed(strategy))
        continue; // Guard rejected it: leave it for the existing pipeline.
      bool committed = false;
      if (mlir::succeeded(distribute(rewriter, op, *strategy, committed)))
        continue;
      if (!committed) {
        op->emitWarning(
            "linalg-matmul-blocked-distribute: tiling failed, leaving matmul "
            "for the existing pipeline");
        continue;
      }
      func->emitError("linalg-matmul-blocked-distribute: fill tiling failed "
                      "after the matmul was distributed");
      return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createLinalgMatmulBlockedDistributePass() {
  return std::make_unique<LinalgMatmulBlockedDistributePass>();
}

} // namespace mlir_edsl
