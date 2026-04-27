#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Transform/Transforms/TransformInterpreterUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

// Tiles the outermost elementwise linalg.generic (the bias+relu epilogue
// produced by --linalg-fuse-elementwise-ops) and greedily fuses its linalg
// producers (matmul, fill) upward into the generated scf.forall loops.
//
// Running in tensor land (before bufferization) enables epilogue fusion:
// fill → matmul → bias+relu all execute on the same [tileM×tileN] tile,
// keeping the matmul output in L2 cache instead of writing it to DRAM first.
//
// After this pass the body of the scf.forall contains:
//   linalg.fill (tile) → linalg.matmul (tile, full K) → linalg.generic (tile)
// The scf.forall is subsequently converted to omp.parallel by the existing
// ForallToParallelLoop + ConvertSCFToOpenMP pass sequence.
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
    return "Tile bias+relu linalg.generic and fuse matmul+fill producers into "
           "scf.forall loops (epilogue fusion, pre-bufferization)";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    // Find the merged bias+relu generic: its first ins operand is a matmul
    // result (guaranteed by --linalg-fuse-elementwise-ops running first).
    mlir::linalg::GenericOp consumer;
    func.walk([&](mlir::linalg::GenericOp op) {
      if (!op.getInputs().empty() &&
          op.getInputs()[0].getDefiningOp<mlir::linalg::MatmulOp>())
        consumer = op;
    });
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
        rewriter,
        mlir::cast<mlir::TilingInterface>(consumer.getOperation()),
        opts);
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

// Lowers linalg.matmul tiles with static 8x8 shape directly to vector.contract
// using standard 2D indexing maps {(m,k),(k,n),(m,n)}. This bypasses the
// linalg vectorizer which always produces a 3D double-broadcast form
// {(d0,d1,d2),(d0,d1,d2),(d0,d1)} that the OuterProduct lowering strategy
// cannot decompose into vector.outerproduct → vector.fma.
struct LinalgMatmulToContractPass
    : public mlir::PassWrapper<LinalgMatmulToContractPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulToContractPass)
  llvm::StringRef getArgument() const override {
    return "linalg-matmul-to-contract";
  }
  llvm::StringRef getDescription() const override {
    return "Lower static 8x8 linalg.matmul tiles to vector.contract with "
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

      auto aType = mlir::dyn_cast<mlir::MemRefType>(A.getType());
      auto bType = mlir::dyn_cast<mlir::MemRefType>(B.getType());
      auto cType = mlir::dyn_cast<mlir::MemRefType>(C.getType());
      if (!aType || !bType || !cType)
        continue;

      // Only handle static 8x8 tiles — dynamic boundary tiles fall through
      // to convert-linalg-to-loops for scalar lowering.
      if (!aType.hasStaticShape() || aType.getShape() != llvm::ArrayRef<int64_t>{8, 8})
        continue;
      if (!bType.hasStaticShape() || bType.getShape() != llvm::ArrayRef<int64_t>{8, 8})
        continue;
      if (!cType.hasStaticShape() || cType.getShape() != llvm::ArrayRef<int64_t>{8, 8})
        continue;

      auto f32 = mlir::Float32Type::get(ctx);
      auto vecType = mlir::VectorType::get({8, 8}, f32);
      mlir::Location loc = matmul.getLoc();
      rewriter.setInsertionPoint(matmul);

      auto zero = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto pad = rewriter.create<mlir::arith::ConstantOp>(
          loc, f32, rewriter.getF32FloatAttr(0.0f));

      llvm::SmallVector<bool> inBounds = {true, true};
      mlir::Value vA = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecType, A, mlir::ValueRange{zero, zero}, pad, inBounds);
      mlir::Value vB = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecType, B, mlir::ValueRange{zero, zero}, pad, inBounds);
      mlir::Value vC = rewriter.create<mlir::vector::TransferReadOp>(
          loc, vecType, C, mlir::ValueRange{zero, zero}, pad, inBounds);

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

      rewriter.create<mlir::vector::TransferWriteOp>(
          loc, result, C, mlir::ValueRange{zero, zero}, inBounds);

      rewriter.eraseOp(matmul);
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
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

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

  llvm::StringRef getArgument() const override { return "linalg-tile-generic"; }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.generic ops along the innermost dimension";
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::GenericOp> generics;
    func.walk([&](mlir::linalg::GenericOp op) { generics.push_back(op); });

    for (mlir::linalg::GenericOp op : generics) {
      unsigned rank = op.getNumLoops();
      if (rank == 0)
        continue;

      // Tile only the innermost loop; leave all outer loops untiled.
      llvm::SmallVector<int64_t> sizes(rank, 0);
      sizes.back() = tileSize;

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

// Applies a pre-parsed transform dialect strategy to the module.
// The strategy module is parsed eagerly at pipeline setup time (when the
// context is fully configured) and shared via shared_ptr across clones.
// This avoids context mutation during pass execution and parse-time
// "unknown op" errors caused by extensions not yet being applied.
struct TransformStrategyPass
    : public mlir::PassWrapper<TransformStrategyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransformStrategyPass)

  // Shared across clones — parsed once at factory time.
  std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> strategyModule;
  // If non-empty, skip modules that don't contain a linalg.generic with
  // this library_call string (e.g. skip pure matmul for a relu strategy).
  std::string guardLibraryCall;

  TransformStrategyPass(
      std::shared_ptr<mlir::OwningOpRef<mlir::ModuleOp>> strategyMod,
      llvm::StringRef guard)
      : strategyModule(std::move(strategyMod)),
        guardLibraryCall(guard.str()) {}

  llvm::StringRef getArgument() const override {
    return "apply-transform-strategy";
  }
  llvm::StringRef getDescription() const override {
    return "Apply a pre-parsed transform dialect strategy to the module";
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    if (!guardLibraryCall.empty()) {
      bool found = false;
      module.walk([&](mlir::linalg::GenericOp op) {
        auto lc = op->getAttrOfType<mlir::StringAttr>("library_call");
        if (lc && lc.getValue() == guardLibraryCall) {
          found = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (!found)
        return;
    }

    mlir::Operation *transformRoot =
        mlir::transform::detail::findTransformEntryPoint(
            module, **strategyModule);
    if (!transformRoot) {
      signalPassFailure();
      return;
    }
    mlir::transform::TransformOptions options;
    if (mlir::failed(mlir::transform::applyTransformNamedSequence(
            module, transformRoot, **strategyModule, options)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createTransformStrategyPass(mlir::MLIRContext *ctx,
                                                        llvm::StringRef strategy,
                                                        llvm::StringRef guardLibraryCall) {
  auto strategyMod = std::make_shared<mlir::OwningOpRef<mlir::ModuleOp>>(
      mlir::parseSourceString<mlir::ModuleOp>(strategy, ctx));
  if (!*strategyMod)
    llvm::report_fatal_error("TransformStrategyPass: failed to parse strategy");
  return std::make_unique<TransformStrategyPass>(std::move(strategyMod),
                                                 guardLibraryCall);
}
std::unique_ptr<mlir::Pass> createLinalgOuterTileAndFusePass() {
  return std::make_unique<LinalgOuterTileAndFusePass>(64, 64);
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

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass() {
  using LoopType = mlir::scf::SCFTilingOptions::LoopType;
  return std::make_unique<LinalgMatmulTilingPass>(32, 32, 0, LoopType::ForallOp);
}
#endif

} // namespace mlir_edsl
