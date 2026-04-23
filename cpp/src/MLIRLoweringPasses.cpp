#include "mlir_edsl/MLIRLoweringPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace {

// Converts tensor-returning functions to void + writable-out-param convention
// before bufferization. This lets one-shot-bufferize write the result directly
// into the caller's buffer (no memref.copy), while the tensor return kept the
// op chain live through linalg-fuse-elementwise-ops' greedy DCE.
//
// Transform per function with a tensor return type:
//   1. Append a new {bufferization.writable} tensor arg as the out-param.
//   2. Replace each `return %val` with:
//        bufferization.materialize_in_destination %val in %out_param
//        return
//   3. Update the function type to void return.
struct TensorReturnToOutParamPass
    : public mlir::PassWrapper<TensorReturnToOutParamPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorReturnToOutParamPass)
  llvm::StringRef getArgument() const override {
    return "tensor-return-to-out-param";
  }
  llvm::StringRef getDescription() const override {
    return "Convert tensor-returning functions to void + writable out-param";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::FunctionType funcType = func.getFunctionType();

    // Only handle single tensor return.
    if (funcType.getNumResults() != 1)
      return;
    auto tensorType =
        mlir::dyn_cast<mlir::RankedTensorType>(funcType.getResult(0));
    if (!tensorType)
      return;

    mlir::IRRewriter rewriter(func->getContext());

    // Append the out-param to the function's entry block and type.
    mlir::Block &entryBlock = func.getBody().front();
    mlir::BlockArgument outParam =
        entryBlock.addArgument(tensorType, func.getLoc());
    auto outArgIdx = funcType.getNumInputs(); // index of the new arg

    // Update function type: same inputs + new out-param, void return.
    llvm::SmallVector<mlir::Type> newInputs(funcType.getInputs());
    newInputs.push_back(tensorType);
    func.setType(rewriter.getFunctionType(newInputs, {}));

    // Mark the new arg writable so the bufferizer writes in-place.
    func.setArgAttr(outArgIdx, "bufferization.writable",
                    rewriter.getBoolAttr(true));

    // Replace each return with void return (+ optional materialize_in_destination fallback).
    func.walk([&](mlir::func::ReturnOp ret) {
      rewriter.setInsertionPoint(ret);
      mlir::Value val = ret.getOperands()[0];

      // Trace backwards through linalg outs chain to find the root
      // tensor.empty() and replace it with outParam. This makes the entire
      // chain write directly into the caller's buffer. If the trace succeeds,
      // the result already aliases outParam and bufferization writes in-place
      // with no copy needed — so we skip materialize_in_destination entirely.
      // Only emit it as a fallback when the trace fails (e.g. the chain
      // doesn't terminate in a tensor.empty we can replace).
      bool replacedEmpty = false;
      mlir::Value cursor = val;
      while (auto linalgOp =
                 cursor.getDefiningOp<mlir::linalg::LinalgOp>()) {
        unsigned idx =
            mlir::cast<mlir::OpResult>(cursor).getResultNumber();
        mlir::Value outsVal = linalgOp.getDpsInits()[idx];
        if (auto emptyOp =
                outsVal.getDefiningOp<mlir::tensor::EmptyOp>()) {
          rewriter.replaceAllUsesWith(outsVal, outParam);
          rewriter.eraseOp(emptyOp);
          replacedEmpty = true;
          break;
        }
        cursor = outsVal;
      }

      if (!replacedEmpty) {
        // Fallback for return patterns that don't terminate in a linalg op
        // with a tensor.empty outs (e.g. tensor.insert, scf.if results).
        // materialize_in_destination tells the bufferizer where the result
        // must land; may produce a memref.copy if aliasing can't be proven.
        rewriter.create<mlir::bufferization::MaterializeInDestinationOp>(
            ret.getLoc(), mlir::TypeRange{tensorType}, val, outParam);
      }
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(ret);
    });
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

// Tiles linalg.matmul into 8x8x8 scf.for loops (M, N, K all tiled to 8).
// K tiling is required for square vector<8x8x8> contracts, which the
// OuterProduct lowering strategy decomposes into vector.outerproduct →
// vector.fma (vfmadd231ps on AVX2). Without K tiling the contract is
// vector<8x8xK> which degenerates to scalar dot products per output element.
// Placed after bufferization so tiling operates on memref semantics.
struct LinalgMatmulTilingPass
    : public mlir::PassWrapper<LinalgMatmulTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulTilingPass)
  llvm::StringRef getArgument() const override { return "linalg-tile-matmul"; }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.matmul into scf.for loops over cache-friendly tiles";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp op : matmuls) {
      // Store tile sizes in a named variable — setTileSizes captures an
      // ArrayRef (non-owning), so the data must outlive the tileUsingSCF call.
      llvm::SmallVector<mlir::OpFoldResult> tileSizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), {8, 8, 8});
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(tileSizes);
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

// TODO: LinalgMatmulParallelTilingPass and LinalgGPUMatmulTilingPass are
// structurally identical — both tile linalg.matmul into scf.forall with K=0.
// They differ only in tile size (64x64 CPU, 32x32 GPU) and description.
// Collapse into one parameterised pass: LinalgMatmulForallTilingPass(M, N).

// Tiles linalg.matmul to 64x64 outer scf.forall blocks for CPU multicore.
// createForallToParallelLoopPass converts the forall to scf.parallel, then
// createConvertSCFToOpenMPPass converts that to omp.parallel + omp.wsloop.
// The inner LinalgMatmulTilingPass then tiles each 64x64 tile to 8x8 for
// vectorization. K is untiled (K=0) to avoid reduction races between threads.
struct LinalgMatmulParallelTilingPass
    : public mlir::PassWrapper<LinalgMatmulParallelTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgMatmulParallelTilingPass)
  llvm::StringRef getArgument() const override {
    return "linalg-tile-matmul-parallel";
  }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.matmul to 64x64 scf.forall outer tiles for CPU multicore";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp op : matmuls) {
      llvm::SmallVector<mlir::OpFoldResult> sizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), {64, 64, 0});
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(sizes);
      opts.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForallOp);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op.getOperation()), opts);
      if (mlir::failed(result)) {
        op->emitWarning("linalg-tile-matmul-parallel: tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
    }
  }
};

#ifdef MLIR_EDSL_CUDA_ENABLED
// Tiles linalg.matmul to 32x32 outer scf.forall blocks for GPU mapping.
// 32x32 = 1024 threads/block (CUDA max). K is untiled (K=0) so each thread
// iterates the full reduction independently.
struct LinalgGPUMatmulTilingPass
    : public mlir::PassWrapper<LinalgGPUMatmulTilingPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgGPUMatmulTilingPass)
  llvm::StringRef getArgument() const override {
    return "linalg-tile-matmul-gpu";
  }
  llvm::StringRef getDescription() const override {
    return "Tile linalg.matmul to 32x32 scf.forall blocks for GPU mapping";
  }
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::IRRewriter rewriter(func->getContext());

    llvm::SmallVector<mlir::linalg::MatmulOp> matmuls;
    func.walk([&](mlir::linalg::MatmulOp op) { matmuls.push_back(op); });

    for (mlir::linalg::MatmulOp op : matmuls) {
      llvm::SmallVector<mlir::OpFoldResult> sizes =
          mlir::getAsIndexOpFoldResult(op->getContext(), {32, 32, 0});
      mlir::scf::SCFTilingOptions opts;
      opts.setTileSizes(sizes);
      opts.setLoopType(mlir::scf::SCFTilingOptions::LoopType::ForallOp);
      rewriter.setInsertionPoint(op);
      auto result = mlir::scf::tileUsingSCF(
          rewriter, llvm::cast<mlir::TilingInterface>(op.getOperation()), opts);
      if (mlir::failed(result)) {
        op->emitWarning("linalg-tile-matmul-gpu: tiling failed, skipping op");
        continue;
      }
      if (op->getNumResults() == 0)
        rewriter.eraseOp(op);
      else
        rewriter.replaceOp(op, result->mergeResult.replacements);
    }
  }
};
#endif // MLIR_EDSL_CUDA_ENABLED

} // namespace

namespace mlir_edsl {

std::unique_ptr<mlir::Pass> createTensorReturnToOutParamPass() {
  return std::make_unique<TensorReturnToOutParamPass>();
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
std::unique_ptr<mlir::Pass> createLinalgMatmulTilingPass() {
  return std::make_unique<LinalgMatmulTilingPass>();
}
std::unique_ptr<mlir::Pass> createLinalgMatmulParallelTilingPass() {
  return std::make_unique<LinalgMatmulParallelTilingPass>();
}

#ifdef MLIR_EDSL_CUDA_ENABLED
std::unique_ptr<mlir::Pass> createLinalgGPUMatmulTilingPass() {
  return std::make_unique<LinalgGPUMatmulTilingPass>();
}
#endif

} // namespace mlir_edsl
