#include "mlir_edsl/MLIRLowering.h"

#include <stdexcept>

#ifdef MLIR_EDSL_CUDA_ENABLED
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/GPUCommon/GPUToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"
#include <cstdlib>
#include <fstream>
#endif

#include "mlir/IR/OwningOpRef.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/raw_ostream.h"

namespace {

class IRSnapshotInstrumentation : public mlir::PassInstrumentation {
public:
  using SnapshotList = std::vector<std::pair<std::string, std::string>>;

  explicit IRSnapshotInstrumentation(SnapshotList *snapshots)
      : snapshots(snapshots) {}

  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override {
    // OpToOpPassAdaptor is an internal MLIR wrapper for nested passes.
    // Its snapshot is always identical to the last inner-pass snapshot, so
    // skip it to avoid duplicate "unchanged" noise in the pipeline view.
    if (pass->getName().contains("OpToOpPassAdaptor"))
      return;

    std::string ir;
    llvm::raw_string_ostream os(ir);
    // Walk up to module root for consistent full-module snapshots
    mlir::Operation *root = op;
    while (root->getParentOp())
      root = root->getParentOp();
    root->print(os);
    // Prefer human-readable pass argument name, fall back to class name
    std::string passName = pass->getArgument().str();
    if (passName.empty())
      passName = pass->getName().str();
    snapshots->emplace_back(std::move(passName), std::move(ir));
  }

  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override {
    std::string ir;
    llvm::raw_string_ostream os(ir);
    mlir::Operation *root = op;
    while (root->getParentOp())
      root = root->getParentOp();
    root->print(os);
    std::string passName = pass->getArgument().str();
    if (passName.empty())
      passName = pass->getName().str();
    snapshots->emplace_back("[FAILED] " + passName, std::move(ir));
  }

private:
  SnapshotList *snapshots;
};

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
      if (op->getNumResults() == 0) {
        rewriter.eraseOp(op);
      } else {
        rewriter.replaceOp(op, result->mergeResult.replacements);
      }
    }
  }
};

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
      // M=32, N=32 → 1024 threads/block (CUDA max). K=0 = untiled;
      // each thread iterates the full K reduction independently.
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

} // anonymous namespace

namespace mlir_edsl {

MLIRLowering::MLIRLowering()
    : context(std::make_unique<mlir::MLIRContext>()),
      passManager(context.get()) {
  registerRequiredDialects();
  setupLoweringPipeline();
}

MLIRLowering::MLIRLowering(mlir::MLIRContext *sharedContext,
                           bool captureSnapshots)
    : context(nullptr), passManager(sharedContext),
      snapshotsEnabled(captureSnapshots) {
  registerRequiredDialects(sharedContext);
  setupLoweringPipeline();
}

void MLIRLowering::registerRequiredDialects() {
  registerRequiredDialects(context.get());
}

void MLIRLowering::registerRequiredDialects(mlir::MLIRContext *context) {
  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  context->getOrLoadDialect<mlir::cf::ControlFlowDialect>();
  context->getOrLoadDialect<mlir::tensor::TensorDialect>();
  context->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context->getOrLoadDialect<mlir::vector::VectorDialect>();
  context->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
  context->getOrLoadDialect<mlir::omp::OpenMPDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  // Register bufferizable op interfaces (tells one-shot-bufferize how to
  // convert each op)
  mlir::DialectRegistry registry;
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  context->appendDialectRegistry(registry);

  // Register LLVM translation interfaces
  mlir::registerLLVMDialectTranslation(*context);
  mlir::registerBuiltinDialectTranslation(*context);
}

void MLIRLowering::setupLoweringPipeline() {
  passManager.enableVerifier(true);
  if (snapshotsEnabled) {
    passManager.addInstrumentation(
        std::make_unique<IRSnapshotInstrumentation>(&snapshots));
  }
  addConversionPasses();
}

void MLIRLowering::addConversionPasses() {

  // Fuse elementwise linalg ops (e.g. matmul + bias generic) before bufferization.
  // Tensor-returning functions keep the op chain live through DCE in this pass.
  passManager.addPass(mlir::createLinalgElementwiseOpFusionPass());

  // Convert tensor-returning functions to void + writable-out-param before
  // bufferization, so one-shot-bufferize can write results in-place into the
  // caller's buffer without a memref.copy.
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<TensorReturnToOutParamPass>());

  // Bufferize tensor ops to memref ops, including function boundaries.
  // identity-layout-map produces plain memref<NxT> (no strided layout) at
  // function boundaries, matching the memref descriptors Python passes in.
  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  passManager.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Insert deallocs for buffers created during bufferization
  passManager.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(
      mlir::bufferization::createBufferDeallocationSimplificationPass());
  passManager.addPass(mlir::bufferization::createLowerDeallocationsPass());

  // Outer 64x64 parallel tiles → scf.forall → scf.parallel (one body per CPU thread)
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgMatmulParallelTilingPass>());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createForallToParallelLoopPass());

  // Inner 8x8 serial tiles for vectorization (runs inside each parallel tile body)
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgMatmulTilingPass>());
  passManager.addPass(mlir::createCanonicalizerPass());

  // Lower static 8x8 linalg.matmul tiles to vector.contract with standard
  // 2D indexing maps (m,k)x(k,n)->(m,n). Must run before LinalgVectorizationPass
  // which skips matmul — linalg::vectorize always produces a 3D double-broadcast
  // form that the OuterProduct lowering cannot decompose into vector.fma.
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgMatmulToContractPass>());

  // Phase 9.1: Vectorize remaining linalg structured ops → vector dialect
  // (linalg.matmul is already handled by LinalgMatmulToContractPass above)
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgVectorizationPass>());
  passManager.addPass(mlir::createCanonicalizerPass());

  // Fuse mulf + multi_reduction → vector.contract for better LLVM codegen
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<VectorCleanupPass>());

  // Lower vector.contract → vector.outerproduct on rank-1 slices.
  // Must happen before convert-vector-to-scf: if a rank-3 contract is still
  // present at that pass, it expands the 3D transfer_reads into
  // broadcast+transpose+alloca loops, defeating vectorization entirely.
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<VectorContractToOuterProductPass>());

  // Fallback: lower any remaining (un-vectorized) linalg ops to scf.for loops
  passManager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Lower vector.multi_reduction (produced by linalg.reduce vectorization)
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::vector::createLowerVectorMultiReductionPass());

  // Lower complex vector.transfer_read/write (permutation maps, broadcasts)
  // to scalar SCF loops before LLVM conversion
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertVectorToSCFPass());

  // Lower all remaining vector ops → LLVM intrinsics.
  // x86Vector enables AVX/FMA intrinsic emission for vector.fma on x86.
  mlir::ConvertVectorToLLVMPassOptions vecToLLVMOpts;
  vecToLLVMOpts.x86Vector = true;
  passManager.addPass(mlir::createConvertVectorToLLVMPass(vecToLLVMOpts));

  // Lower ub.poison (generated by VectorToSCF for out-of-bounds positions)
  passManager.addPass(mlir::createUBToLLVMConversionPass());
  // Lower inner scf.for → CF while still inside scf.parallel; convert-scf-to-cf
  // cannot lower scf.for once it is nested inside omp.loop_nest (OMP structural
  // constraints prevent the multi-block CFG expansion in that context).
  passManager.addPass(mlir::createSCFToControlFlowPass());
  // Convert outer scf.parallel → omp.parallel+wsloop; body already has CF ops.
  passManager.addPass(mlir::createConvertSCFToOpenMPPass());

  // Expand memref.subview with dynamic offsets (produced by tiling) into
  // explicit arith/affine pointer arithmetic — must run before lower-affine
  // and finalize-memref-to-llvm.
  passManager.addPass(mlir::memref::createExpandStridedMetadataPass());

  // Lower affine.apply (produced by expand-strided-metadata) to arith ops.
  passManager.addPass(mlir::createLowerAffinePass());

  // Lower arith ops → LLVM (after affine is gone).
  passManager.addPass(mlir::createArithToLLVMConversionPass());

  passManager.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  passManager.addPass(mlir::createConvertControlFlowToLLVMPass());
  passManager.addPass(mlir::createConvertFuncToLLVMPass());
  passManager.addPass(mlir::createConvertOpenMPToLLVMPass());

  passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool MLIRLowering::runLoweringPipeline(mlir::ModuleOp module) {
  if (std::getenv("TRACE_PASSES")) {
    passManager.getContext()->disableMultithreading();
    passManager.enableIRPrinting(
        /*shouldPrintBeforePass=*/[](mlir::Pass *,
                                     mlir::Operation *) { return true; },
        /*shouldPrintAfterPass=*/nullptr,
        /*printModuleScope=*/true,
        /*printAfterOnlyOnChange=*/false,
        /*printAfterOnlyOnFailure=*/false, llvm::errs());
  }
  if (mlir::succeeded(passManager.run(module)))
    return true;
  // Capture partially-lowered module — always, regardless of snapshotsEnabled
  llvm::raw_string_ostream os(failureIR_);
  module.print(os);
  llvm::errs() << "\n[mlir_edsl] Lowering pipeline failed. IR at failure:\n"
               << failureIR_ << "\n";
  return false;
}

LoweredModule MLIRLowering::lowerToLLVMModule(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::ModuleOp> clonedModule = module.clone();

  if (!runLoweringPipeline(*clonedModule)) {
    throw std::runtime_error("Lowering pipeline failed");
  }

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(*clonedModule, *llvmContext);

  if (!llvmModule) {
    throw std::runtime_error("Translation to LLVM IR failed");
  }

  return {std::move(llvmModule), std::move(llvmContext)};
}

std::string MLIRLowering::lowerToLLVMIR(mlir::ModuleOp module) {
  auto lowered = lowerToLLVMModule(module);
  std::string result;
  llvm::raw_string_ostream stream(result);
  lowered.module->print(stream, nullptr);
  return result;
}

#ifdef MLIR_EDSL_CUDA_ENABLED

void MLIRLowering::registerGPUDialects(mlir::MLIRContext *ctx) {
  ctx->getOrLoadDialect<mlir::gpu::GPUDialect>();
  ctx->getOrLoadDialect<mlir::NVVM::NVVMDialect>();
  mlir::registerGPUDialectTranslation(*ctx);
  mlir::registerNVVMDialectTranslation(*ctx);

  // finalize-memref-to-llvm queries ConvertToLLVMPatternInterface on all loaded
  // dialects. Vector dialect is loaded (via linalg setup) but its LLVM
  // conversion extension isn't registered by default — register it here.
  mlir::DialectRegistry reg;
  mlir::arith::registerConvertArithToLLVMInterface(reg);
  mlir::registerConvertComplexToLLVMInterface(reg);
  mlir::cf::registerConvertControlFlowToLLVMInterface(reg);
  mlir::registerConvertFuncToLLVMInterface(reg);
  mlir::gpu::registerConvertGpuToLLVMInterface(reg);
  mlir::index::registerConvertIndexToLLVMInterface(reg);
  mlir::registerConvertMathToLLVMInterface(reg);
  mlir::registerConvertMemRefToLLVMInterface(reg);
  mlir::registerConvertNVVMToLLVMInterface(reg);
  mlir::NVVM::registerConvertGpuToNVVMInterface(reg);
  mlir::ub::registerConvertUBToLLVMInterface(reg);
  mlir::vector::registerConvertVectorToLLVMInterface(reg);
  ctx->appendDialectRegistry(reg);
}

// Phase 1: fuse + bufferize + linalg→parallel→gpu + kernel outlining.
// After this runs, gpu.launch_func ops are present and can be analyzed.
void MLIRLowering::addGPUPreOutliningPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());

  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());

  // Tile matmul into 32x32 scf.forall blocks before parallel loop conversion.
  // Outer tile loops (M/32, N/32) → blockIdx; inner 32x32 → threadIdx (1024 max).
  pm.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgGPUMatmulTilingPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  // Convert scf.forall (tile loops) → scf.parallel so gpu-map-parallel-loops
  // can map them to blockIdx alongside the inner thread-level parallel loops.
  pm.addPass(mlir::createForallToParallelLoopPass());

  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  // GpuMapParallelLoopsPass is OperationPass<func::FuncOp> — must be nested
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createGpuMapParallelLoopsPass());
  pm.addPass(mlir::createConvertParallelLoopToGpuPass());
  pm.addPass(mlir::createGpuKernelOutliningPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

// Phase 2: NVVM lowering on the gpu.module, then host-level LLVM passes.
// convert-gpu-to-nvvm must be nested inside gpu.module; finalize-memref-to-llvm
// must run on the outer builtin.module (it recurses into gpu.module contents).
void MLIRLowering::addGPUNVVMPasses(mlir::PassManager &pm) {
  auto &gpuPm = pm.nest<mlir::gpu::GPUModuleOp>();
  gpuPm.addPass(mlir::createConvertGpuOpsToNVVMOps());

  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool MLIRLowering::runGPULoweringPipeline(mlir::ModuleOp module,
                                          mlir::PassManager &pm) {
  pm.enableVerifier(true);
  if (mlir::succeeded(pm.run(module)))
    return true;
  llvm::raw_string_ostream os(failureIR_);
  module.print(os);
  llvm::errs() << "\n[mlir_edsl] GPU lowering pipeline failed. IR at failure:\n"
               << failureIR_ << "\n";
  return false;
}

// Walk gpu.launch_func ops and classify each kernel argument so that
// executeGPUFunction can pack cuLaunchKernel params without guessing the layout.
void MLIRLowering::analyzeKernelLaunches(mlir::ModuleOp module,
                                          GPULoweredModule &result) {
  auto extractConstIndex = [](mlir::Value v) -> uint32_t {
    mlir::Operation *defOp = v.getDefiningOp();
    if (!defOp) return 1;
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
      return (uint32_t)c.value();
    if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
      return (uint32_t)c.value();
    return 1;
  };

  module.walk([&](mlir::gpu::LaunchFuncOp launchOp) {
    GPUKernelLaunch kernel;
    kernel.moduleName = launchOp.getKernelModuleName().getValue().str();
    kernel.funcName   = launchOp.getKernelName().getValue().str();

    auto grid  = launchOp.getGridSizeOperandValues();
    auto block = launchOp.getBlockSizeOperandValues();
    kernel.gridX  = extractConstIndex(grid.x);
    kernel.gridY  = extractConstIndex(grid.y);
    kernel.gridZ  = extractConstIndex(grid.z);
    kernel.blockX = extractConstIndex(block.x);
    kernel.blockY = extractConstIndex(block.y);
    kernel.blockZ = extractConstIndex(block.z);

    for (mlir::Value arg : launchOp.getKernelOperands()) {
      mlir::Type ty = arg.getType();
      GPUKernelArg ka;

      if (auto memTy = mlir::dyn_cast<mlir::MemRefType>(ty)) {
        auto shape = std::vector<int64_t>(memTy.getShape().begin(),
                                          memTy.getShape().end());
        if (mlir::isa<mlir::BlockArgument>(arg)) {
          ka.kind     = GPUKernelArg::Kind::InputMemRef;
          ka.paramIdx = mlir::cast<mlir::BlockArgument>(arg).getArgNumber();
          ka.shape    = shape;
        } else {
          // Defined by memref.alloc — this is the function's output buffer.
          ka.kind  = GPUKernelArg::Kind::OutputMemRef;
          ka.shape = shape;
        }
      } else if (ty.isIndex() || ty.isInteger(64) || ty.isInteger(32)) {
        int64_t val = 0;
        mlir::Operation *defOp = arg.getDefiningOp();
        if (defOp) {
          if (auto c = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp))
            val = c.value();
          else if (auto c = mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp))
            val = c.value();
        }
        ka.kind   = GPUKernelArg::Kind::I64;
        ka.i64Val = val;
      } else if (ty.isF32() || ty.isF64()) {
        float val = 0.0f;
        mlir::Operation *defOp = arg.getDefiningOp();
        if (defOp) {
          if (auto c = mlir::dyn_cast<mlir::arith::ConstantFloatOp>(defOp))
            val = (float)c.value().convertToDouble();
        }
        ka.kind   = GPUKernelArg::Kind::F32;
        ka.f32Val = val;
      } else {
        llvm::report_fatal_error("Unhandled kernel operand type in GPU lowering");
      }
      kernel.args.push_back(ka);
    }
    result.kernels.push_back(std::move(kernel));
  });
}

// Translate one gpu.module to PTX and return the PTX string.
static std::string gpuModuleToPTX(mlir::gpu::GPUModuleOp gpuModule) {
  // translateModuleToLLVMIR requires llvm.func ops directly in a builtin.module.
  // Wrapping the gpu.module produces an empty result because the GPU dialect
  // translation interface handles gpu.module as an offloading container.
  // Clone the gpu.module body ops (llvm.func etc.) directly into a plain module.
  mlir::OwningOpRef<mlir::ModuleOp> wrapper =
      mlir::ModuleOp::create(gpuModule.getLoc());
  mlir::OpBuilder b(wrapper->getContext());
  b.setInsertionPointToStart(wrapper->getBody());
  for (mlir::Operation &op : gpuModule.getBody()->getOperations())
    b.clone(op);

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(*wrapper, llvmCtx);
  if (!llvmModule)
    throw std::runtime_error("Translation of gpu.module to LLVM IR failed");

  if (std::getenv("SAVE_IR")) {
    std::string irStr;
    llvm::raw_string_ostream irOs(irStr);
    llvmModule->print(irOs, nullptr);
    std::ofstream f("ir_output/gpu_llvm.ll");
    f << irStr;
  }

  llvm::Triple triple("nvptx64-nvidia-cuda");
  llvmModule->setTargetTriple(triple);

  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();

  std::string err;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.getTriple(), err);
  if (!target)
    throw std::runtime_error("NVPTX target not found: " + err);

  llvm::TargetOptions opts;
  auto tm = std::unique_ptr<llvm::TargetMachine>(
      target->createTargetMachine(triple, "sm_75", "+ptx64",
                                  opts, llvm::Reloc::PIC_));
  if (!tm)
    throw std::runtime_error("Failed to create NVPTX TargetMachine");

  llvmModule->setDataLayout(tm->createDataLayout());

  llvm::SmallVector<char, 0> ptxBuf;
  llvm::raw_svector_ostream ptxStream(ptxBuf);
  llvm::legacy::PassManager codegenPm;
  if (tm->addPassesToEmitFile(codegenPm, ptxStream, nullptr,
                              llvm::CodeGenFileType::AssemblyFile))
    throw std::runtime_error("NVPTX target cannot emit PTX assembly");

  codegenPm.run(*llvmModule);
  return std::string(ptxBuf.begin(), ptxBuf.end());
}

GPULoweredModule MLIRLowering::lowerToGPUModule(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::ModuleOp> cloned = module.clone();
  mlir::MLIRContext *ctx = cloned->getContext();
  registerGPUDialects(ctx);

  // Phase 1: outline kernels
  {
    mlir::PassManager pm1(ctx);
    if (snapshotsEnabled)
      pm1.addInstrumentation(
          std::make_unique<IRSnapshotInstrumentation>(&snapshots));
    addGPUPreOutliningPasses(pm1);
    if (!runGPULoweringPipeline(*cloned, pm1))
      throw std::runtime_error("GPU pre-outlining pipeline failed");
  }

  // Analyze gpu.launch_func ops to capture arg layout before NVVM lowering
  // destroys the high-level type info.
  GPULoweredModule result;
  analyzeKernelLaunches(*cloned, result);
  if (result.kernels.empty())
    throw std::runtime_error("GPU outlining produced no kernels");

  // Phase 2: lower gpu.module contents to NVVM/LLVM dialect
  {
    mlir::PassManager pm2(ctx);
    if (snapshotsEnabled)
      pm2.addInstrumentation(
          std::make_unique<IRSnapshotInstrumentation>(&snapshots));
    addGPUNVVMPasses(pm2);
    if (!runGPULoweringPipeline(*cloned, pm2))
      throw std::runtime_error("GPU NVVM lowering pipeline failed");
  }

  // Translate each gpu.module to PTX and attach to the matching kernel info.
  cloned->walk([&](mlir::gpu::GPUModuleOp gpuModule) {
    std::string modName = gpuModule.getName().str();
    std::string ptx = gpuModuleToPTX(gpuModule);

    for (auto &k : result.kernels) {
      if (k.moduleName == modName) {
        k.ptxImage = ptx;
        break;
      }
    }
  });

  // Save PTX files when SAVE_IR=1 for post-crash diagnosis.
  if (std::getenv("SAVE_IR")) {
    llvm::sys::fs::create_directories("ir_output");
    for (size_t i = 0; i < result.kernels.size(); ++i) {
      std::string path = "ir_output/gpu_kernel_" + std::to_string(i) + ".ptx";
      std::ofstream f(path);
      f << result.kernels[i].ptxImage;
    }
  }

  return result;
}

#else // MLIR_EDSL_CUDA_ENABLED

void MLIRLowering::registerGPUDialects(mlir::MLIRContext *) {}
void MLIRLowering::addGPUPreOutliningPasses(mlir::PassManager &) {}
void MLIRLowering::addGPUNVVMPasses(mlir::PassManager &) {}
bool MLIRLowering::runGPULoweringPipeline(mlir::ModuleOp, mlir::PassManager &) {
  return false;
}
void MLIRLowering::analyzeKernelLaunches(mlir::ModuleOp, GPULoweredModule &) {}
GPULoweredModule MLIRLowering::lowerToGPUModule(mlir::ModuleOp) {
  throw std::runtime_error(
      "GPU support not compiled in (rebuild with -DMLIR_EDSL_CUDA=ON)");
}

#endif // MLIR_EDSL_CUDA_ENABLED

} // namespace mlir_edsl