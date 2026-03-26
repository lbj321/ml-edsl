#include "mlir_edsl/MLIRLowering.h"

#include <stdexcept>

#include "mlir/IR/OwningOpRef.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

// Tiles linalg.matmul into scf.for loops over cache-friendly tiles.
// Placed after bufferization so tiling operates on memref semantics;
// strided subviews are handled downstream by convert-vector-to-scf.
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
          mlir::getAsIndexOpFoldResult(op->getContext(), {8, 8, 0});
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
}

void MLIRLowering::addConversionPasses() {

  // Bufferize tensor ops to memref ops.
  passManager.addPass(mlir::bufferization::createOneShotBufferizePass());

  // Insert deallocs for buffers created during bufferization
  passManager.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(
      mlir::bufferization::createBufferDeallocationSimplificationPass());
  passManager.addPass(mlir::bufferization::createLowerDeallocationsPass());

  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgMatmulTilingPass>());
  passManager.addPass(mlir::createCanonicalizerPass());

  // Phase 9.1: Vectorize linalg structured ops → vector dialect
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<LinalgVectorizationPass>());
  passManager.addPass(mlir::createCanonicalizerPass());

  // Fuse mulf + multi_reduction → vector.contract for better LLVM codegen
  passManager.addNestedPass<mlir::func::FuncOp>(
      std::make_unique<VectorCleanupPass>());

  // Fallback: lower any remaining (un-vectorized) linalg ops to scf.for loops
  passManager.addPass(mlir::createConvertLinalgToLoopsPass());

  // Lower vector.multi_reduction (produced by linalg.reduce vectorization)
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::vector::createLowerVectorMultiReductionPass());

  // Lower complex vector.transfer_read/write (permutation maps, broadcasts)
  // to scalar SCF loops before LLVM conversion
  passManager.addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertVectorToSCFPass());

  // Lower all remaining vector ops → LLVM intrinsics
  passManager.addPass(mlir::createConvertVectorToLLVMPass());

  // Lower ub.poison (generated by VectorToSCF for out-of-bounds positions)
  passManager.addPass(mlir::createUBToLLVMConversionPass());
  // Lower SCF to ControlFlow dialect
  passManager.addPass(mlir::createSCFToControlFlowPass());

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

  passManager.addPass(mlir::createReconcileUnrealizedCastsPass());
}

bool MLIRLowering::runLoweringPipeline(mlir::ModuleOp module) {
  passManager.clear();
  addConversionPasses();
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

} // namespace mlir_edsl