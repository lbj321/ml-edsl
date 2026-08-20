#include "mlir_edsl/MLIRLowering.h"
#include "mlir_edsl/MLIRLoweringPasses.h"

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
#include "llvm/Support/FileSystem.h"

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
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/AllocationOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
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


} // anonymous namespace

namespace mlir_edsl {

MLIRLowering::MLIRLowering()
    : context(std::make_unique<mlir::MLIRContext>()),
      ctx_(context.get()) {
  registerRequiredDialects(ctx_);
}

MLIRLowering::MLIRLowering(mlir::MLIRContext *sharedContext,
                           bool captureSnapshots)
    : ctx_(sharedContext), snapshotsEnabled(captureSnapshots) {
  registerRequiredDialects(ctx_);
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
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);
  mlir::linalg::registerTilingInterfaceExternalModels(registry);
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::tensor::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::affine::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  // Required by ownership-based-buffer-deallocation when scf.if (or other
  // control-flow ops) appear inside linalg regions (e.g. tensor_map with If).
  mlir::arith::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  mlir::cf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  // Required by PromoteBuffersToStackPass (GPU epilogue fusion pipeline) to
  // query whether a memref.alloc can be replaced with memref.alloca.
  mlir::memref::registerAllocationOpInterfaceExternalModels(registry);

  context->appendDialectRegistry(registry);

  // Register LLVM translation interfaces
  mlir::registerLLVMDialectTranslation(*context);
  mlir::registerBuiltinDialectTranslation(*context);
  mlir::registerOpenMPDialectTranslation(*context);
}

void MLIRLowering::attachInstrumentation(mlir::PassManager &pm) {
  if (snapshotsEnabled)
    pm.addInstrumentation(
        std::make_unique<IRSnapshotInstrumentation>(&snapshots));
}

bool MLIRLowering::runPipeline(mlir::PassManager &pm, mlir::ModuleOp module) {
  pm.enableVerifier(true);
  // IRSnapshotInstrumentation (attached below whenever snapshotsEnabled) mutates
  // a plain std::vector from PassInstrumentation callbacks, which the pass
  // manager invokes from worker threads whenever it runs nested per-FuncOp
  // passes (pm.addNestedPass<FuncOp>, used throughout this pipeline) on a
  // module with more than one function concurrently. Multithreading must be
  // off whenever that instrumentation is attached, or those callbacks race on
  // the unsynchronized vector — silent heap corruption, surfacing later as an
  // unrelated-looking segfault (e.g. auto-invalidate compiling a second
  // function into the same module after a DYN shape change).
  if (snapshotsEnabled) {
    pm.getContext()->disableMultithreading();
  }
  if (mlir::succeeded(pm.run(module)))
    return true;
  llvm::raw_string_ostream os(failureIR_);
  module.print(os);
  llvm::errs() << "\n[mlir_edsl] Lowering pipeline failed. IR at failure:\n"
               << failureIR_ << "\n";
  return false;
}

void MLIRLowering::addCPUPasses(mlir::PassManager &pm) {
  // Outer 64×64 tile-and-fuse epilogue fusion, run on tensor semantics
  // (pre-bufferize). LinalgOuterTileAndFusePass tiles the relu generic (when
  // present) via the TilingInterface and fuses bias_add/matmul/fill into the
  // resulting scf.forall via tileConsumerAndFuseProducersUsingSCF — fusion
  // legality is straightforward on tensor SSA values but hard to prove once
  // operands are aliasing memrefs, hence doing this before bufferization.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgOuterTileAndFusePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Outer 64×64 parallel tiling for any matmul not already covered by the
  // fusion above (e.g. a bare matmul with no relu epilogue). Matmuls already
  // nested inside the scf.forall the fusion pass produced are skipped — see
  // the guard in LinalgMatmulTilingPass — so this never double-tiles a
  // fused matmul.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgMatmulParallelTilingPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Cache-block K into serial 256-wide chunks for matmuls whose K is large
  // (a no-op below that threshold). Targets both the epilogue-fusion path and
  // the fallback above, since both leave the matmul's K full-length inside
  // the outer forall. See LinalgMatmulKTilingPass for why this matters — CPU
  // cache sizes, not correctness.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgMatmulKTilingPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Inner 8x8 serial tiling, also run on tensor semantics (pre-bufferize) for
  // the same reason as the outer tiling above. Nesting inside the outer
  // forall's boundary tile (e.g. the 32-wide remainder on a 96x96 matmul)
  // requires ValueBoundsOpInterface support for affine ops — see the
  // affine::registerValueBoundsOpInterfaceExternalModels registration in
  // registerRequiredDialects.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgMatmulTilingPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Tile linalg.generic ops (elementwise, bias, relu, etc.) to strips of 8
  // along the innermost dimension before vectorization, also on tensor
  // semantics (pre-bufferize) — same TilingInterface-based pass, no memref
  // dependency. Without this, the vectorizer sees the full tensor as a
  // single vector<NxNxf32>, causing LLVM O3 to hang on large shapes (e.g.
  // 512x512) due to combinatorial explosion in its analysis passes.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgGenericTilingPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Lower static 8x8 linalg.matmul tiles to vector.contract with standard
  // 2D indexing maps (m,k)x(k,n)->(m,n), on tensor semantics (pre-bufferize).
  // Must run before LinalgVectorizationPass (which stays post-bufferize
  // below) — linalg::vectorize always produces a 3D double-broadcast form
  // that the OuterProduct lowering cannot decompose into vector.fma, so
  // matmul must never reach it. Running this pass earlier still guarantees
  // that ordering since it consumes/erases every linalg.matmul it touches.
  // Bufferizing the vector.transfer_read/write this produces requires
  // vector::registerBufferizableOpInterfaceExternalModels (see
  // registerRequiredDialects).
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgMatmulToContractPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Vectorize remaining linalg structured ops → vector dialect, on tensor
  // semantics (pre-bufferize). linalg::vectorize is dialect-agnostic
  // upstream (the standard "vectorize before bufferize" pattern), so no
  // rewrite was needed here, unlike LinalgMatmulToContractPass above.
  // (linalg.matmul is already handled by LinalgMatmulToContractPass above,
  // so this only ever sees generic/fill ops.)
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgVectorizationPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Fuse mulf + multi_reduction → vector.contract for better LLVM codegen
  pm.addNestedPass<mlir::func::FuncOp>(createVectorCleanupPass());

  // Lower vector.contract → vector.outerproduct on rank-1 slices, on tensor
  // semantics (pre-bufferize). Pure vector.*-to-vector.* rewrite — no tensor
  // or memref operands involved (vector.contract/outerproduct/fma only ever
  // touch vector<> values fed by vector.transfer_read/write), so bufferization
  // state is irrelevant to it. Must happen before convert-vector-to-scf: if a
  // rank-3 contract is still present at that pass, it expands the 3D
  // transfer_reads into broadcast+transpose+alloca loops, defeating
  // vectorization entirely.
  pm.addNestedPass<mlir::func::FuncOp>(createVectorContractToOuterProductPass());

  // Lower vector.multi_reduction (produced by linalg.reduce vectorization),
  // on tensor semantics (pre-bufferize). Pure vector.*-to-vector.* rewrite —
  // no tensor or memref operands involved — so, like the two passes above,
  // bufferization state is irrelevant to it.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::vector::createLowerVectorMultiReductionPass());

  // Bufferize tensor ops to memref ops, including function boundaries.
  // identity-layout-map produces plain memref<NxT> (no strided layout) at
  // function boundaries, matching the memref descriptors Python passes in.
  //
  // Fold tensor.empty ops that only serve as destinations into direct writes
  // on the destination buffer. Must run before one-shot-bufferize.
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());

  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // CSE unifies structurally-identical subviews that one-shot-bufferize just
  // produced (e.g. one from linalg.matmul's outs operand, one from the
  // tensor.insert_slice it lowers to memref.copy) so canonicalize can fold
  // the resulting self-copy instead of leaving it as a redundant copy. Doing
  // this now, before the ownership-based dealloc pass, keeps its buffer-alias
  // analysis working over already-deduped IR.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(
      mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());

  // scf.forall → scf.parallel → omp.parallel. Body already contains scf.for
  // from the inner tiling above (it now runs pre-bufferize, unlike before),
  // and that no longer breaks the downstream scf-to-control-flow pass below
  // — confirmed by the full test suite, including the 96x96 boundary-tile
  // case in test_multicore.py.
  pm.addPass(mlir::createForallToParallelLoopPass());
  pm.addPass(mlir::createConvertSCFToOpenMPPass());
  // ConvertSCFToOpenMPPass always wraps the loop body in a memref.alloca_scope
  // for stack-scoping, whether or not anything inside needs it. If nothing
  // does (no memref.alloca), it must be inlined away here, before scf-to-cf:
  // AllocaScopeOp requires a single-block body, and scf-to-cf introduces
  // branches for any scf.for still inside it. See AllocaScopeCleanupPass.
  pm.addNestedPass<mlir::func::FuncOp>(createAllocaScopeCleanupPass());

  // Fallback: lower any remaining (un-vectorized) linalg ops to scf.for loops
  pm.addPass(mlir::createConvertLinalgToLoopsPass());

  // Lower complex vector.transfer_read/write (permutation maps, broadcasts)
  // to scalar SCF loops before LLVM conversion
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());

  // Lower all remaining vector ops → LLVM intrinsics.
  // x86Vector enables AVX/FMA intrinsic emission for vector.fma on x86.
  mlir::ConvertVectorToLLVMPassOptions vecToLLVMOpts;
  vecToLLVMOpts.x86Vector = true;
  pm.addPass(mlir::createConvertVectorToLLVMPass(vecToLLVMOpts));

  // Lower ub.poison (generated by VectorToSCF for out-of-bounds positions)
  pm.addPass(mlir::createUBToLLVMConversionPass());

  // Lower inner scf.for loops → CF (scf.parallel/forall already converted).
  pm.addPass(mlir::createSCFToControlFlowPass());
  // Expand memref.subview with dynamic offsets (produced by tiling) into
  // explicit arith/affine pointer arithmetic — must run before lower-affine
  // and finalize-memref-to-llvm.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  // Lower affine.apply (produced by expand-strided-metadata) to arith ops.
  pm.addPass(mlir::createLowerAffinePass());
  // Lower arith ops → LLVM (after affine is gone).
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertOpenMPToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

LoweredModule MLIRLowering::lowerToLLVMModule(mlir::ModuleOp module) {
  mlir::OwningOpRef<mlir::ModuleOp> clonedModule = module.clone();

  mlir::PassManager pm(ctx_);
  attachInstrumentation(pm);
  addCPUPasses(pm);
  if (!runPipeline(pm, *clonedModule)) {
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
  mlir::registerConvertOpenMPToLLVMInterface(reg);
  ctx->appendDialectRegistry(reg);
}

// Phase 1: fuse + bufferize + linalg→parallel→gpu + kernel outlining.
// After this runs, gpu.launch_func ops are present and can be analyzed.
void MLIRLowering::addGPUPreOutliningPasses(mlir::PassManager &pm) {
  // Two-level epilogue fusion (tensor semantics, pre-bufferize): tile the
  // relu (found by its "relu" library_call attribute) to a 32x32 block-level
  // scf.forall and fuse bias_add/matmul/fill producers into it, then tile
  // the resulting nested relu again to a 1x1 thread-level scf.forall and
  // fuse the block-level ops into that. Reuses the CPU epilogue fusion pass
  // unchanged (it already finds ops purely by library_call, with no
  // CPU/GPU-specific assumptions) — tiling clones op attributes, so the
  // library_call survives onto the nested relu that the second call finds.
  // No-ops cleanly (existing guard) when there's no relu, e.g. a bare
  // matmul or a bias-only epilogue, falling through to the fallback below.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgOuterTileAndFusePass(32, 32));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgOuterTileAndFusePass(1, 1));
  pm.addPass(mlir::createCanonicalizerPass());

  // Fallback: tile any matmul the fusion above didn't reach (bare matmul, or
  // an epilogue without relu) into 32x32 scf.forall blocks for GPU block
  // mapping, on tensor semantics (pre-bufferize) — same TilingInterface-based
  // pass (and same no-memref-dependency reasoning) as the CPU outer 64x64
  // tiling. Matmuls already nested inside the fused forall(s) above are
  // skipped by this pass's existing guard, so this never double-tiles.
  pm.addNestedPass<mlir::func::FuncOp>(createLinalgGPUMatmulTilingPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Fold tensor.empty ops that only serve as destinations into direct writes
  // on the destination buffer. Must run before one-shot-bufferize.
  pm.addPass(mlir::bufferization::createEmptyTensorEliminationPass());

  mlir::bufferization::OneShotBufferizePassOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.functionBoundaryTypeConversion =
      mlir::bufferization::LayoutMapOption::IdentityLayoutMap;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // CSE unifies structurally-identical subviews that one-shot-bufferize just
  // produced (e.g. one from linalg.matmul's outs operand, one from the
  // tensor.insert_slice it lowers to memref.copy) so canonicalize can fold
  // the resulting self-copy instead of leaving it as a redundant copy. Doing
  // this now, before the ownership-based dealloc pass, keeps its buffer-alias
  // analysis working over already-deduped IR.
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Promote small per-tile scratch allocations (the fill/bias_add
  // intermediate results inside the fused thread tile, e.g. a 1x1xf32 alloc
  // per thread) from heap memref.alloc to stack memref.alloca. This is
  // mandatory for GPU, not just an optimization: memref.alloc lowers to
  // llvm.call @malloc, and a malloc call left inside gpu.module has no
  // resolvable symbol on the device — it assembles to PTX "successfully"
  // but the kernel is broken. alloca lowers to ordinary local/register
  // storage instead, which the NVPTX backend handles fine. The default
  // size threshold (1024 bytes) leaves the actual output buffer, which
  // isn't a per-tile scratch alloc, on the heap as before.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createPromoteBuffersToStackPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(
      mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
  pm.addPass(
      mlir::bufferization::createBufferDeallocationSimplificationPass());
  pm.addPass(mlir::bufferization::createLowerDeallocationsPass());

  // Convert scf.forall (tile loops, with GPU mapping) → scf.parallel so
  // gpu-map-parallel-loops can annotate them for blockIdx mapping. Must stay
  // post-bufferize: scf.parallel has no results (side-effecting only), so it
  // can't represent the tensor-SSA form scf.forall has pre-bufferize — same
  // hard gate as the CPU forall-to-parallel/convert-scf-to-openmp passes.
  pm.addPass(mlir::createForallToParallelLoopPass());

  pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
  // Fold the trivial [1×1] scf.parallel loops produced by converting the
  // 1×1 linalg ops (fill, bias, relu) inside the thread forall. Without
  // this, GpuMapParallelLoopsPass encounters depth-2 parallels it cannot
  // assign to any GPU dimension and fails to map the whole structure.
  pm.addPass(mlir::createCanonicalizerPass());
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

  // Lower inner scf.for loops → CF (scf.parallel/forall already converted).
  pm.addPass(mlir::createSCFToControlFlowPass());
  // Expand memref.subview with dynamic offsets (produced by tiling) into
  // explicit arith/affine pointer arithmetic — must run before lower-affine
  // and finalize-memref-to-llvm.
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  // Lower affine.apply (produced by expand-strided-metadata) to arith ops.
  pm.addPass(mlir::createLowerAffinePass());
  // Lower arith ops → LLVM (after affine is gone).
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());

  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
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

    // Convention: void-returning host functions use the last block arg as the
    // caller-allocated output out-param. Scalar-returning functions have no
    // out-param (UINT_MAX = sentinel for "no output arg").
    auto hostFunc = launchOp->getParentOfType<mlir::func::FuncOp>();
    const unsigned numHostArgs = hostFunc ? hostFunc.getNumArguments() : 0;
    const bool hasOutParam =
        hostFunc && hostFunc.getResultTypes().empty() && numHostArgs > 0;
    const unsigned outParamArgIdx = hasOutParam ? (numHostArgs - 1) : UINT_MAX;

    for (mlir::Value arg : launchOp.getKernelOperands()) {
      mlir::Type ty = arg.getType();
      GPUKernelArg ka;

      if (auto memTy = mlir::dyn_cast<mlir::MemRefType>(ty)) {
        auto shape = std::vector<int64_t>(memTy.getShape().begin(),
                                          memTy.getShape().end());
        if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(arg)) {
          if (blockArg.getArgNumber() == outParamArgIdx) {
            ka.kind  = GPUKernelArg::Kind::OutputMemRef;
            ka.shape = shape;
          } else {
            ka.kind     = GPUKernelArg::Kind::InputMemRef;
            ka.paramIdx = blockArg.getArgNumber();
            ka.shape    = shape;
          }
        } else {
          // Non-block-arg memref — not expected with the new IR pattern but
          // kept for robustness.
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
    attachInstrumentation(pm1);
    addGPUPreOutliningPasses(pm1);
    if (!runPipeline(pm1, *cloned))
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
    attachInstrumentation(pm2);
    addGPUNVVMPasses(pm2);
    if (!runPipeline(pm2, *cloned))
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
void MLIRLowering::analyzeKernelLaunches(mlir::ModuleOp, GPULoweredModule &) {}
GPULoweredModule MLIRLowering::lowerToGPUModule(mlir::ModuleOp) {
  throw std::runtime_error(
      "GPU support not compiled in (rebuild with -DMLIR_EDSL_CUDA=ON)");
}

#endif // MLIR_EDSL_CUDA_ENABLED

} // namespace mlir_edsl