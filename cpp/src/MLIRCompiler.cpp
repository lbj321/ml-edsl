#include "mlir_edsl/MLIRCompiler.h"
#include "ast.pb.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRExecutor.h"
#include "mlir_edsl/MLIRLowering.h"

#ifdef MLIR_EDSL_CUDA_ENABLED
#include <cuda.h>
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <stdexcept>

namespace mlir_edsl {

MLIRCompiler::MLIRCompiler()
    : state(State::Building), optimizationLevel(OptLevel::O2) {
  // Initialize MLIR context and load dialects
  mlirContext = std::make_unique<mlir::MLIRContext>();
  mlirContext->getOrLoadDialect<mlir::arith::ArithDialect>();
  mlirContext->getOrLoadDialect<mlir::func::FuncDialect>();
  mlirContext->getOrLoadDialect<mlir::scf::SCFDialect>();
  mlirContext->getOrLoadDialect<mlir::memref::MemRefDialect>();
  mlirContext->getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlirContext->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  mlirContext->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();

  // Initialize OpBuilder and module
  opBuilder = std::make_unique<mlir::OpBuilder>(mlirContext.get());
  module = mlir::ModuleOp::create(opBuilder->getUnknownLoc());
  opBuilder->setInsertionPointToEnd(module->getBody());

  // Initialize MLIRBuilder with non-owning pointers
  builder = std::make_unique<MLIRBuilder>(mlirContext.get(), opBuilder.get());
  builder->setParameterMap(&parameterMap);
  builder->setFunctionTable(&functionTable);

  // Initialize executor
  executor = std::make_unique<MLIRExecutor>();
  executor->setOptimizationLevel(MLIRExecutor::OptLevel::O2);

#ifdef MLIR_EDSL_CUDA_ENABLED
  gpuExecutor_ = std::make_unique<MLIRGPUExecutor>();
#endif
}

MLIRCompiler::~MLIRCompiler() = default;

// ==================== FUNCTION STATE ====================

void MLIRCompiler::resetFunctionState() {
  currentFunction = nullptr;
  currentOutParam = {};
  parameterMap.clear();
  builder->clearValueCache();
  opBuilder->setInsertionPointToEnd(module->getBody());
}

// ==================== FUNCTION BUILDING ====================

void MLIRCompiler::createFunction(
    const std::string &name,
    const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> &params,
    mlir::Type returnType) {
  resetFunctionState();

  // Build parameter types directly — tensor params stay as tensor types.
  // bufferize-function-boundaries will convert them to memref at the ABI
  // boundary during lowering.
  std::vector<mlir::Type> paramTypes;
  for (const auto &[paramName, typeSpec] : params) {
    paramTypes.push_back(builder->convertType(typeSpec));
  }

  // Aggregate return: append a hidden out-param and use void return.
  // Python allocates the output buffer and passes it as a memref descriptor.
  // Tensor returns stay as explicit tensor return values — buffer-results-to-out-params
  // converts them to out-params after bufferization, appending at the same position.
  const bool memrefReturn = mlir::isa<mlir::MemRefType>(returnType);
  const bool tensorReturn = mlir::isa<mlir::RankedTensorType>(returnType);

  if (memrefReturn) {
    paramTypes.push_back(returnType);
  }

  auto funcType = opBuilder->getFunctionType(
      paramTypes,
      memrefReturn ? mlir::TypeRange{} : mlir::TypeRange{returnType});
  currentFunction = opBuilder->create<mlir::func::FuncOp>(
      opBuilder->getUnknownLoc(), name, funcType);

  functionTable[name] = currentFunction;

  auto *entryBlock = currentFunction.addEntryBlock();
  opBuilder->setInsertionPointToStart(entryBlock);

  // Map parameter names to block arguments directly — params are already the
  // right types (tensor or memref), no casts needed.
  for (size_t i = 0; i < params.size(); i++) {
    parameterMap[params[i].first] = entryBlock->getArgument(i);
  }

  // Store the out-param block arg for use in finalizeFunction (memref path only).
  if (memrefReturn) {
    currentOutParam = entryBlock->getArgument(params.size());
  }
  (void)tensorReturn;
}

void MLIRCompiler::finalizeFunction(const std::string &name,
                                    mlir::Value result) {
  if (!currentFunction) {
    throw std::runtime_error("No current function to finish");
  }

  // Aggregate return: write result into the caller-allocated out-param.
  if (currentOutParam) {
    auto loc = opBuilder->getUnknownLoc();
    // currentOutParam is only set for memref returns.
    if (mlir::isa<mlir::MemRefType>(result.getType())) {
      // MemRef result: copy only when result is not already the out-param.
      if (result != currentOutParam)
        opBuilder->create<mlir::memref::CopyOp>(loc, result, currentOutParam);
    } else {
      throw std::runtime_error(
          "finalizeFunction: out-param set for non-aggregate type");
    }
    opBuilder->create<mlir::func::ReturnOp>(loc);
  } else if (mlir::isa<mlir::RankedTensorType>(result.getType())) {
    // Tensor result: return directly. buffer-results-to-out-params converts
    // this to an out-param after bufferization, preserving the Python ABI.
    opBuilder->create<mlir::func::ReturnOp>(opBuilder->getUnknownLoc(), result);
  } else if (mlir::isa<mlir::IntegerType, mlir::FloatType>(result.getType())) {
    // Scalar return — normal path.
    opBuilder->create<mlir::func::ReturnOp>(opBuilder->getUnknownLoc(), result);
  } else {
    std::string typeName;
    llvm::raw_string_ostream os(typeName);
    result.getType().print(os);
    throw std::runtime_error("finalizeFunction: unhandled result type '" +
                             typeName + "'");
  }

  compiledFunctions.insert(name);

  // Save MLIR to file if SAVE_IR environment variable is set
  const bool saveIR = std::getenv("SAVE_IR") != nullptr;
  if (saveIR) {
    std::string filename = "ir_output/" + name + ".mlir";
    std::error_code EC;
    llvm::raw_fd_ostream outFile(filename, EC);
    if (!EC) {
      module->print(outFile);
    }
  }
}

// ==================== TYPE HELPERS ====================

mlir::Type
MLIRCompiler::convertType(const mlir_edsl::TypeSpec &typeSpec) const {
  return builder->convertType(typeSpec);
}

bool MLIRCompiler::isValidParameterType(const mlir_edsl::TypeSpec &type) const {
  return type.has_scalar() || type.has_memref() || type.has_tensor();
}

bool MLIRCompiler::isValidReturnType(const mlir_edsl::TypeSpec &type) const {
  return type.has_scalar() || type.has_memref() || type.has_tensor();
}

// ==================== COMPILATION ====================

void MLIRCompiler::compileFunction(const mlir_edsl::FunctionDef &funcDef) {
  // Auto-invalidate JIT if adding function after finalization
  if (state == State::Finalized) {
    executor->clear();
    state = State::Building;
  }

  // Validate and extract parameters
  std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> params;
  for (const auto &param : funcDef.params()) {
    if (!isValidParameterType(param.type())) {
      throw std::runtime_error("Parameter '" + param.name() +
                               "': unsupported type");
    }
    params.push_back({param.name(), param.type()});
  }

  // Validate return type
  const auto &retType = funcDef.return_type();
  if (!isValidReturnType(retType)) {
    throw std::runtime_error("Unsupported return type");
  }

  mlir::Type returnType = convertType(retType);

  // Create function, build body, finalize
  createFunction(funcDef.name(), params, returnType);
  // Pass currentOutParam so array-producing ops write directly into the
  // Python-allocated output buffer, skipping the intermediate alloca+copy.
  mlir::Value result =
      builder->buildFromProtobufNode(funcDef.body(), currentOutParam);
  finalizeFunction(funcDef.name(), result);
}

// ==================== FINALIZATION ====================

void MLIRCompiler::ensureFinalized() {
  if (state == State::Finalized) {
    return;
  }

  const bool saveIR = std::getenv("SAVE_IR") != nullptr;
  const bool doCapture = saveIR || captureSnapshots;
  MLIRLowering lowering(mlirContext.get(), /*captureSnapshots=*/doCapture);

#ifdef MLIR_EDSL_CUDA_ENABLED
  if (target_ == Target::GPU) {
    GPULoweredModule gpuLowered;
    try {
      gpuLowered = lowering.lowerToGPUModule(*module);
    } catch (...) {
      failureIR_ = lowering.takeFailureIR();
      if (doCapture)
        loweringSnapshots = lowering.takeSnapshots();
      throw;
    }
    if (doCapture)
      loweringSnapshots = lowering.takeSnapshots();

    for (const auto &name : compiledFunctions) {
      gpuModules_[name] = gpuLowered;
      for (const auto &k : gpuLowered.kernels)
        gpuExecutor_->loadKernel(k.moduleName, k.ptxImage, k.funcName);
    }
    state = State::Finalized;
    return;
  }
#endif

  // CPU path
  LoweredModule lowered;
  try {
    lowered = lowering.lowerToLLVMModule(*module);
  } catch (...) {
    failureIR_ = lowering.takeFailureIR();
    if (doCapture)
      loweringSnapshots = lowering.takeSnapshots();
    throw;
  }

  if (doCapture)
    loweringSnapshots = lowering.takeSnapshots();

  std::vector<std::string> names(compiledFunctions.begin(),
                                 compiledFunctions.end());
  executor->compileModule(std::move(lowered.module), std::move(lowered.context),
                          names);

  state = State::Finalized;
}

// ==================== EXECUTION ====================

uintptr_t MLIRCompiler::getFunctionPointer(const std::string &name) {
  ensureFinalized();
  return executor->getFunctionPointer(name);
}

// ==================== STATE MANAGEMENT ====================

void MLIRCompiler::clear() {
  // Reset module
  module = mlir::ModuleOp::create(opBuilder->getUnknownLoc());
  opBuilder->setInsertionPointToEnd(module->getBody());

  // Clear function state
  currentFunction = nullptr;
  parameterMap.clear();
  functionTable.clear();
  compiledFunctions.clear();
  builder->clearValueCache();

  // Clear executor
  executor->clear();

#ifdef MLIR_EDSL_CUDA_ENABLED
  gpuModules_.clear();
  if (gpuExecutor_)
    gpuExecutor_->clear();
#endif

  // Clear lowering snapshots and failure IR
  loweringSnapshots.clear();
  failureIR_.clear();

  state = State::Building;
}

// ==================== INSPECTION ====================

bool MLIRCompiler::hasFunction(const std::string &name) const {
  return compiledFunctions.count(name) > 0;
}

std::vector<std::string> MLIRCompiler::listFunctions() const {
  return {compiledFunctions.begin(), compiledFunctions.end()};
}

std::string MLIRCompiler::getModuleIR() {
  std::string result;
  llvm::raw_string_ostream os(result);
  module->print(os);
  return result;
}

// ==================== TESTING UTILITIES ====================

void MLIRCompiler::injectTestFailure() {
  auto f64Type = mlir::Float64Type::get(mlirContext.get());
  auto funcType =
      mlir::FunctionType::get(mlirContext.get(), {}, {f64Type});

  mlir::OpBuilder::InsertionGuard guard(*opBuilder);
  opBuilder->setInsertionPointToEnd(module->getBody());

  auto func = opBuilder->create<mlir::func::FuncOp>(
      opBuilder->getUnknownLoc(), "__test_failure_inject__", funcType);
  auto *block = func.addEntryBlock();
  opBuilder->setInsertionPointToStart(block);
  // Return i32 where f64 is expected — type mismatch triggers verifier failure
  auto zero = opBuilder->create<mlir::arith::ConstantIntOp>(
      opBuilder->getUnknownLoc(), 0, 32);
  opBuilder->create<mlir::func::ReturnOp>(opBuilder->getUnknownLoc(),
                                          zero.getResult());
}

// ==================== GPU EXECUTION ====================

void MLIRCompiler::executeGPUFunction(
    const std::string &name,
    const std::vector<std::pair<const void *, std::vector<int64_t>>> &inputs,
    void *output,
    const std::vector<int64_t> &outputShape,
    size_t elementSize) {
#ifdef MLIR_EDSL_CUDA_ENABLED
  ensureFinalized();

  auto elemCount = [](const std::vector<int64_t> &shape) {
    int64_t n = 1;
    for (auto d : shape) n *= d;
    return n;
  };

  // Allocate device memory and copy inputs H2D
  std::vector<CUdeviceptr> dInputs;
  dInputs.reserve(inputs.size());
  for (const auto &[hostPtr, shape] : inputs) {
    CUdeviceptr d = gpuExecutor_->allocDevice((size_t)elemCount(shape) * elementSize);
    gpuExecutor_->copyH2D(d, hostPtr, (size_t)elemCount(shape) * elementSize);
    dInputs.push_back(d);
  }
  size_t outBytes = (size_t)elemCount(outputShape) * elementSize;
  CUdeviceptr dOutput = gpuExecutor_->allocDevice(outBytes);

  // Helper: push a flattened memref descriptor onto params.
  // Both storage vectors must outlive launchKernel; we pass them by ref so
  // that push_back never invalidates pointers into an already-appended block
  // (we pre-reserve below).
  auto pushMemRef = [](CUdeviceptr dPtr,
                       const std::vector<int64_t> &shape,
                       std::vector<void *> &params,
                       std::vector<uint64_t> &ptrStore,
                       std::vector<int64_t> &intStore) {
    uint64_t ptrVal = (uint64_t)dPtr;
    ptrStore.push_back(ptrVal);
    params.push_back(&ptrStore.back());
    ptrStore.push_back(ptrVal);
    params.push_back(&ptrStore.back());

    intStore.push_back(0); // offset
    params.push_back(&intStore.back());

    for (auto s : shape) {
      intStore.push_back(s);
      params.push_back(&intStore.back());
    }
    for (size_t d = 0; d < shape.size(); ++d) {
      int64_t stride = 1;
      for (size_t k = d + 1; k < shape.size(); ++k) stride *= shape[k];
      intStore.push_back(stride);
      params.push_back(&intStore.back());
    }
  };

  const GPULoweredModule &lowered = gpuModules_.at(name);

  // Launch each kernel in order (e.g. fill then matmul).
  for (const auto &kernel : lowered.kernels) {
    // Reserve enough storage so push_back never reallocates mid-build.
    // Worst case: 2 ptr fields + (1+rank+rank) int fields per MemRef arg,
    // plus 1 field per scalar. Over-allocate generously.
    std::vector<void *>    params;
    std::vector<uint64_t>  ptrStore;
    std::vector<int64_t>   intStore;
    std::vector<float>     f32Store;
    size_t nArgs = kernel.args.size();
    ptrStore.reserve(nArgs * 2);
    intStore.reserve(nArgs * 8);
    f32Store.reserve(nArgs);

    for (const auto &arg : kernel.args) {
      switch (arg.kind) {
      case GPUKernelArg::Kind::I64:
        intStore.push_back(arg.i64Val);
        params.push_back(&intStore.back());
        break;
      case GPUKernelArg::Kind::F32:
        f32Store.push_back(arg.f32Val);
        params.push_back(&f32Store.back());
        break;
      case GPUKernelArg::Kind::InputMemRef:
        pushMemRef(dInputs[arg.paramIdx], arg.shape,
                   params, ptrStore, intStore);
        break;
      case GPUKernelArg::Kind::OutputMemRef:
        pushMemRef(dOutput, arg.shape, params, ptrStore, intStore);
        break;
      default:
        llvm::report_fatal_error("Unhandled GPUKernelArg::Kind");
      }
    }

    gpuExecutor_->launchKernel(kernel.moduleName, params.data(),
                                kernel.gridX, kernel.gridY, kernel.gridZ,
                                kernel.blockX, kernel.blockY, kernel.blockZ);
  }

  gpuExecutor_->copyD2H(output, dOutput, outBytes);

  for (auto d : dInputs) gpuExecutor_->freeDevice(d);
  gpuExecutor_->freeDevice(dOutput);
#else
  (void)name; (void)inputs; (void)output;
  (void)outputShape; (void)elementSize;
  throw std::runtime_error(
      "GPU support not compiled in (rebuild with -DMLIR_EDSL_CUDA=ON)");
#endif
}

// ==================== CONFIGURATION ====================

void MLIRCompiler::setOptimizationLevel(OptLevel level) {
  optimizationLevel = level;

  MLIRExecutor::OptLevel execLevel;
  switch (level) {
  case OptLevel::O0:
    execLevel = MLIRExecutor::OptLevel::O0;
    break;
  case OptLevel::O2:
    execLevel = MLIRExecutor::OptLevel::O2;
    break;
  case OptLevel::O3:
    execLevel = MLIRExecutor::OptLevel::O3;
    break;
  }
  executor->setOptimizationLevel(execLevel);
}

} // namespace mlir_edsl
