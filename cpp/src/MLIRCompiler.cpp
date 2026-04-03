#include "mlir_edsl/MLIRCompiler.h"
#include "ast.pb.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRExecutor.h"
#include "mlir_edsl/MLIRLowering.h"

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
  // Tensor out-params are kept as tensor (bufferization handles the ABI).
  const bool aggregateReturn = mlir::isa<mlir::MemRefType>(returnType) ||
                               mlir::isa<mlir::RankedTensorType>(returnType);

  if (aggregateReturn) {
    paramTypes.push_back(returnType);
  }

  auto funcType = opBuilder->getFunctionType(
      paramTypes,
      aggregateReturn ? mlir::TypeRange{} : mlir::TypeRange{returnType});
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

  // Store the out-param block arg for use in finalizeFunction.
  // For tensor out-params, mark writable so the bufferizer can write in-place.
  if (aggregateReturn) {
    auto outArgIdx = params.size();
    currentOutParam = entryBlock->getArgument(outArgIdx);
    if (mlir::isa<mlir::RankedTensorType>(returnType)) {
      currentFunction.setArgAttr(outArgIdx, "bufferization.writable",
                                 opBuilder->getBoolAttr(true));
    }
  }
}

void MLIRCompiler::finalizeFunction(const std::string &name,
                                    mlir::Value result) {
  if (!currentFunction) {
    throw std::runtime_error("No current function to finish");
  }

  // Aggregate return: write result into the caller-allocated out-param.
  if (currentOutParam) {
    auto loc = opBuilder->getUnknownLoc();
    if (mlir::isa<mlir::RankedTensorType>(result.getType())) {
      // Tensor result: materialize into the out-param.
      // When dest is a tensor, a result type is required (even if unused).
      // The bufferizer eliminates this as a no-op when result already aliases
      // the out-param's buffer.
      mlir::Type destType = currentOutParam.getType();
      bool tensorDest = mlir::isa<mlir::RankedTensorType>(destType);
      mlir::Type resultType = tensorDest ? destType : mlir::Type{};
      // restrict and writable are only valid for memref destinations.
      opBuilder->create<mlir::bufferization::MaterializeInDestinationOp>(
          loc, resultType, result, currentOutParam,
          /*restrict=*/!tensorDest, /*writable=*/!tensorDest);
    } else if (mlir::isa<mlir::MemRefType>(result.getType())) {
      // MemRef result: copy only when result is not already the out-param.
      if (result != currentOutParam)
        opBuilder->create<mlir::memref::CopyOp>(loc, result, currentOutParam);
    } else {
      throw std::runtime_error(
          "finalizeFunction: out-param set for non-aggregate type");
    }
    opBuilder->create<mlir::func::ReturnOp>(loc);
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

  // Lower MLIR to LLVM module
  const bool saveIR = std::getenv("SAVE_IR") != nullptr;
  const bool doCapture = saveIR || captureSnapshots;
  MLIRLowering lowering(mlirContext.get(), /*captureSnapshots=*/doCapture);

  LoweredModule lowered;
  try {
    lowered = lowering.lowerToLLVMModule(*module);
  } catch (...) {
    // Extract failure state before lowering is destroyed on stack unwind
    failureIR_ = lowering.takeFailureIR();
    if (doCapture)
      loweringSnapshots = lowering.takeSnapshots();
    throw;
  }

  if (doCapture)
    loweringSnapshots = lowering.takeSnapshots();

  // JIT compile with function names to look up
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
