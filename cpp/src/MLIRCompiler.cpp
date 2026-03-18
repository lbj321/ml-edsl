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

  // Convert parameter types: tensor params are always pre-converted to memref
  // in the function signature, keeping a pure-memref boundary for all
  // functions. to_tensor is inserted at the top of the body so the function
  // body still operates on tensors.
  std::vector<mlir::Type> paramTypes;
  for (const auto &[paramName, typeSpec] : params) {
    mlir::Type t = builder->convertType(typeSpec);
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(t))
      t = mlir::MemRefType::get(tensorType.getShape(),
                                tensorType.getElementType());
    paramTypes.push_back(t);
  }

  // Aggregate return: append a hidden memref out-param and use void return.
  // Python allocates the output buffer and passes it as a memref descriptor.
  const bool aggregateReturn = mlir::isa<mlir::MemRefType>(returnType) ||
                               mlir::isa<mlir::RankedTensorType>(returnType);

  mlir::Type outParamType = returnType;
  if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(returnType)) {
    outParamType = mlir::MemRefType::get(tensorType.getShape(),
                                         tensorType.getElementType());
  }
  if (aggregateReturn) {
    paramTypes.push_back(outParamType);
  }

  auto funcType = opBuilder->getFunctionType(
      paramTypes,
      aggregateReturn ? mlir::TypeRange{} : mlir::TypeRange{returnType});
  currentFunction = opBuilder->create<mlir::func::FuncOp>(
      opBuilder->getUnknownLoc(), name, funcType);

  functionTable[name] = currentFunction;

  auto *entryBlock = currentFunction.addEntryBlock();
  opBuilder->setInsertionPointToStart(entryBlock);

  // Map parameter names to block arguments.
  // Tensor params were converted to memref in the signature — insert to_tensor
  // so the body still operates on tensors.
  auto loc = opBuilder->getUnknownLoc();
  for (size_t i = 0; i < params.size(); i++) {
    mlir::Value arg = entryBlock->getArgument(i);
    mlir::Type originalType = builder->convertType(params[i].second);
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(originalType)) {
      auto toTensor = opBuilder->create<mlir::bufferization::ToTensorOp>(
          loc, tensorType, arg, /*restrict=*/true, /*writable=*/true);
      parameterMap[params[i].first] = toTensor.getResult();
    } else {
      parameterMap[params[i].first] = arg;
    }
  }

  // Store the out-param block arg (memref) for use in finalizeFunction.
  if (aggregateReturn) {
    currentOutParam = entryBlock->getArgument(params.size());
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
      // Tensor result: materialize into the Python-owned memref out-param.
      // restrict+writable: the out-param is the sole alias and is writable,
      // enabling empty-tensor-elimination to skip the memcpy when possible.
      opBuilder->create<mlir::bufferization::MaterializeInDestinationOp>(
          loc, /*result=*/mlir::Type{}, result, currentOutParam,
          /*restrict=*/true, /*writable=*/true);
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
  auto lowered = lowering.lowerToLLVMModule(*module);
  if (doCapture) {
    loweringSnapshots = lowering.takeSnapshots();
  }

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

  // Clear lowering snapshots
  loweringSnapshots.clear();

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
