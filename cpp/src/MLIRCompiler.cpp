#include "mlir_edsl/MLIRCompiler.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRExecutor.h"
#include "mlir_edsl/MLIRLowering.h"
#include "ast.pb.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
  executor->initialize();
  executor->setOptimizationLevel(MLIRExecutor::OptLevel::O2);
}

MLIRCompiler::~MLIRCompiler() = default;

// ==================== FUNCTION STATE ====================

void MLIRCompiler::resetFunctionState() {
  currentFunction = nullptr;
  parameterMap.clear();
  builder->clearValueCache();
  opBuilder->setInsertionPointToEnd(module->getBody());
}

// ==================== FUNCTION BUILDING ====================

void MLIRCompiler::createFunction(
    const std::string& name,
    const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>>& params,
    mlir::Type returnType) {
  resetFunctionState();

  // Convert parameter types
  std::vector<mlir::Type> paramTypes;
  for (const auto& [paramName, typeSpec] : params) {
    paramTypes.push_back(builder->convertType(typeSpec));
  }

  auto funcType = opBuilder->getFunctionType(paramTypes, {returnType});
  currentFunction = opBuilder->create<mlir::func::FuncOp>(
      opBuilder->getUnknownLoc(), name, funcType);

  functionTable[name] = currentFunction;

  auto *entryBlock = currentFunction.addEntryBlock();
  opBuilder->setInsertionPointToStart(entryBlock);

  // Map parameter names to block arguments
  for (size_t i = 0; i < params.size(); i++) {
    parameterMap[params[i].first] = entryBlock->getArgument(i);
  }
}

void MLIRCompiler::finalizeFunction(const std::string& name,
                                     mlir::Value result) {
  if (!currentFunction) {
    throw std::runtime_error("No current function to finish");
  }
  opBuilder->create<mlir::func::ReturnOp>(opBuilder->getUnknownLoc(), result);
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

mlir::Type MLIRCompiler::convertType(const mlir_edsl::TypeSpec& typeSpec) const {
  return builder->convertType(typeSpec);
}

bool MLIRCompiler::isValidParameterType(const mlir_edsl::TypeSpec& type) const {
  return type.has_scalar();
}

bool MLIRCompiler::isValidReturnType(const mlir_edsl::TypeSpec& type) const {
  return type.has_scalar();
}

// ==================== COMPILATION ====================

void MLIRCompiler::compileFunction(const std::string& functionDefBytes) {
  // Auto-invalidate JIT if adding function after finalization
  if (state == State::Finalized) {
    executor->clearJit();
    state = State::Building;
  }

  // Deserialize FunctionDef
  mlir_edsl::FunctionDef funcDef;
  if (!funcDef.ParseFromString(functionDefBytes)) {
    throw std::runtime_error("Failed to parse FunctionDef protobuf");
  }

  // Validate and extract parameters
  std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> params;
  for (const auto& param : funcDef.params()) {
    if (!isValidParameterType(param.type())) {
      throw std::runtime_error("Parameter '" + param.name() + "': unsupported type");
    }
    params.push_back({param.name(), param.type()});
  }

  // Validate return type
  const auto& retType = funcDef.return_type();
  if (!isValidReturnType(retType)) {
    throw std::runtime_error("Unsupported return type");
  }

  mlir::Type returnType = convertType(retType);

  // Create function, build body, finalize
  createFunction(funcDef.name(), params, returnType);
  mlir::Value result = builder->buildFromProtobufNode(funcDef.body());
  finalizeFunction(funcDef.name(), result);

  // Build and store serialized signature for ctypes
  mlir_edsl::FunctionSignature sig;
  sig.set_name(funcDef.name());
  for (const auto& param : funcDef.params()) {
    sig.add_param_types()->CopyFrom(param.type());
  }
  sig.mutable_return_type()->CopyFrom(funcDef.return_type());
  signatures[funcDef.name()] = sig.SerializeAsString();

  // Register with executor so compileModule() can cache function pointers
  executor->registerFunctionSignature(sig);
}

// ==================== FINALIZATION ====================

void MLIRCompiler::ensureFinalized() {
  if (state == State::Finalized) {
    return;
  }

  // Lower MLIR to LLVM IR
  MLIRLowering lowering(mlirContext.get());
  std::string llvmIR = lowering.lowerToLLVMIR(*module);

  // JIT compile
  if (!executor->compileModule(llvmIR)) {
    throw std::runtime_error(
        "JIT compilation failed: " + executor->getLastError());
  }

  state = State::Finalized;
}

// ==================== EXECUTION ====================

uintptr_t MLIRCompiler::getFunctionPointer(const std::string& name) {
  ensureFinalized();
  return executor->getFunctionPointer(name);
}

std::string MLIRCompiler::getFunctionSignature(const std::string& name) const {
  auto it = signatures.find(name);
  if (it == signatures.end()) {
    throw std::runtime_error("Function signature not found: " + name);
  }
  return it->second;
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

  // Clear executor and signatures
  executor->clearAll();
  signatures.clear();

  state = State::Building;
}

// ==================== INSPECTION ====================

bool MLIRCompiler::hasFunction(const std::string& name) const {
  return compiledFunctions.count(name) > 0;
}

std::vector<std::string> MLIRCompiler::listFunctions() const {
  return {compiledFunctions.begin(), compiledFunctions.end()};
}

// ==================== CONFIGURATION ====================

void MLIRCompiler::setOptimizationLevel(OptLevel level) {
  optimizationLevel = level;

  MLIRExecutor::OptLevel execLevel;
  switch (level) {
    case OptLevel::O0: execLevel = MLIRExecutor::OptLevel::O0; break;
    case OptLevel::O2: execLevel = MLIRExecutor::OptLevel::O2; break;
    case OptLevel::O3: execLevel = MLIRExecutor::OptLevel::O3; break;
  }
  executor->setOptimizationLevel(execLevel);
}

} // namespace mlir_edsl
