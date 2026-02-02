#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir_edsl/SCFBuilder.h"
#include "mlir_edsl/MemRefBuilder.h"

#include "ast.pb.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir_edsl/MLIRLowering.h"
#include "llvm/Support/raw_ostream.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <cstdlib>

namespace mlir_edsl {

MLIRBuilder::MLIRBuilder() {
  context = std::make_unique<mlir::MLIRContext>();

  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();

  builder = std::make_unique<mlir::OpBuilder>(context.get());

  // Initialize dialect builders
  arithBuilder = std::make_unique<mlir_edsl::ArithBuilder>(*builder);
  scfBuilder = std::make_unique<mlir_edsl::SCFBuilder>(*builder);
  memrefBuilder = std::make_unique<mlir_edsl::MemRefBuilder>(*builder, context.get(), this, arithBuilder.get(), scfBuilder.get());
}

MLIRBuilder::~MLIRBuilder() = default;

void MLIRBuilder::initializeModule() {
  module = mlir::ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToEnd(module->getBody());
}

mlir::Value MLIRBuilder::buildFromProtobufNode(const mlir_edsl::ASTNode &node) {
  // Two-tier dispatch for scalability (switch on category, then specific type)
  switch (node.node_case()) {
    case mlir_edsl::ASTNode::kScalar:
      return buildFromScalarNode(node.scalar());
    case mlir_edsl::ASTNode::kArray:
      return buildFromArrayNode(node.array());
    case mlir_edsl::ASTNode::kControlFlow:
      return buildFromControlFlowNode(node.control_flow());
    case mlir_edsl::ASTNode::kFunction:
      return buildFromFunctionNode(node.function());
    case mlir_edsl::ASTNode::kBinding:
      return buildFromBindingNode(node.binding());
    default:
      throw std::runtime_error("Unknown AST node category");
  }
}

mlir::Value MLIRBuilder::buildFromScalarNode(const mlir_edsl::ScalarNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::ScalarNode::kConstant:
      return handleConstant(node.constant());
    case mlir_edsl::ScalarNode::kBinaryOp:
      return handleBinaryOp(node.binary_op());
    case mlir_edsl::ScalarNode::kCompareOp:
      return handleCompareOp(node.compare_op());
    case mlir_edsl::ScalarNode::kCastOp:
      return handleCastOp(node.cast_op());
    default:
      throw std::runtime_error("Unknown scalar node type");
  }
}

mlir::Value MLIRBuilder::buildFromArrayNode(const mlir_edsl::ArrayNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::ArrayNode::kLiteral:
      return memrefBuilder->buildArrayLiteral(node.literal());
    case mlir_edsl::ArrayNode::kAccess:
      return memrefBuilder->buildArrayAccess(node.access());
    case mlir_edsl::ArrayNode::kStore:
      return memrefBuilder->buildArrayStore(node.store());
    case mlir_edsl::ArrayNode::kBinaryOp:
      return memrefBuilder->buildArrayBinaryOp(node.binary_op());
    default:
      throw std::runtime_error("Unknown array node type");
  }
}

mlir::Value MLIRBuilder::buildFromControlFlowNode(const mlir_edsl::ControlFlowNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::ControlFlowNode::kIfOp:
      return handleIfOp(node.if_op());
    // case mlir_edsl::ControlFlowNode::kForLoop:
    //   return handleForLoopOp(node.for_loop());
    default:
      throw std::runtime_error("Unknown control flow node type");
  }
}

mlir::Value MLIRBuilder::buildFromFunctionNode(const mlir_edsl::FunctionNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::FunctionNode::kParameter:
      return handleParameter(node.parameter());
    case mlir_edsl::FunctionNode::kCall:
      return handleCallOp(node.call());
    default:
      throw std::runtime_error("Unknown function node type");
  }
}

mlir::Value MLIRBuilder::buildFromBindingNode(const mlir_edsl::BindingNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::BindingNode::kLet:
      return handleLetBinding(node.let());
    case mlir_edsl::BindingNode::kRef:
      return handleValueRef(node.ref());
    default:
      throw std::runtime_error("Unknown binding node type");
  }
}

// ==================== AST Node Handlers ====================

// Binding handlers
mlir::Value MLIRBuilder::handleLetBinding(const mlir_edsl::LetBinding &binding) {
  int64_t nodeId = binding.node_id();
  mlir::Value result = buildFromProtobufNode(binding.value());
  valueCache[nodeId] = result;
  return result;
}

mlir::Value MLIRBuilder::handleValueRef(const mlir_edsl::ValueReference &ref) {
  int64_t nodeId = ref.node_id();
  auto it = valueCache.find(nodeId);
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("Reference to unbound value ID: " + std::to_string(nodeId));
}

// Scalar handlers
mlir::Value MLIRBuilder::handleConstant(const mlir_edsl::Constant &constant) {
  const auto &typeSpec = constant.type();

  if (!typeSpec.has_scalar()) {
    throw std::runtime_error("Constant must have scalar type");
  }

  switch (typeSpec.scalar().kind()) {
  case mlir_edsl::ScalarTypeSpec::I32:
    return arithBuilder->buildConstant(constant.int_value());
  case mlir_edsl::ScalarTypeSpec::F32:
    return arithBuilder->buildConstant(constant.float_value());
  case mlir_edsl::ScalarTypeSpec::I1:
    return arithBuilder->buildConstant(constant.bool_value());
  default:
    throw std::runtime_error("Unsupported constant type");
  }
}

mlir::Value MLIRBuilder::handleBinaryOp(const mlir_edsl::BinaryOp &binop) {
  mlir::Value left = buildFromProtobufNode(binop.left());
  mlir::Value right = buildFromProtobufNode(binop.right());

  mlir::Type targetType = convertType(binop.result_type());
  auto [promotedLeft, promotedRight] = promoteToMatchDataType(left, right, targetType);

  switch (binop.op_type()) {
  case mlir_edsl::BinaryOpType::ADD:
    return arithBuilder->buildAdd(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::SUB:
    return arithBuilder->buildSub(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::MUL:
    return arithBuilder->buildMul(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::DIV:
    return arithBuilder->buildDiv(promotedLeft, promotedRight);
  default:
    throw std::runtime_error("Unsupported binary operation");
  }
}

mlir::Value MLIRBuilder::handleCompareOp(const mlir_edsl::CompareOp &cmp) {
  mlir::Value left = buildFromProtobufNode(cmp.left());
  mlir::Value right = buildFromProtobufNode(cmp.right());

  mlir::Type targetType = convertType(cmp.operand_type());
  auto [promotedLhs, promotedRhs] = promoteToMatchDataType(left, right, targetType);

  return arithBuilder->buildCompare(cmp.predicate(), promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::handleCastOp(const mlir_edsl::CastOp &cast) {
  mlir::Value sourceValue = buildFromProtobufNode(cast.value());
  mlir::Type targetType = convertType(cast.target_type());
  return arithBuilder->buildCast(sourceValue, targetType);
}

// Control flow handlers
mlir::Value MLIRBuilder::handleIfOp(const mlir_edsl::IfOp &ifop) {
  mlir::Value cond = buildFromProtobufNode(ifop.condition());
  mlir::Type resultType = convertType(ifop.result_type());

  auto buildThen = [this, &ifop]() {
    return buildFromProtobufNode(ifop.then_value());
  };
  auto buildElse = [this, &ifop]() {
    return buildFromProtobufNode(ifop.else_value());
  };

  return scfBuilder->buildIf(cond, buildThen, buildElse, resultType);
}

// mlir::Value MLIRBuilder::handleForLoopOp(const mlir_edsl::ForLoopOp &forloop) {
//   mlir::Value start = buildFromProtobufNode(forloop.start());
//   mlir::Value end = buildFromProtobufNode(forloop.end());
//   mlir::Value step = buildFromProtobufNode(forloop.step());
//   mlir::Value init_value = buildFromProtobufNode(forloop.init_value());
//   return scfBuilder->buildForWithOp(start, end, step, init_value, forloop.operation());
// }

// Function handlers
mlir::Value MLIRBuilder::handleParameter(const mlir_edsl::Parameter &param) {
  return getParameter(param.name());
}

mlir::Value MLIRBuilder::handleCallOp(const mlir_edsl::CallOp &call) {
  std::vector<mlir::Value> args;
  for (const auto &arg : call.args()) {
    args.push_back(buildFromProtobufNode(arg));
  }
  return callFunction(call.func_name(), args);
}

std::string MLIRBuilder::getMLIRString() {
  std::string result;
  llvm::raw_string_ostream stream(result);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  flags.printGenericOpForm(false);
  module->print(stream, flags);
  return result;
}

std::string MLIRBuilder::getLLVMIRString() {
  MLIRLowering lowering(context.get());
  return lowering.lowerToLLVMIR(*module);
}

void MLIRBuilder::reset() {
  // Reset builder state without destroying the module
  currentFunction = nullptr;
  parameterMap.clear();
  valueCache.clear();  // Clear SSA value cache between functions

  // Move insertion point back to module level
  builder->setInsertionPointToEnd(module->getBody());
}

bool MLIRBuilder::isIntegerType(mlir::Type type) const {
  return mlir::isa<mlir::IntegerType>(type);
}

bool MLIRBuilder::isFloatType(mlir::Type type) const {
  return mlir::isa<mlir::FloatType>(type);
}

// Explicit type promotion - Python provides the target type
std::pair<mlir::Value, mlir::Value>
MLIRBuilder::promoteToMatchDataType(mlir::Value lhs, mlir::Value rhs,
                           mlir::Type targetType) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();

  // Promote left operand if needed
  if (lhsType != targetType && isIntegerType(lhsType) &&
      isFloatType(targetType)) {
    lhs = arithBuilder->buildCast(lhs, targetType);
  }

  // Promote right operand if needed
  if (rhsType != targetType && isIntegerType(rhsType) &&
      isFloatType(targetType)) {
    rhs = arithBuilder->buildCast(rhs, targetType);
  }

  return {lhs, rhs};
}

// ==================== ALGEBRAIC TYPE SYSTEM ====================

mlir::Type MLIRBuilder::convertType(const mlir_edsl::TypeSpec &typeSpec) const {
  switch (typeSpec.type_kind_case()) {
    case mlir_edsl::TypeSpec::kScalar:
      return convertScalarType(typeSpec.scalar());

    case mlir_edsl::TypeSpec::kMemref:
      return convertMemRefType(typeSpec.memref());

    case mlir_edsl::TypeSpec::TYPE_KIND_NOT_SET:
      throw std::runtime_error("TypeSpec has no type set");

    default:
      throw std::runtime_error("Unknown TypeSpec kind: " +
                               std::to_string(typeSpec.type_kind_case()));
  }
}

mlir::Type MLIRBuilder::convertScalarType(
    const mlir_edsl::ScalarTypeSpec &scalarSpec) const {
  switch (scalarSpec.kind()) {
    case mlir_edsl::ScalarTypeSpec::I32:
      return builder->getI32Type();

    case mlir_edsl::ScalarTypeSpec::F32:
      return builder->getF32Type();

    case mlir_edsl::ScalarTypeSpec::I1:
      return builder->getI1Type();

    default:
      throw std::runtime_error("Unknown ScalarTypeSpec kind: " +
                               std::to_string(static_cast<int>(scalarSpec.kind())));
  }
}

mlir::Type MLIRBuilder::convertMemRefType(
    const mlir_edsl::MemRefTypeSpec &memrefSpec) const {

  // Recursively convert element type - THIS IS THE KEY!
  mlir::Type elementType = convertType(memrefSpec.element_type());

  // Extract shape dimensions
  llvm::SmallVector<int64_t> shape(
      memrefSpec.shape().begin(),
      memrefSpec.shape().end()
  );

  // Validate dimensions
  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("Only 1D, 2D, and 3D arrays supported, got " +
                             std::to_string(shape.size()) + "D");
  }

  // Create memref type: memref<2x3xi32>
  return mlir::MemRefType::get(shape, elementType);
}

// ==================== Infrastructure Utilities ====================

mlir::Value MLIRBuilder::buildIndexConstant(int64_t value) {
  auto loc = builder->getUnknownLoc();
  return builder->create<mlir::arith::ConstantIndexOp>(loc, value);
}

mlir::Value MLIRBuilder::castToIndexType(mlir::Value value) {
  // If already index type, return as-is
  if (value.getType().isIndex()) {
    return value;
  }

  // Convert integer to index type
  if (mlir::isa<mlir::IntegerType>(value.getType())) {
    auto loc = builder->getUnknownLoc();
    return builder->create<mlir::arith::IndexCastOp>(
        loc, builder->getIndexType(), value);
  }

  throw std::runtime_error("Cannot cast to index type: unsupported source type");
}

mlir::Value MLIRBuilder::getParameter(const std::string &name) {
  auto it = parameterMap.find(name);
  if (it != parameterMap.end()) {
    return it->second;
  }
  throw std::runtime_error("Parameter not found: " + name);
}

// ==================== Type Validation ====================

bool MLIRBuilder::isValidParameterType(const mlir_edsl::TypeSpec &type) const {
  return type.has_scalar();
}

bool MLIRBuilder::isValidReturnType(const mlir_edsl::TypeSpec &type) const {
  return type.has_scalar();
}

void MLIRBuilder::compileFunctionFromDef(const mlir_edsl::FunctionDef &func_def) {
  // Validate and extract parameters
  std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> params;
  for (const auto &param : func_def.params()) {
    if (!isValidParameterType(param.type())) {
      throw std::runtime_error("Parameter '" + param.name() + "': unsupported type");
    }
    params.push_back({param.name(), param.type()});
  }

  // Validate return type
  const auto &retType = func_def.return_type();
  if (!isValidReturnType(retType)) {
    throw std::runtime_error("Unsupported return type");
  }

  mlir::Type returnType = convertType(retType);

  // Compile function
  createFunction(func_def.name(), params, returnType);
  mlir::Value result = buildFromProtobufNode(func_def.body());
  finalizeFunction(func_def.name(), result);
}

void MLIRBuilder::createFunction(
    const std::string &name,
    const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> &params,
    mlir::Type returnType) {

  // Reset builder state from previous function
  reset();

  // Convert parameter types using new unified type conversion
  std::vector<mlir::Type> paramTypes;
  for (const auto &[paramName, typeSpec] : params) {
    paramTypes.push_back(convertType(typeSpec));
  }

  // returnType is already an mlir::Type - use it directly
  auto funcType = builder->getFunctionType(paramTypes, {returnType});
  currentFunction = builder->create<mlir::func::FuncOp>(
      builder->getUnknownLoc(), name, funcType);

  functionTable[name] = currentFunction;

  auto *entryBlock = currentFunction.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);

  // Map parameter names to block arguments
  for (size_t i = 0; i < params.size(); i++) {
    parameterMap[params[i].first] = entryBlock->getArgument(i);
  }
}

void MLIRBuilder::finalizeFunction(const std::string &name,
                                   mlir::Value result) {
  if (!currentFunction) {
    throw std::runtime_error("No current function to finish");
  }
  builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc(), result);
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

mlir::Value MLIRBuilder::callFunction(const std::string &name,
                                      const std::vector<mlir::Value> &args) {
  auto it = functionTable.find(name);
  if (it == functionTable.end()) {
    throw std::runtime_error("Function not found: " + name);
  }
  auto funcOp = it->second;
  return builder
      ->create<mlir::func::CallOp>(builder->getUnknownLoc(), funcOp, args)
      .getResult(0);
}

bool MLIRBuilder::hasFunction(const std::string &name) const {
  return compiledFunctions.count(name) > 0;
}

void MLIRBuilder::clearModule() {
  module = mlir::ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToEnd(module->getBody());

  compiledFunctions.clear();
  parameterMap.clear();
  functionTable.clear();
  currentFunction = nullptr;
}

std::vector<std::string> MLIRBuilder::listFunctions() const {
  return {compiledFunctions.begin(), compiledFunctions.end()};
}

} // namespace mlir_edsl