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
  arithBuilder = std::make_unique<mlir_edsl::ArithBuilder>(*builder, context.get(), this);
  scfBuilder = std::make_unique<mlir_edsl::SCFBuilder>(*builder, context.get(), this, arithBuilder.get());
  memrefBuilder = std::make_unique<mlir_edsl::MemRefBuilder>(*builder, context.get(), this, arithBuilder.get(), scfBuilder.get());
}

MLIRBuilder::~MLIRBuilder() {
  if (currentFunction) {
    currentFunction = nullptr;
  }
  if (module) {
    module = nullptr;
  }
  builder.reset();
  context.reset();
}

void MLIRBuilder::initializeModule() {
  module = mlir::ModuleOp::create(builder->getUnknownLoc());
  builder->setInsertionPointToEnd(module.getBody());
}

mlir::Value MLIRBuilder::buildFromProtobufNode(const mlir_edsl::ASTNode &node) {
  // Cross-cutting concerns (value caching, parameters, function calls)
  if (node.has_let_binding()) return handleLetBinding(node);
  if (node.has_value_ref()) return handleValueRef(node);
  if (node.has_parameter()) return handleParameter(node);
  if (node.has_call_op()) return handleCallOp(node);

  // Arithmetic operations
  if (node.has_constant()) return handleConstant(node);
  if (node.has_binary_op()) return handleBinaryOp(node);
  if (node.has_compare_op()) return handleCompareOp(node);
  if (node.has_cast_op()) return handleCastOp(node);

  // Control flow operations
  if (node.has_if_op()) return handleIfOp(node);
  // if (node.has_for_loop_op()) return handleForLoopOp(node);
  if (node.has_while_loop_op()) return handleWhileLoopOp(node);

  // Memory operations
  if (node.has_array_literal()) return handleArrayLiteral(node);
  if (node.has_array_access()) return handleArrayAccess(node);
  if (node.has_array_store()) return handleArrayStore(node);
  if (node.has_array_binary_op()) return handleArrayBinaryOp(node);

  throw std::runtime_error("Unknown protobuf node type");
}

// ==================== AST Node Handlers ====================

mlir::Value MLIRBuilder::handleLetBinding(const mlir_edsl::ASTNode &node) {
  int64_t nodeId = node.let_binding().node_id();
  mlir::Value result = buildFromProtobufNode(node.let_binding().value());
  valueCache[nodeId] = result;
  return result;
}

mlir::Value MLIRBuilder::handleValueRef(const mlir_edsl::ASTNode &node) {
  int64_t nodeId = node.value_ref().node_id();
  auto it = valueCache.find(nodeId);
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("Reference to unbound value ID: " + std::to_string(nodeId));
}

mlir::Value MLIRBuilder::handleConstant(const mlir_edsl::ASTNode &node) {
  const auto &constant = node.constant();
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
    // Boolean constants - use i32 constant with 0 or 1
    return arithBuilder->buildConstant(constant.bool_value() ? 1 : 0);
  default:
    throw std::runtime_error("Unsupported constant type");
  }
}

mlir::Value MLIRBuilder::handleParameter(const mlir_edsl::ASTNode &node) {
  const auto &param = node.parameter();
  return getParameter(param.name());
}

mlir::Value MLIRBuilder::handleBinaryOp(const mlir_edsl::ASTNode &node) {
  const auto &binop = node.binary_op();
  mlir::Value left = buildFromProtobufNode(binop.left());
  mlir::Value right = buildFromProtobufNode(binop.right());

  mlir::Type targetType = convertType(binop.result_type());
  auto [promotedLeft, promotedRight] = promoteToType(left, right, targetType);

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

mlir::Value MLIRBuilder::handleCompareOp(const mlir_edsl::ASTNode &node) {
  const auto &cmp = node.compare_op();
  mlir::Value left = buildFromProtobufNode(cmp.left());
  mlir::Value right = buildFromProtobufNode(cmp.right());

  mlir::Type targetType = convertType(cmp.operand_type());
  auto [promotedLhs, promotedRhs] = promoteToType(left, right, targetType);

  return arithBuilder->buildCompare(cmp.predicate(), promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::handleIfOp(const mlir_edsl::ASTNode &node) {
  const auto &ifop = node.if_op();
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

mlir::Value MLIRBuilder::handleCallOp(const mlir_edsl::ASTNode &node) {
  const auto &call = node.call_op();
  std::vector<mlir::Value> args;
  for (const auto &arg : call.args()) {
    args.push_back(buildFromProtobufNode(arg));
  }
  return callFunction(call.func_name(), args);
}

// mlir::Value MLIRBuilder::handleForLoopOp(const mlir_edsl::ASTNode &node) {
//   const auto &forloop = node.for_loop_op();
//   mlir::Value start = buildFromProtobufNode(forloop.start());
//   mlir::Value end = buildFromProtobufNode(forloop.end());
//   mlir::Value step = buildFromProtobufNode(forloop.step());
//   mlir::Value init_value = buildFromProtobufNode(forloop.init_value());
//   return scfBuilder->buildForWithOp(start, end, step, init_value, forloop.operation());
// }

mlir::Value MLIRBuilder::handleWhileLoopOp(const mlir_edsl::ASTNode &node) {
  const auto &whileloop = node.while_loop_op();
  mlir::Value init_value = buildFromProtobufNode(whileloop.init_value());
  mlir::Value target = buildFromProtobufNode(whileloop.target());
  return scfBuilder->buildWhileWithOp(init_value, target, whileloop.operation(),
                                      whileloop.predicate());
}

mlir::Value MLIRBuilder::handleCastOp(const mlir_edsl::ASTNode &node) {
  const auto &cast = node.cast_op();
  mlir::Value sourceValue = buildFromProtobufNode(cast.value());
  mlir::Type targetType = convertType(cast.target_type());
  return arithBuilder->buildCast(sourceValue, targetType);
}

mlir::Value MLIRBuilder::handleArrayLiteral(const mlir_edsl::ASTNode &node) {
  return memrefBuilder->buildArrayLiteral(node.array_literal());
}

mlir::Value MLIRBuilder::handleArrayAccess(const mlir_edsl::ASTNode &node) {
  return memrefBuilder->buildArrayAccess(node.array_access());
}

mlir::Value MLIRBuilder::handleArrayStore(const mlir_edsl::ASTNode &node) {
  return memrefBuilder->buildArrayStore(node.array_store());
}

mlir::Value MLIRBuilder::handleArrayBinaryOp(const mlir_edsl::ASTNode &node) {
  return memrefBuilder->buildArrayBinaryOp(node.array_binary_op());
}

std::string MLIRBuilder::getMLIRString() {
  std::string result;
  llvm::raw_string_ostream stream(result);
  mlir::OpPrintingFlags flags;
  flags.enableDebugInfo(false);
  flags.printGenericOpForm(false);
  module.print(stream, flags);
  return result;
}

std::string MLIRBuilder::getLLVMIRString() {
  MLIRLowering lowering(context.get());
  return lowering.lowerToLLVMIR(module);
}

void MLIRBuilder::reset() {
  // Reset builder state without destroying the module
  currentFunction = nullptr;
  parameterMap.clear();
  valueCache.clear();  // Clear SSA value cache between functions

  // Move insertion point back to module level
  builder->setInsertionPointToEnd(module.getBody());
}

bool MLIRBuilder::isIntegerType(mlir::Type type) const {
  return mlir::isa<mlir::IntegerType>(type);
}

bool MLIRBuilder::isFloatType(mlir::Type type) const {
  return mlir::isa<mlir::FloatType>(type);
}

// Explicit type promotion - Python provides the target type
std::pair<mlir::Value, mlir::Value>
MLIRBuilder::promoteToType(mlir::Value lhs, mlir::Value rhs,
                           mlir::Type targetType) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();

  // Promote left operand if needed
  if (lhsType != targetType && isIntegerType(lhsType) &&
      isFloatType(targetType)) {
    lhs = arithBuilder->convertIntToFloat(lhs);
  }

  // Promote right operand if needed
  if (rhsType != targetType && isIntegerType(rhsType) &&
      isFloatType(targetType)) {
    rhs = arithBuilder->convertIntToFloat(rhs);
  }

  return {lhs, rhs};
}

mlir::Type
MLIRBuilder::protoTypeToMLIRType(mlir_edsl::ValueType protoType) const {
  switch (protoType) {
  case mlir_edsl::ValueType::I32:
    return builder->getI32Type();
  case mlir_edsl::ValueType::F32:
    return builder->getF32Type();
  case mlir_edsl::ValueType::I1:
    return builder->getI1Type();
  default:
    throw std::runtime_error("Unknown protobuf type value: " +
                             std::to_string(static_cast<int>(protoType)));
  }
}

mlir::Type MLIRBuilder::arrayTypeSpecToMLIRType(
    const mlir_edsl::ArrayTypeSpec &arraySpec) const {

  // Extract element type
  mlir::Type elementType = protoTypeToMLIRType(arraySpec.element_type());

  // Extract shape dimensions
  llvm::SmallVector<int64_t> shape;
  for (int32_t dim : arraySpec.shape()) {
    shape.push_back(static_cast<int64_t>(dim));
  }

  // Create memref type: memref<2x3xi32>
  return mlir::MemRefType::get(shape, elementType);
}

// ==================== NEW ALGEBRAIC TYPE SYSTEM ====================

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

void MLIRBuilder::compileFunctionFromDef(const mlir_edsl::FunctionDef &func_def) {
  // Extract parameters using new TypeSpec-based system
  std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> params;
  for (const auto &param : func_def.params()) {
    params.push_back({param.name(), param.type()});
  }

  // Get return type using unified TypeSpec
  mlir::Type returnType = convertType(func_def.return_type());

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
      module.print(outFile);
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
  builder->setInsertionPointToEnd(module.getBody());

  compiledFunctions.clear();
  parameterMap.clear();
  functionTable.clear();
  currentFunction = nullptr;
}

std::vector<std::string> MLIRBuilder::listFunctions() const {
  std::vector<std::string> result;
  for (const auto &name : compiledFunctions) {
    result.push_back(name);
  }
  return result;
}

} // namespace mlir_edsl