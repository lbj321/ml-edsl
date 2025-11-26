#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/ArithBuilder.h"
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
  memrefBuilder = std::make_unique<mlir_edsl::MemRefBuilder>(*builder, context.get(), this);
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
  // ==================== Handle Let Bindings (SSA Value Reuse) ====================
  if (node.has_let_binding()) {
    int64_t nodeId = node.let_binding().node_id();

    // Generate the MLIR value once
    mlir::Value result = buildFromProtobufNode(node.let_binding().value());

    // Cache it for future references
    valueCache[nodeId] = result;

    return result;
  }

  // ==================== Handle Value References (SSA Value Reuse) ====================
  if (node.has_value_ref()) {
    int64_t nodeId = node.value_ref().node_id();

    // Look up the previously cached value
    auto it = valueCache.find(nodeId);
    if (it != valueCache.end()) {
      return it->second;  // Return the cached SSA value - no new operation!
    }

    throw std::runtime_error("Reference to unbound value ID: " +
                             std::to_string(nodeId));
  }

  // ==================== Handle All Other Node Types ====================
  if (node.has_constant()) {
    const auto &constant = node.constant();
    switch (constant.value_type()) {
    case mlir_edsl::ValueType::I32:
      return arithBuilder->buildConstant(constant.int_value());
    case mlir_edsl::ValueType::F32:
      return arithBuilder->buildConstant(constant.float_value());
    default:
      throw std::runtime_error("Unsupported constant type");
    }

  } else if (node.has_parameter()) {
    const auto &param = node.parameter();
    return getParameter(param.name());

  } else if (node.has_binary_op()) {
    const auto &binop = node.binary_op();
    mlir::Value left = buildFromProtobufNode(binop.left());
    mlir::Value right = buildFromProtobufNode(binop.right());

    // Python already computed the result type - use it for promotion
    mlir::Type targetType = protoTypeToMLIRType(binop.result_type());
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

  } else if (node.has_compare_op()) {
    const auto &cmp = node.compare_op();
    mlir::Value left = buildFromProtobufNode(cmp.left());
    mlir::Value right = buildFromProtobufNode(cmp.right());

    // Python already computed the operand type - use it for promotion
    mlir::Type targetType = protoTypeToMLIRType(cmp.operand_type());
    auto [promotedLhs, promotedRhs] = promoteToType(left, right, targetType);

    return arithBuilder->buildCompare(cmp.predicate(), promotedLhs, promotedRhs);

  } else if (node.has_if_op()) {
    const auto &ifop = node.if_op();
    mlir::Value cond = buildFromProtobufNode(ifop.condition());
    mlir::Type resultType = protoTypeToMLIRType(ifop.result_type());

    // Create callbacks that capture the protobuf nodes
    auto buildThen = [this, &ifop]() {
      return buildFromProtobufNode(ifop.then_value());
    };
    auto buildElse = [this, &ifop]() {
      return buildFromProtobufNode(ifop.else_value());
    };

    return buildIf(cond, buildThen, buildElse, resultType);

  } else if (node.has_call_op()) {
    const auto &call = node.call_op();
    std::vector<mlir::Value> args;
    for (const auto &arg : call.args()) {
      args.push_back(buildFromProtobufNode(arg));
    }
    return callFunction(call.func_name(), args);

  } else if (node.has_for_loop_op()) {
    const auto &forloop = node.for_loop_op();
    mlir::Value start = buildFromProtobufNode(forloop.start());
    mlir::Value end = buildFromProtobufNode(forloop.end());
    mlir::Value step = buildFromProtobufNode(forloop.step());
    mlir::Value init_value = buildFromProtobufNode(forloop.init_value());
    return buildForWithOp(start, end, step, init_value, forloop.operation());

  } else if (node.has_while_loop_op()) {
    const auto &whileloop = node.while_loop_op();
    mlir::Value init_value = buildFromProtobufNode(whileloop.init_value());
    mlir::Value target = buildFromProtobufNode(whileloop.target());
    return buildWhileWithOp(init_value, target, whileloop.operation(),
                            whileloop.predicate());

  } else if (node.has_cast_op()) {
    const auto &cast = node.cast_op();
    mlir::Value sourceValue = buildFromProtobufNode(cast.value());
    return arithBuilder->buildCast(sourceValue, cast.target_type());

  } else if (node.has_array_literal()) {
    return memrefBuilder->buildArrayLiteral(node.array_literal());

  } else if (node.has_array_access()) {
    return memrefBuilder->buildArrayAccess(node.array_access());

  } else if (node.has_array_store()) {
    return memrefBuilder->buildArrayStore(node.array_store());
  }

  throw std::runtime_error("Unknown protobuf node type");
}

// ==================== Delegation wrappers to ArithBuilder ====================
mlir::Value MLIRBuilder::buildConstant(int32_t value) {
  return arithBuilder->buildConstant(value);
}

mlir::Value MLIRBuilder::buildConstant(float value) {
  return arithBuilder->buildConstant(value);
}

mlir::Value MLIRBuilder::buildAdd(mlir::Value lhs, mlir::Value rhs) {
  return arithBuilder->buildAdd(lhs, rhs);
}

mlir::Value MLIRBuilder::buildSub(mlir::Value lhs, mlir::Value rhs) {
  return arithBuilder->buildSub(lhs, rhs);
}

mlir::Value MLIRBuilder::buildMul(mlir::Value lhs, mlir::Value rhs) {
  return arithBuilder->buildMul(lhs, rhs);
}

mlir::Value MLIRBuilder::buildDiv(mlir::Value lhs, mlir::Value rhs) {
  return arithBuilder->buildDiv(lhs, rhs);
}

mlir::Value MLIRBuilder::convertIntToFloat(mlir::Value intValue) {
  return arithBuilder->convertIntToFloat(intValue);
}

mlir::Value MLIRBuilder::buildCast(mlir::Value sourceValue,
                                   mlir_edsl::ValueType targetType) {
  return arithBuilder->buildCast(sourceValue, targetType);
}

mlir::Value MLIRBuilder::buildCompare(mlir_edsl::ComparisonPredicate predicate,
                                      mlir::Value lhs, mlir::Value rhs) {
  return arithBuilder->buildCompare(predicate, lhs, rhs);
}

mlir::Value MLIRBuilder::buildIf(mlir::Value condition,
                                 std::function<mlir::Value()> buildThen,
                                 std::function<mlir::Value()> buildElse,
                                 mlir::Type resultType) {
  auto loc = builder->getUnknownLoc();

  auto ifOp = builder->create<mlir::scf::IfOp>(loc, resultType, condition,
                                               /*withElseRegion=*/true);

  // Build THEN region
  {
    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value thenVal = buildThen();  // Callback executes here
    builder->create<mlir::scf::YieldOp>(loc, thenVal);
  }

  // Build ELSE region
  {
    mlir::OpBuilder::InsertionGuard guard(*builder);
    builder->setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value elseVal = buildElse();  // Callback executes here
    builder->create<mlir::scf::YieldOp>(loc, elseVal);
  }

  builder->setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

mlir::Value MLIRBuilder::buildForWithOp(mlir::Value start, mlir::Value end,
                                        mlir::Value step,
                                        mlir::Value init_value,
                                        mlir_edsl::BinaryOpType operation) {

  auto body_fn = [this, operation](mlir::Value iv,
                                   mlir::Value iter_arg) -> mlir::Value {

    switch (operation) {
    case mlir_edsl::BinaryOpType::ADD:
      return buildAdd(iter_arg, iv);
    case mlir_edsl::BinaryOpType::MUL:
      return buildMul(iter_arg, iv);
    case mlir_edsl::BinaryOpType::SUB:
      return buildSub(iter_arg, iv);
    case mlir_edsl::BinaryOpType::DIV:
      return buildDiv(iter_arg, iv);
    default:
      throw std::runtime_error("Unsupported binary operation in for loop");
    }
  };

  return buildFor(start, end, step, init_value, body_fn);
}

mlir::Value MLIRBuilder::buildFor(
    mlir::Value start, mlir::Value end, mlir::Value step,
    mlir::Value init_value,
    std::function<mlir::Value(mlir::Value iv, mlir::Value iter_arg)> body_fn) {

  auto loc = builder->getUnknownLoc();
  auto forOp =
      builder->create<mlir::scf::ForOp>(loc, start, end, step, init_value);

  mlir::Block *body = forOp.getBody();
  builder->setInsertionPointToStart(body);

  mlir::Value inductionVar = body->getArgument(0);
  mlir::Value iterArg = body->getArgument(1);

  mlir::Value newIterArg = body_fn(inductionVar, iterArg);

  builder->create<mlir::scf::YieldOp>(loc, newIterArg);

  builder->setInsertionPointAfter(forOp);

  return forOp.getResult(0);
}

mlir::Value
MLIRBuilder::buildWhileWithOp(mlir::Value init, mlir::Value target,
                              mlir_edsl::BinaryOpType operation,
                              mlir_edsl::ComparisonPredicate condition) {

  auto condition_fn = [this, condition,
                       target](mlir::Value current) -> mlir::Value {
    return buildCompare(condition, current, target); // Now uses enum!
  };

  auto body_fn = [this, operation](mlir::Value current) -> mlir::Value {
    switch (operation) {
    case mlir_edsl::BinaryOpType::ADD:
      return buildAdd(current, buildConstant(1));
    case mlir_edsl::BinaryOpType::MUL:
      return buildMul(current, buildConstant(2));
    case mlir_edsl::BinaryOpType::SUB:
      return buildSub(current, buildConstant(1));
    case mlir_edsl::BinaryOpType::DIV:
      return buildDiv(current, buildConstant(2));
    default:
      throw std::runtime_error("Unsupported binary operation");
    }
  };

  return buildWhile(init, condition_fn, body_fn);
}

mlir::Value
MLIRBuilder::buildWhile(mlir::Value init,
                        std::function<mlir::Value(mlir::Value)> condition_fn,
                        std::function<mlir::Value(mlir::Value)> body_fn) {

  auto loc = builder->getUnknownLoc();
  auto resultType = init.getType();

  auto whileOp = builder->create<mlir::scf::WhileOp>(
      loc, mlir::TypeRange{resultType}, init);

  auto &beforeRegion = whileOp.getBefore();
  auto *beforeBlock = builder->createBlock(&beforeRegion, beforeRegion.end(),
                                           {resultType}, {loc});
  builder->setInsertionPointToStart(beforeBlock);

  auto current = beforeBlock->getArgument(0);
  auto condition = condition_fn(current);

  builder->create<mlir::scf::ConditionOp>(loc, condition, current);

  auto &afterRegion = whileOp.getAfter();
  auto *afterBlock = builder->createBlock(&afterRegion, afterRegion.end(),
                                          {resultType}, {loc});
  builder->setInsertionPointToStart(afterBlock);

  auto loopVar = afterBlock->getArgument(0);
  auto newValue = body_fn(loopVar);

  builder->create<mlir::scf::YieldOp>(loc, newValue);

  builder->setInsertionPointAfter(whileOp);
  return whileOp.getResult(0);
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

mlir::Type MLIRBuilder::getIntegerType() const { return builder->getI32Type(); }

mlir::Type MLIRBuilder::getFloatType() const { return builder->getF32Type(); }

mlir::Type MLIRBuilder::getBoolType() const { return builder->getI1Type(); }

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
    lhs = convertIntToFloat(lhs);
  }

  // Promote right operand if needed
  if (rhsType != targetType && isIntegerType(rhsType) &&
      isFloatType(targetType)) {
    rhs = convertIntToFloat(rhs);
  }

  return {lhs, rhs};
}

mlir::Type MLIRBuilder::getPromotedType(mlir::Type lhs, mlir::Type rhs) const {
  if (isFloatType(lhs) || isFloatType(rhs)) {
    return getFloatType();
  }
  return getIntegerType();
}

mlir::Type
MLIRBuilder::protoTypeToMLIRType(mlir_edsl::ValueType protoType) const {
  switch (protoType) {
  case mlir_edsl::ValueType::I32:
    return getIntegerType();
  case mlir_edsl::ValueType::F32:
    return getFloatType();
  case mlir_edsl::ValueType::I1:
    return getBoolType();
  default:
    throw std::runtime_error("Unknown protobuf type value: " +
                             std::to_string(static_cast<int>(protoType)));
  }
}

mlir::Value MLIRBuilder::getParameter(const std::string &name) {
  auto it = parameterMap.find(name);
  if (it != parameterMap.end()) {
    return it->second;
  }
  throw std::runtime_error("Parameter not found: " + name);
}

std::string MLIRBuilder::compileFunctionFromDef(const std::string &buffer) {
  mlir_edsl::FunctionDef func_def;

  if (!func_def.ParseFromString(buffer)) {
    throw std::runtime_error("Failed to parse FunctionDef protobuf");
  }

  // Extract parameters - already protobuf enums
  std::vector<std::pair<std::string, mlir_edsl::ValueType>> params;
  for (const auto &param : func_def.params()) {
    params.push_back({param.name(), param.type()});
  }

  // Everything uses enums - no string conversion!
  createFunction(func_def.name(), params, func_def.return_type());
  mlir::Value result = buildFromProtobufNode(func_def.body());
  finalizeFunction(func_def.name(), result);

  // Build FunctionSignature protobuf to return
  mlir_edsl::FunctionSignature sig;
  sig.set_name(func_def.name());
  sig.set_return_type(func_def.return_type());

  for (const auto &param : func_def.params()) {
    sig.add_param_types(param.type());
  }

  // Return serialized signature
  return sig.SerializeAsString();
}

void MLIRBuilder::createFunction(
    const std::string &name,
    const std::vector<std::pair<std::string, mlir_edsl::ValueType>> &params,
    mlir_edsl::ValueType return_type) {

  // Reset builder state from previous function
  reset();

  // Direct enum to MLIR type conversion using existing helper
  std::vector<mlir::Type> paramTypes;
  for (const auto &[paramName, valueType] : params) {
    paramTypes.push_back(protoTypeToMLIRType(valueType));
  }

  mlir::Type returnType = protoTypeToMLIRType(return_type);

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