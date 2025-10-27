#include "mlir_edsl/MLIRBuilder.h"

#include "ast.pb.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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

namespace mlir_edsl {

MLIRBuilder::MLIRBuilder() {
  context = std::make_unique<mlir::MLIRContext>();

  context->getOrLoadDialect<mlir::arith::ArithDialect>();
  context->getOrLoadDialect<mlir::func::FuncDialect>();
  context->getOrLoadDialect<mlir::scf::SCFDialect>();

  builder = std::make_unique<mlir::OpBuilder>(context.get());
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
  if (node.has_constant()) {
    const auto &constant = node.constant();
    switch (constant.value_type()) {
    case mlir_edsl::ValueType::I32:
      return buildConstant(constant.int_value());
    case mlir_edsl::ValueType::F32:
      return buildConstant(constant.float_value());
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
      return buildAdd(promotedLeft, promotedRight);
    case mlir_edsl::BinaryOpType::SUB:
      return buildSub(promotedLeft, promotedRight);
    case mlir_edsl::BinaryOpType::MUL:
      return buildMul(promotedLeft, promotedRight);
    case mlir_edsl::BinaryOpType::DIV:
      return buildDiv(promotedLeft, promotedRight);
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

    return buildCompare(cmp.predicate(), promotedLhs, promotedRhs);

  } else if (node.has_if_op()) {
    const auto &ifop = node.if_op();
    mlir::Value cond = buildFromProtobufNode(ifop.condition());
    mlir::Value thenVal = buildFromProtobufNode(ifop.then_value());
    mlir::Value elseVal = buildFromProtobufNode(ifop.else_value());
    return buildIf(cond, thenVal, elseVal);

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
  }

  throw std::runtime_error("Unknown protobuf node type");
}

mlir::arith::CmpIPredicate
MLIRBuilder::protobufToIntPredicate(mlir_edsl::ComparisonPredicate pred) const {
  switch (pred) {
  case mlir_edsl::ComparisonPredicate::SGT:
    return mlir::arith::CmpIPredicate::sgt;
  case mlir_edsl::ComparisonPredicate::SLT:
    return mlir::arith::CmpIPredicate::slt;
  case mlir_edsl::ComparisonPredicate::EQ:
    return mlir::arith::CmpIPredicate::eq;
  case mlir_edsl::ComparisonPredicate::NE:
    return mlir::arith::CmpIPredicate::ne;
  case mlir_edsl::ComparisonPredicate::SGE:
    return mlir::arith::CmpIPredicate::sge;
  case mlir_edsl::ComparisonPredicate::SLE:
    return mlir::arith::CmpIPredicate::sle;
  default:
    throw std::runtime_error(
        "Invalid or unsupported integer comparison predicate");
  }
}

mlir::arith::CmpFPredicate MLIRBuilder::protobufToFloatPredicate(
    mlir_edsl::ComparisonPredicate pred) const {
  switch (pred) {
  case mlir_edsl::ComparisonPredicate::OGT:
    return mlir::arith::CmpFPredicate::OGT;
  case mlir_edsl::ComparisonPredicate::OLT:
    return mlir::arith::CmpFPredicate::OLT;
  case mlir_edsl::ComparisonPredicate::OEQ:
    return mlir::arith::CmpFPredicate::OEQ;
  case mlir_edsl::ComparisonPredicate::ONE:
    return mlir::arith::CmpFPredicate::ONE;
  case mlir_edsl::ComparisonPredicate::OGE:
    return mlir::arith::CmpFPredicate::OGE;
  case mlir_edsl::ComparisonPredicate::OLE:
    return mlir::arith::CmpFPredicate::OLE;
  default:
    throw std::runtime_error(
        "Invalid or unsupported float comparison predicate");
  }
}

mlir::Value MLIRBuilder::buildConstant(int32_t value) {
  auto loc = builder->getUnknownLoc();
  auto type = getIntegerType();
  auto attr = builder->getI32IntegerAttr(value);

  return builder->create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value MLIRBuilder::buildConstant(float value) {
  auto loc = builder->getUnknownLoc();
  auto type = getFloatType();
  auto attr = builder->getF32FloatAttr(value);

  return builder->create<mlir::arith::ConstantOp>(loc, type, attr);
}

// Template helper for binary operations - assumes operands already same type
template <typename IntOp, typename FloatOp>
mlir::Value MLIRBuilder::buildBinaryOp(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();

  if (isIntegerType(lhs.getType())) {
    return builder->create<IntOp>(loc, lhs, rhs);
  }
  return builder->create<FloatOp>(loc, lhs, rhs);
}

// Public interface - uses template helper
mlir::Value MLIRBuilder::buildAdd(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(lhs, rhs);
}

mlir::Value MLIRBuilder::buildSub(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(lhs, rhs);
}

mlir::Value MLIRBuilder::buildMul(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(lhs, rhs);
}

mlir::Value MLIRBuilder::buildDiv(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::DivSIOp, mlir::arith::DivFOp>(lhs, rhs);
}

mlir::Value MLIRBuilder::convertIntToFloat(mlir::Value intValue) {
  auto loc = builder->getUnknownLoc();
  auto floatType = getFloatType();
  return builder->create<mlir::arith::SIToFPOp>(loc, floatType, intValue);
}

mlir::Value MLIRBuilder::buildCompare(mlir_edsl::ComparisonPredicate predicate,
                                      mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();

  if (isIntegerType(lhs.getType())) {
    auto pred = protobufToIntPredicate(predicate);
    return builder->create<mlir::arith::CmpIOp>(loc, pred, lhs, rhs);
  }

  auto pred = protobufToFloatPredicate(predicate);
  return builder->create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
}

mlir::Value MLIRBuilder::buildIf(mlir::Value condition, mlir::Value thenValue,
                                 mlir::Value elseValue) {
  auto loc = builder->getUnknownLoc();
  auto resultType = thenValue.getType();

  auto ifOp = builder->create<mlir::scf::IfOp>(loc, resultType, condition,
                                               /*withElseRegion*/ true);

  auto *thenBlock = &ifOp.getThenRegion().front();
  builder->setInsertionPointToStart(thenBlock);
  builder->create<mlir::scf::YieldOp>(loc, thenValue);

  auto *elseBlock = &ifOp.getElseRegion().front();
  builder->setInsertionPointToStart(elseBlock);
  builder->create<mlir::scf::YieldOp>(loc, elseValue);

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

void MLIRBuilder::compileFunctionFromDef(const std::string &buffer) {
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