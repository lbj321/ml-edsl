#include "mlir_edsl/MLIRBuilder.h"

#include <iostream>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir_edsl/MLIRLowering.h"
#include "llvm/Support/raw_ostream.h"

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

  auto funcType = builder->getFunctionType({}, {});
  currentFunction = builder->create<mlir::func::FuncOp>(
      builder->getUnknownLoc(), "temp_function", funcType);

  auto *entryBlock = currentFunction.addEntryBlock();
  builder->setInsertionPointToStart(entryBlock);
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

mlir::Value MLIRBuilder::buildAdd(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();
  auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

  if (isIntegerType(promotedLhs.getType())) {
    return builder->create<mlir::arith::AddIOp>(loc, promotedLhs, promotedRhs);
  }

  return builder->create<mlir::arith::AddFOp>(loc, promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::buildSub(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();
  auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

  if (isIntegerType(promotedLhs.getType())) {
    return builder->create<mlir::arith::SubIOp>(loc, promotedLhs, promotedRhs);
  }

  return builder->create<mlir::arith::SubFOp>(loc, promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::buildMul(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();
  auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

  if (isIntegerType(promotedLhs.getType())) {
    return builder->create<mlir::arith::MulIOp>(loc, promotedLhs, promotedRhs);
  }

  return builder->create<mlir::arith::MulFOp>(loc, promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::buildDiv(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();
  auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

  if (isIntegerType(promotedLhs.getType())) {
    return builder->create<mlir::arith::DivSIOp>(loc, promotedLhs, promotedRhs);
  }

  return builder->create<mlir::arith::DivFOp>(loc, promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::convertIntToFloat(mlir::Value intValue) {
  auto loc = builder->getUnknownLoc();
  auto floatType = getFloatType();
  return builder->create<mlir::arith::SIToFPOp>(loc, floatType, intValue);
}

mlir::arith::CmpIPredicate
MLIRBuilder::getIntegerPredicate(const std::string &pred) const {
  // Map string to integer predicate enum
  if (pred == "sgt") {
    return mlir::arith::CmpIPredicate::sgt; // signed greater than
  } else if (pred == "slt") {
    return mlir::arith::CmpIPredicate::slt; // signed less than
  } else if (pred == "eq") {
    return mlir::arith::CmpIPredicate::eq; // equal
  } else if (pred == "ne") {
    return mlir::arith::CmpIPredicate::ne; // not equal
  } else if (pred == "sge") {
    return mlir::arith::CmpIPredicate::sge; // signed greater or equal
  } else if (pred == "sle") {
    return mlir::arith::CmpIPredicate::sle; // signed less or equal
  } else {
    throw std::runtime_error("Unsupported integer predicate: " + pred);
  }
}

mlir::arith::CmpFPredicate
MLIRBuilder::getFloatPredicate(const std::string &pred) const {
  // Map string to float predicate enum
  if (pred == "ogt") {
    return mlir::arith::CmpFPredicate::OGT; // ordered greater than
  } else if (pred == "olt") {
    return mlir::arith::CmpFPredicate::OLT; // ordered less than
  } else if (pred == "oeq") {
    return mlir::arith::CmpFPredicate::OEQ; // ordered equal
  } else if (pred == "one") {
    return mlir::arith::CmpFPredicate::ONE; // ordered not equal
  } else if (pred == "oge") {
    return mlir::arith::CmpFPredicate::OGE; // ordered greater or equal
  } else if (pred == "ole") {
    return mlir::arith::CmpFPredicate::OLE; // ordered less or equal
  } else {
    throw std::runtime_error("Unsupported float predicate: " + pred);
  }
}

mlir::Value MLIRBuilder::buildCompare(const std::string &predicate,
                                      mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder->getUnknownLoc();
  auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

  if (isIntegerType(promotedLhs.getType())) {
    auto pred = getIntegerPredicate(predicate);
    return builder->create<mlir::arith::CmpIOp>(loc, pred, promotedLhs,
                                                promotedRhs);
  }

  auto pred = getFloatPredicate(predicate);
  return builder->create<mlir::arith::CmpFOp>(loc, pred, promotedLhs,
                                              promotedRhs);
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
                                        const std::string &operation) {

  auto body_fn = [this, operation](mlir::Value iv,
                                   mlir::Value iter_arg) -> mlir::Value {
    if (operation == "add") {
      return buildAdd(iter_arg, iv); // iter_arg + iv
    } else if (operation == "mul") {
      return buildMul(iter_arg, iv); // iter_arg * iv
    } else if (operation == "sub") {
      return buildSub(iter_arg, iv); // iter_arg - iv
    } else if (operation == "div") {
      return buildDiv(iter_arg, iv); // iter_arg / iv
    }
    throw std::runtime_error("Unsupported operation: " + operation);
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

mlir::Value MLIRBuilder::buildWhileWithOp(mlir::Value init, mlir::Value target,
                                          const std::string &operation,
                                          const std::string &condition) {

  auto condition_fn = [this, condition,
                       target](mlir::Value current) -> mlir::Value {
    return buildCompare(condition, current, target);
  };

  auto body_fn = [this, operation](mlir::Value current) -> mlir::Value {
    if (operation == "add") {
      return buildAdd(current, buildConstant(1)); // current + 1
    } else if (operation == "mul") {
      return buildMul(current, buildConstant(2)); // current * 2
    } else if (operation == "sub") {
      return buildSub(current, buildConstant(1)); // current - 1
    } else if (operation == "div") {
      return buildDiv(current, buildConstant(2)); // current / 2
    }
    throw std::runtime_error("Unsupported operation: " + operation);
  };

  return buildWhile(init, condition_fn, body_fn);
};

mlir::Value
MLIRBuilder::buildWhile(mlir::Value init,
                        std::function<mlir::Value(mlir::Value)> condition_fn,
                        std::function<mlir::Value(mlir::Value)> body_fn) {

  auto loc = builder->getUnknownLoc();
  auto resultType = init.getType();

  auto whileOp = builder->create<mlir::scf::WhileOp>(loc, mlir::TypeRange{resultType}, init);

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

void MLIRBuilder::reset() { initializeModule(); }

mlir::Type MLIRBuilder::getIntegerType() const { return builder->getI32Type(); }

mlir::Type MLIRBuilder::getFloatType() const { return builder->getF32Type(); }

mlir::Type MLIRBuilder::getBoolType() const { return builder->getI1Type(); }

bool MLIRBuilder::isIntegerType(mlir::Type type) const {
  return mlir::isa<mlir::IntegerType>(type);
}

bool MLIRBuilder::isFloatType(mlir::Type type) const {
  return mlir::isa<mlir::FloatType>(type);
}

std::pair<mlir::Value, mlir::Value> MLIRBuilder::promoteTypes(mlir::Value lhs,
                                                              mlir::Value rhs) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();
  mlir::Type promotedType = getPromotedType(lhsType, rhsType);

  if (lhsType != promotedType && isIntegerType(lhsType)) {
    lhs = convertIntToFloat(lhs);
  }

  if (rhsType != promotedType && isIntegerType(rhsType)) {
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

mlir::Value MLIRBuilder::getParameter(const std::string &name) {
  auto it = parameterMap.find(name);
  if (it != parameterMap.end()) {
    return it->second;
  }
  throw std::runtime_error("Parameter not found: " + name);
}

void MLIRBuilder::createFunctionWithParamsSetup(
    const std::vector<std::pair<std::string, std::string>> &params) {
  if (!currentFunction) {
    std::cerr << "Error: No current function!\n";
    return;
  }

  std::cerr << "MLIRBuilder: Setting up function with " << params.size()
            << " parameters\n";

  parameterMap.clear();

  std::vector<mlir::Type> paramTypes;
  for (const auto &[paramName, typeStr] : params) {
    mlir::Type paramType;
    if (typeStr == "i32") {
      paramType = builder->getI32Type();
    } else if (typeStr == "f32") {
      paramType = builder->getF32Type();
    } else {
      throw std::runtime_error("Unsupported parameter type: " + typeStr);
    }
    paramTypes.push_back(paramType);
  }

  // Set up parameters for the function
  auto &entryBlock = currentFunction.front();

  // Only clear and rebuild the block if we have parameters
  if (!params.empty()) {
    entryBlock.clear();

    for (size_t i = 0; i < params.size(); i++) {
      entryBlock.addArgument(paramTypes[i], builder->getUnknownLoc());
    }

    for (size_t i = 0; i < params.size(); i++) {
      parameterMap[params[i].first] = entryBlock.getArgument(i);
    }

    builder->setInsertionPointToStart(&entryBlock);
  }
  // For parameterless functions, leave existing operations intact
}

void MLIRBuilder::finalizeFunctionWithParams(const std::string &name,
                                             mlir::Value result) {
  std::cerr << "MLIRBuilder: Finalizing function " << name << "\n";

  // Get argument types from the entry block
  auto &entryBlock = currentFunction.front();
  std::vector<mlir::Type> argTypes;
  for (auto arg : entryBlock.getArguments()) {
    argTypes.push_back(arg.getType());
  }

  // Set function name and type with correct argument types
  currentFunction.setName(name);
  currentFunction.setFunctionType(
      builder->getFunctionType(argTypes, {result.getType()}));

  builder->setInsertionPointToEnd(&entryBlock);
  builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc(), result);
}

} // namespace mlir_edsl