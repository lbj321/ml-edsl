// cpp/src/builders/SCFBuilder.cpp
#include "mlir_edsl/SCFBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir_edsl {

SCFBuilder::SCFBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context,
                       MLIRBuilder* parent, ArithBuilder* arithBuilder)
  : builder(builder), context(context), parent(parent), arithBuilder(arithBuilder) {}

mlir::Value SCFBuilder::buildIf(mlir::Value condition,
                                std::function<mlir::Value()> buildThen,
                                std::function<mlir::Value()> buildElse,
                                mlir::Type resultType) {
  auto loc = builder.getUnknownLoc();

  auto ifOp = builder.create<mlir::scf::IfOp>(loc, resultType, condition,
                                              /*withElseRegion=*/true);

  // Build THEN region
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value thenVal = buildThen();  // Callback executes here
    builder.create<mlir::scf::YieldOp>(loc, thenVal);
  }

  // Build ELSE region
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value elseVal = buildElse();  // Callback executes here
    builder.create<mlir::scf::YieldOp>(loc, elseVal);
  }

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

mlir::Value SCFBuilder::buildForWithOp(mlir::Value start, mlir::Value end,
                                       mlir::Value step,
                                       mlir::Value init_value,
                                       mlir_edsl::BinaryOpType operation) {

  auto body_fn = [this, operation](mlir::Value iv,
                                   mlir::Value iter_arg) -> mlir::Value {

    // Call arithBuilder directly (temporary coupling - should be in AST)
    switch (operation) {
    case mlir_edsl::BinaryOpType::ADD:
      return arithBuilder->buildAdd(iter_arg, iv);
    case mlir_edsl::BinaryOpType::MUL:
      return arithBuilder->buildMul(iter_arg, iv);
    case mlir_edsl::BinaryOpType::SUB:
      return arithBuilder->buildSub(iter_arg, iv);
    case mlir_edsl::BinaryOpType::DIV:
      return arithBuilder->buildDiv(iter_arg, iv);
    default:
      throw std::runtime_error("Unsupported binary operation in for loop");
    }
  };

  return buildFor(start, end, step, init_value, body_fn);
}

mlir::Value SCFBuilder::buildFor(
    mlir::Value start, mlir::Value end, mlir::Value step,
    mlir::Value init_value,
    std::function<mlir::Value(mlir::Value iv, mlir::Value iter_arg)> body_fn) {

  auto loc = builder.getUnknownLoc();
  auto forOp =
      builder.create<mlir::scf::ForOp>(loc, start, end, step, init_value);

  mlir::Block *body = forOp.getBody();
  builder.setInsertionPointToStart(body);

  mlir::Value inductionVar = body->getArgument(0);
  mlir::Value iterArg = body->getArgument(1);

  mlir::Value newIterArg = body_fn(inductionVar, iterArg);

  builder.create<mlir::scf::YieldOp>(loc, newIterArg);

  builder.setInsertionPointAfter(forOp);

  return forOp.getResult(0);
}

mlir::Value
SCFBuilder::buildWhileWithOp(mlir::Value init, mlir::Value target,
                             mlir_edsl::BinaryOpType operation,
                             mlir_edsl::ComparisonPredicate condition) {

  auto condition_fn = [this, condition,
                       target](mlir::Value current) -> mlir::Value {
    return arithBuilder->buildCompare(condition, current, target);
  };

  auto body_fn = [this, operation](mlir::Value current) -> mlir::Value {
    // Call arithBuilder directly (temporary coupling - should be in AST)
    switch (operation) {
    case mlir_edsl::BinaryOpType::ADD:
      return arithBuilder->buildAdd(current, arithBuilder->buildConstant(1));
    case mlir_edsl::BinaryOpType::MUL:
      return arithBuilder->buildMul(current, arithBuilder->buildConstant(2));
    case mlir_edsl::BinaryOpType::SUB:
      return arithBuilder->buildSub(current, arithBuilder->buildConstant(1));
    case mlir_edsl::BinaryOpType::DIV:
      return arithBuilder->buildDiv(current, arithBuilder->buildConstant(2));
    default:
      throw std::runtime_error("Unsupported binary operation");
    }
  };

  return buildWhile(init, condition_fn, body_fn);
}

mlir::Value
SCFBuilder::buildWhile(mlir::Value init,
                       std::function<mlir::Value(mlir::Value)> condition_fn,
                       std::function<mlir::Value(mlir::Value)> body_fn) {

  auto loc = builder.getUnknownLoc();
  auto resultType = init.getType();

  auto whileOp = builder.create<mlir::scf::WhileOp>(
      loc, mlir::TypeRange{resultType}, init);

  auto &beforeRegion = whileOp.getBefore();
  auto *beforeBlock = builder.createBlock(&beforeRegion, beforeRegion.end(),
                                          {resultType}, {loc});
  builder.setInsertionPointToStart(beforeBlock);

  auto current = beforeBlock->getArgument(0);
  auto condition = condition_fn(current);

  builder.create<mlir::scf::ConditionOp>(loc, condition, current);

  auto &afterRegion = whileOp.getAfter();
  auto *afterBlock = builder.createBlock(&afterRegion, afterRegion.end(),
                                         {resultType}, {loc});
  builder.setInsertionPointToStart(afterBlock);

  auto loopVar = afterBlock->getArgument(0);
  auto newValue = body_fn(loopVar);

  builder.create<mlir::scf::YieldOp>(loc, newValue);

  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(0);
}

} // namespace mlir_edsl
