// cpp/src/builders/SCFBuilder.cpp
#include "mlir_edsl/SCFBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"

namespace mlir_edsl {

SCFBuilder::SCFBuilder(mlir::OpBuilder &builder, mlir::MLIRContext *context,
                       MLIRBuilder *parent, ArithBuilder *arithBuilder)
    : builder(builder), context(context), parent(parent),
      arithBuilder(arithBuilder) {}

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
    mlir::Value thenVal = buildThen(); // Callback executes here
    builder.create<mlir::scf::YieldOp>(loc, thenVal);
  }

  // Build ELSE region
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value elseVal = buildElse(); // Callback executes here
    builder.create<mlir::scf::YieldOp>(loc, elseVal);
  }

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

void SCFBuilder::buildForEach(
    mlir::Value start, mlir::Value end, mlir::Value step,
    std::function<void(mlir::OpBuilder &, mlir::Location, mlir::Value iv)>
        body_fn) {

  auto loc = builder.getUnknownLoc();

  // Create ForOp with lambda body builder (avoids auto-generated yield)
  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, start, end, step, mlir::ValueRange{},
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        // Execute user's body function
        body_fn(loopBuilder, loc, iv);

        // Yield with no values (since no iter_args)
        loopBuilder.create<mlir::scf::YieldOp>(loc);
      });

  // Restore insertion point after loop
  builder.setInsertionPointAfter(forOp);
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
  auto *afterBlock =
      builder.createBlock(&afterRegion, afterRegion.end(), {resultType}, {loc});
  builder.setInsertionPointToStart(afterBlock);

  auto loopVar = afterBlock->getArgument(0);
  auto newValue = body_fn(loopVar);

  builder.create<mlir::scf::YieldOp>(loc, newValue);

  builder.setInsertionPointAfter(whileOp);
  return whileOp.getResult(0);
}

} // namespace mlir_edsl
