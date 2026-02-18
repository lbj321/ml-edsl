// cpp/src/builders/SCFBuilder.cpp
#include "mlir_edsl/SCFBuilder.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir_edsl {

SCFBuilder::SCFBuilder(mlir::OpBuilder &builder)
    : builder(builder) {}

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
          mlir::ValueRange /*iterArgs*/) {
        // Execute user's body function
        body_fn(loopBuilder, loc, iv);

        // Yield with no values (since no iter_args)
        loopBuilder.create<mlir::scf::YieldOp>(loc);
      });

  // Restore insertion point after loop
  builder.setInsertionPointAfter(forOp);
}

mlir::Value SCFBuilder::buildForWithIterArgs(
    mlir::Value start, mlir::Value end, mlir::Value step,
    mlir::ValueRange initValues,
    std::function<mlir::Value(mlir::Value iv, mlir::Value iterArg)> body_fn) {

  auto loc = builder.getUnknownLoc();

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, start, end, step, initValues,
      [&](mlir::OpBuilder &loopBuilder, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        // Call body with induction var and single iter_arg
        mlir::Value result = body_fn(iv, iterArgs[0]);

        // Yield the new accumulator value
        loopBuilder.create<mlir::scf::YieldOp>(loc, result);
      });

  builder.setInsertionPointAfter(forOp);
  return forOp.getResult(0);
}

} // namespace mlir_edsl
