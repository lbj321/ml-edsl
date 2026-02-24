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

mlir::ResultRange SCFBuilder::buildFor(
    mlir::Value start, mlir::Value end, mlir::Value step,
    mlir::ValueRange initValues,
    std::function<llvm::SmallVector<mlir::Value>(mlir::Value iv, mlir::ValueRange iterArgs)> body_fn) {

  auto loc = builder.getUnknownLoc();

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc, start, end, step, initValues,
      [&](mlir::OpBuilder &, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        llvm::SmallVector<mlir::Value> results = body_fn(iv, iterArgs);
        builder.create<mlir::scf::YieldOp>(loc, results);
      });

  builder.setInsertionPointAfter(forOp);
  return forOp.getResults();
}

} // namespace mlir_edsl
