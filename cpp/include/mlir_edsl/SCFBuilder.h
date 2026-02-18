// cpp/include/mlir_edsl/SCFBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <functional>

namespace mlir_edsl {

/// Builder for scf dialect operations (control flow)
class SCFBuilder {
public:
  SCFBuilder(mlir::OpBuilder& builder);

  /// Build if-else operation with callbacks
  mlir::Value buildIf(mlir::Value condition,
                      std::function<mlir::Value()> buildThen,
                      std::function<mlir::Value()> buildElse,
                      mlir::Type resultType);

  /// Build for loop for iteration (no loop-carried values, for side effects)
  void buildForEach(mlir::Value start, mlir::Value end, mlir::Value step,
                    std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value iv)> body_fn);

  /// Build for loop with iter_args (loop-carried values)
  mlir::Value buildForWithIterArgs(
      mlir::Value start, mlir::Value end, mlir::Value step,
      mlir::ValueRange initValues,
      std::function<mlir::Value(mlir::Value iv, mlir::Value iterArg)> body_fn);

private:
  mlir::OpBuilder& builder;
};

} // namespace mlir_edsl
