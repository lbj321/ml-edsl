// cpp/include/mlir_edsl/SCFBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "ast.pb.h"
#include <functional>

namespace mlir_edsl {

// Forward declarations to avoid circular dependency
class MLIRBuilder;
class ArithBuilder;

/// Builder for scf dialect operations (control flow)
class SCFBuilder {
public:
  SCFBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context,
             MLIRBuilder* parent, ArithBuilder* arithBuilder);

  /// Build if-else operation with callbacks
  mlir::Value buildIf(mlir::Value condition,
                      std::function<mlir::Value()> buildThen,
                      std::function<mlir::Value()> buildElse,
                      mlir::Type resultType);

  /// Build for loop for iteration (no loop-carried values, for side effects)
  void buildForEach(mlir::Value start, mlir::Value end, mlir::Value step,
                    std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value iv)> body_fn);

  /// Build while loop with custom condition and body functions
  mlir::Value buildWhile(mlir::Value init,
                         std::function<mlir::Value(mlir::Value)> condition_fn,
                         std::function<mlir::Value(mlir::Value)> body_fn);

  /// Build while loop with binary operation and comparison (simplified interface)
  mlir::Value buildWhileWithOp(mlir::Value init, mlir::Value target,
                               mlir_edsl::BinaryOpType operation,
                               mlir_edsl::ComparisonPredicate condition);

private:
  mlir::OpBuilder& builder;
  mlir::MLIRContext* context;
  MLIRBuilder* parent;
  ArithBuilder* arithBuilder;  // For arithmetic operations in control flow
};

} // namespace mlir_edsl
