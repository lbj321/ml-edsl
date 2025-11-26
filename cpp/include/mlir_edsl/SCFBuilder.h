// cpp/include/mlir_edsl/SCFBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

  /// Build for loop with custom body function
  mlir::Value buildFor(mlir::Value start, mlir::Value end, mlir::Value step,
                       mlir::Value init_value,
                       std::function<mlir::Value(mlir::Value iv, mlir::Value iter_arg)> body_fn);

  /// Build for loop with binary operation (simplified interface)
  mlir::Value buildForWithOp(mlir::Value start, mlir::Value end, mlir::Value step,
                             mlir::Value init_value, mlir_edsl::BinaryOpType operation);

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
