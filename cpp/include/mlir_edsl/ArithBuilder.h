// cpp/include/mlir_edsl/ArithBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "ast.pb.h"

namespace mlir_edsl {

// Forward declaration to avoid circular dependency
class MLIRBuilder;

/// Builder for arith dialect operations
class ArithBuilder {
public:
  ArithBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context, MLIRBuilder* parent);

  /// Build integer constant
  mlir::Value buildConstant(int32_t value);

  /// Build float constant
  mlir::Value buildConstant(float value);

  /// Arithmetic operations (operands must be same type)
  mlir::Value buildAdd(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildSub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildMul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildDiv(mlir::Value lhs, mlir::Value rhs);

  /// Comparison operation
  mlir::Value buildCompare(mlir_edsl::ComparisonPredicate predicate,
                           mlir::Value lhs, mlir::Value rhs);

  /// Type conversion operations
  mlir::Value convertIntToFloat(mlir::Value intValue);
  mlir::Value buildCast(mlir::Value sourceValue, mlir_edsl::ValueType targetType);

private:
  mlir::OpBuilder& builder;
  mlir::MLIRContext* context;
  MLIRBuilder* parent;

  /// Template helper for binary operations (assumes operands already same type)
  template <typename IntOp, typename FloatOp>
  mlir::Value buildBinaryOp(mlir::Value lhs, mlir::Value rhs);

  /// Convert protobuf predicates to MLIR predicates
  mlir::arith::CmpIPredicate protobufToIntPredicate(mlir_edsl::ComparisonPredicate pred) const;
  mlir::arith::CmpFPredicate protobufToFloatPredicate(mlir_edsl::ComparisonPredicate pred) const;
};

} // namespace mlir_edsl
