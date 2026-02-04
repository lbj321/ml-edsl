// cpp/include/mlir_edsl/ArithBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "ast.pb.h"

namespace mlir_edsl {

/// Builder for arith dialect operations
class ArithBuilder {
public:
  ArithBuilder(mlir::OpBuilder& builder);

  /// Build integer constant
  mlir::Value buildConstant(int32_t value);

  /// Build float constant
  mlir::Value buildConstant(float value);

  /// Build boolean constant (i1)
  mlir::Value buildConstant(bool value);

  /// Build index constant (for array indexing)
  mlir::Value buildIndexConstant(int64_t value);

  /// Arithmetic operations (operands must be same type)
  mlir::Value buildAdd(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildSub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildMul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildDiv(mlir::Value lhs, mlir::Value rhs);

  /// Comparison operation
  mlir::Value buildCompare(mlir_edsl::ComparisonPredicate predicate,
                           mlir::Value lhs, mlir::Value rhs);

  /// Cast between i32 and f32
  mlir::Value buildCast(mlir::Value sourceValue, mlir::Type targetType);

private:
  mlir::OpBuilder& builder;

  /// Template helper for binary operations (assumes operands already same type)
  template <typename IntOp, typename FloatOp>
  mlir::Value buildBinaryOp(mlir::Value lhs, mlir::Value rhs);

  /// Convert protobuf predicates to MLIR predicates
  mlir::arith::CmpIPredicate protobufToIntPredicate(mlir_edsl::ComparisonPredicate pred) const;
  mlir::arith::CmpFPredicate protobufToFloatPredicate(mlir_edsl::ComparisonPredicate pred) const;
};

} // namespace mlir_edsl
