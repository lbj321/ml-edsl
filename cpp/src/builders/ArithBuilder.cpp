// cpp/src/builders/ArithBuilder.cpp
#include "mlir_edsl/ArithBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir_edsl {

ArithBuilder::ArithBuilder(mlir::OpBuilder& builder)
  : builder(builder) {}

mlir::Value ArithBuilder::buildConstant(int32_t value) {
  auto loc = builder.getUnknownLoc();
  auto type = builder.getI32Type();
  auto attr = builder.getI32IntegerAttr(value);

  return builder.create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value ArithBuilder::buildConstant(float value) {
  auto loc = builder.getUnknownLoc();
  auto type = builder.getF32Type();
  auto attr = builder.getF32FloatAttr(value);

  return builder.create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value ArithBuilder::buildConstant(bool value) {
  auto loc = builder.getUnknownLoc();
  auto type = builder.getI1Type();
  auto attr = builder.getBoolAttr(value);

  return builder.create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value ArithBuilder::buildIndexConstant(int64_t value) {
  auto loc = builder.getUnknownLoc();
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

// Template helper for binary operations - assumes operands already same type
template <typename IntOp, typename FloatOp>
mlir::Value ArithBuilder::buildBinaryOp(mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder.getUnknownLoc();

  if (mlir::isa<mlir::IntegerType>(lhs.getType())) {
    return builder.create<IntOp>(loc, lhs, rhs);
  }
  return builder.create<FloatOp>(loc, lhs, rhs);
}

// Public interface - uses template helper
mlir::Value ArithBuilder::buildAdd(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(lhs, rhs);
}

mlir::Value ArithBuilder::buildSub(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(lhs, rhs);
}

mlir::Value ArithBuilder::buildMul(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(lhs, rhs);
}

mlir::Value ArithBuilder::buildDiv(mlir::Value lhs, mlir::Value rhs) {
  return buildBinaryOp<mlir::arith::DivSIOp, mlir::arith::DivFOp>(lhs, rhs);
}

mlir::Value ArithBuilder::buildCast(mlir::Value sourceValue,
                                    mlir::Type targetType) {
  auto loc = builder.getUnknownLoc();
  mlir::Type sourceType = sourceValue.getType();

  if (sourceType == targetType) {
    return sourceValue;
  }

  // f32 -> i32
  if (mlir::isa<mlir::FloatType>(sourceType) && mlir::isa<mlir::IntegerType>(targetType)) {
    return builder.create<mlir::arith::FPToSIOp>(loc, targetType, sourceValue);
  }
  // i32 -> f32
  if (mlir::isa<mlir::IntegerType>(sourceType) && mlir::isa<mlir::FloatType>(targetType)) {
    return builder.create<mlir::arith::SIToFPOp>(loc, targetType, sourceValue);
  }

  throw std::runtime_error("Invalid cast: unsupported type combination");
}

mlir::Value ArithBuilder::buildCompare(mlir_edsl::ComparisonPredicate predicate,
                                       mlir::Value lhs, mlir::Value rhs) {
  auto loc = builder.getUnknownLoc();

  if (mlir::isa<mlir::IntegerType>(lhs.getType())) {
    auto pred = protobufToIntPredicate(predicate);
    return builder.create<mlir::arith::CmpIOp>(loc, pred, lhs, rhs);
  }

  auto pred = protobufToFloatPredicate(predicate);
  return builder.create<mlir::arith::CmpFOp>(loc, pred, lhs, rhs);
}

mlir::arith::CmpIPredicate
ArithBuilder::protobufToIntPredicate(mlir_edsl::ComparisonPredicate pred) const {
  switch (pred) {
  case mlir_edsl::ComparisonPredicate::SGT:
    return mlir::arith::CmpIPredicate::sgt;
  case mlir_edsl::ComparisonPredicate::SLT:
    return mlir::arith::CmpIPredicate::slt;
  case mlir_edsl::ComparisonPredicate::EQ:
    return mlir::arith::CmpIPredicate::eq;
  case mlir_edsl::ComparisonPredicate::NE:
    return mlir::arith::CmpIPredicate::ne;
  case mlir_edsl::ComparisonPredicate::SGE:
    return mlir::arith::CmpIPredicate::sge;
  case mlir_edsl::ComparisonPredicate::SLE:
    return mlir::arith::CmpIPredicate::sle;
  default:
    throw std::runtime_error("Invalid or unsupported integer comparison predicate");
  }
}

mlir::arith::CmpFPredicate
ArithBuilder::protobufToFloatPredicate(mlir_edsl::ComparisonPredicate pred) const {
  switch (pred) {
  case mlir_edsl::ComparisonPredicate::OGT:
    return mlir::arith::CmpFPredicate::OGT;
  case mlir_edsl::ComparisonPredicate::OLT:
    return mlir::arith::CmpFPredicate::OLT;
  case mlir_edsl::ComparisonPredicate::OEQ:
    return mlir::arith::CmpFPredicate::OEQ;
  case mlir_edsl::ComparisonPredicate::ONE:
    return mlir::arith::CmpFPredicate::ONE;
  case mlir_edsl::ComparisonPredicate::OGE:
    return mlir::arith::CmpFPredicate::OGE;
  case mlir_edsl::ComparisonPredicate::OLE:
    return mlir::arith::CmpFPredicate::OLE;
  default:
    throw std::runtime_error("Invalid or unsupported float comparison predicate");
  }
}

} // namespace mlir_edsl
