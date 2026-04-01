// cpp/include/mlir_edsl/LinalgBuilder.h
#pragma once

#include "ast.pb.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir_edsl {

// Forward declarations to avoid circular dependency
class MLIRBuilder;

/// Builder for linalg dialect operations (structured linear algebra)
class LinalgBuilder {
public:
  LinalgBuilder(mlir::OpBuilder &builder, mlir::MLIRContext *context,
                MLIRBuilder *parent);

  /// Build dot product: linalg.dot ins(%a, %b) outs(%acc) → returns scalar
  mlir::Value buildDot(const mlir_edsl::LinalgDot &node);

  /// Build matrix multiply: linalg.matmul ins(%A, %B) outs(%C) → returns tensor
  mlir::Value buildMatmul(const mlir_edsl::LinalgMatmul &node,
                          mlir::Value outParam = {});

  /// Element-wise map via linalg.map: returns output tensor of same type as input.
  mlir::Value buildMap(const mlir_edsl::LinalgMap &node,
                       mlir::Value outParam = {});

  /// Reduction over a 1D tensor: linalg.reduce ins(%input) outs(%acc) → returns scalar.
  mlir::Value buildReduce(const mlir_edsl::LinalgReduce &node);

  /// Element-wise binary op on tensors with optional broadcasting.
  /// Handles NONE (same-shape), SCALAR_*, and TENSOR_BIAS_* broadcast modes.
  mlir::Value buildBinaryOp(const mlir_edsl::LinalgBinaryOp &node,
                             mlir::Value outParam = {});

  /// Known activation function as linalg.generic with arith ops.
  /// RELU: arith.maximumf(x, 0). LEAKY_RELU: arith.select(x > 0, x, alpha*x).
  mlir::Value buildActivation(const mlir_edsl::LinalgActivation &node,
                              mlir::Value outParam = {});

private:
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  MLIRBuilder *parent;
};

} // namespace mlir_edsl
