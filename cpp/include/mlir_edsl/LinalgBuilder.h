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

  /// Build dot product: linalg.dot ins(%a, %b) outs(%out) → returns scalar
  mlir::Value buildDot(const mlir_edsl::LinalgDot &node);

  /// Build matrix multiply: linalg.matmul ins(%A, %B) outs(%C) → returns memref
  mlir::Value buildMatmul(const mlir_edsl::LinalgMatmul &node,
                          mlir::Value outParam = {});

  /// Element-wise map via linalg.generic: returns output memref of same type as input.
  mlir::Value buildMap(const mlir_edsl::LinalgMap &node,
                       mlir::Value outParam = {});

  /// Reduction over a 1D memref: linalg.reduce ins(%input) outs(%acc) → returns scalar.
  mlir::Value buildReduce(const mlir_edsl::LinalgReduce &node);

private:
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  MLIRBuilder *parent;
};

} // namespace mlir_edsl
