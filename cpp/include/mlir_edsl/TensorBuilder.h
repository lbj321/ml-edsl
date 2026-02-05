// cpp/include/mlir_edsl/TensorBuilder.h
#pragma once

#include "ast.pb.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir_edsl {

// Forward declarations to avoid circular dependency
class MLIRBuilder;

/// Builder for tensor dialect operations (value-semantic tensors)
class TensorBuilder {
public:
  TensorBuilder(mlir::OpBuilder &builder, mlir::MLIRContext *context,
                MLIRBuilder *parent);

  /// Build tensor from scalar elements: tensor.from_elements
  mlir::Value buildFromElements(const TensorFromElements &node);

  /// Build tensor extract: tensor.extract
  mlir::Value buildExtract(const TensorExtract &node);

private:
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  MLIRBuilder *parent;
};

} // namespace mlir_edsl
