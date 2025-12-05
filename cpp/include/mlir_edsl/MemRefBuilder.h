// cpp/include/mlir_edsl/MemRefBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ast.pb.h"

namespace mlir_edsl {

// Forward declaration to avoid circular dependency
class MLIRBuilder;

/// Builder for memref dialect operations (arrays)
class MemRefBuilder {
public:
  MemRefBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context, MLIRBuilder* parent);

  /// Build memref type from protobuf ArrayTypeSpec
  mlir::MemRefType buildMemRefType(const ArrayTypeSpec& spec);

  /// Build array literal: allocate + initialize
  mlir::Value buildArrayLiteral(const ArrayLiteral& arrayLit);

  /// Build array access (memref.load)
  mlir::Value buildArrayAccess(const ArrayAccess& access);

  /// Build array store (memref.store)
  mlir::Value buildArrayStore(const ArrayStore& store);

  /// Build element-wise binary operation on arrays
  mlir::Value buildArrayBinaryOp(const ArrayBinaryOp& op);

private:
  mlir::OpBuilder& builder;
  mlir::MLIRContext* context;
  MLIRBuilder* parent;  // Back-reference to MLIRBuilder for dispatching
};

} // namespace mlir_edsl
