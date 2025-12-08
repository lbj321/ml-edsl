// cpp/include/mlir_edsl/MemRefBuilder.h
#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ast.pb.h"

namespace mlir_edsl {

// Forward declarations to avoid circular dependency
class MLIRBuilder;
class ArithBuilder;
class SCFBuilder;

/// Builder for memref dialect operations (arrays)
class MemRefBuilder {
public:
  MemRefBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context, MLIRBuilder* parent, ArithBuilder* arithBuilder, SCFBuilder* scfBuilder);

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
  MLIRBuilder* parent;           // Back-reference to MLIRBuilder for dispatching
  ArithBuilder* arithBuilder;    // For arithmetic operations
  SCFBuilder* scfBuilder;        // For loop operations

  /// Helper: Build loop body for array binary operations
  void buildArrayBinaryOpElement(
      mlir::OpBuilder& loopBuilder,
      mlir::Location loc,
      mlir::Value iv,
      mlir::Value left,
      mlir::Value right,
      BroadcastMode broadcastMode,
      mlir_edsl::BinaryOpType opType,
      mlir::Value resultArray);
};

} // namespace mlir_edsl
