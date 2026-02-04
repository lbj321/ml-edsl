// cpp/include/mlir_edsl/MemRefBuilder.h
#pragma once

#include "ast.pb.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include <functional>

namespace mlir_edsl {

// Forward declarations to avoid circular dependency
class MLIRBuilder;
class ArithBuilder;
class SCFBuilder;

/// Builder for memref dialect operations (arrays)
class MemRefBuilder {
public:
  MemRefBuilder(mlir::OpBuilder &builder, mlir::MLIRContext *context,
                MLIRBuilder *parent, ArithBuilder *arithBuilder,
                SCFBuilder *scfBuilder);

  /// Build memref type from protobuf MemRefTypeSpec (new type system)
  mlir::MemRefType buildMemRefType(const MemRefTypeSpec &spec);

  /// Build array literal: allocate + initialize
  mlir::Value buildArrayLiteral(const ArrayLiteral &arrayLit);

  /// Build array access (memref.load)
  mlir::Value buildArrayAccess(const ArrayAccess &access);

  /// Build array store (memref.store)
  mlir::Value buildArrayStore(const ArrayStore &store);

  /// Build element-wise binary operation on arrays
  mlir::Value buildArrayBinaryOp(const ArrayBinaryOp &op);

private:
  mlir::OpBuilder &builder;
  mlir::MLIRContext *context;
  MLIRBuilder *parent;        // Back-reference to MLIRBuilder for dispatching
  ArithBuilder *arithBuilder; // For arithmetic operations
  SCFBuilder *scfBuilder;     // For loop operations

  /// Helper: Build loop body for array binary operations
  void buildArrayBinaryOpElement(mlir::OpBuilder &loopBuilder,
                                 mlir::Location loc,
                                 llvm::ArrayRef<mlir::Value> indices,
                                 mlir::Value left, mlir::Value right,
                                 BroadcastMode broadcastMode,
                                 mlir_edsl::BinaryOpType opType,
                                 mlir::Value resultArray);

  /// Convert flat index to multi-dimensional indices (row-major order)
  llvm::SmallVector<int64_t, 4> flatToMultiIndex(int64_t flatIndex,
                                                 llvm::ArrayRef<int64_t> shape);

  /// Build N nested scf.for loops over shape, pre-building loop bound
  /// constants at the current scope before entering any loops.
  void buildNestedForLoops(llvm::ArrayRef<int64_t> shape,
                           llvm::SmallVectorImpl<mlir::Value> &indices,
                           std::function<void(mlir::OpBuilder &, mlir::Location,
                                              llvm::ArrayRef<mlir::Value>)>
                               bodyFn);

  /// Recursive emitter: emits scf.for at dimension `dim` using pre-built
  /// loop bound constants, then recurses for inner dimensions.
  void emitNestedForLoops(llvm::ArrayRef<int64_t> shape, int dim,
                          llvm::SmallVectorImpl<mlir::Value> &indices,
                          std::function<void(mlir::OpBuilder &, mlir::Location,
                                             llvm::ArrayRef<mlir::Value>)>
                              bodyFn,
                          mlir::Value c0, mlir::Value c1,
                          llvm::ArrayRef<mlir::Value> dimSizes);
};

} // namespace mlir_edsl
