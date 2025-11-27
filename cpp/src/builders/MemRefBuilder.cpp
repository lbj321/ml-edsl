// cpp/src/builders/MemRefBuilder.cpp
#include "mlir_edsl/MemRefBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir_edsl {

MemRefBuilder::MemRefBuilder(mlir::OpBuilder& builder, mlir::MLIRContext* context, MLIRBuilder* parent)
  : builder(builder), context(context), parent(parent) {}

mlir::MemRefType MemRefBuilder::buildMemRefType(const ArrayTypeSpec& spec) {
  // Reuse parent's type converter - no duplication!
  mlir::Type elementType = parent->protoTypeToMLIRType(spec.element_type());

  int64_t size = spec.size();
  return mlir::MemRefType::get({size}, elementType);
}

mlir::Value MemRefBuilder::buildArrayLiteral(const ArrayLiteral& arrayLit) {
  auto loc = builder.getUnknownLoc();

  // 1. Build memref type
  mlir::MemRefType memrefType = buildMemRefType(arrayLit.array_type());

  // 2. Allocate with memref.alloca (stack allocation)
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, memrefType);
  mlir::Value memref = allocOp.getResult();

  // 3. Initialize elements with memref.store
  for (int i = 0; i < arrayLit.elements_size(); ++i) {
    // Dispatcher handles any node type + valueCache lookup
    mlir::Value element = parent->buildFromProtobufNode(arrayLit.elements(i));

    // Build index constant via parent utility
    mlir::Value index = parent->buildIndexConstant(i);

    // Store element at index
    builder.create<mlir::memref::StoreOp>(loc, element, memref, index);
  }

  return memref;
}

mlir::Value MemRefBuilder::buildArrayAccess(const ArrayAccess& access) {
  auto loc = builder.getUnknownLoc();

  // Get memref (might be ValueReference to cached array)
  mlir::Value memref = parent->buildFromProtobufNode(access.array());

  // Get index and convert to index type via parent utility
  mlir::Value indexRaw = parent->buildFromProtobufNode(access.index());
  mlir::Value index = parent->castToIndexType(indexRaw);

  // Load from memref
  auto loadOp = builder.create<mlir::memref::LoadOp>(loc, memref, index);
  return loadOp.getResult();
}

mlir::Value MemRefBuilder::buildArrayStore(const ArrayStore& store) {
  auto loc = builder.getUnknownLoc();

  // Get memref, index, value
  mlir::Value memref = parent->buildFromProtobufNode(store.array());
  mlir::Value indexRaw = parent->buildFromProtobufNode(store.index());
  mlir::Value index = parent->castToIndexType(indexRaw);
  mlir::Value value = parent->buildFromProtobufNode(store.value());

  // Store to memref
  builder.create<mlir::memref::StoreOp>(loc, value, memref, index);

  // Return the memref (for SSA chaining: arr = arr.at[i].set(v))
  return memref;
}

} // namespace mlir_edsl
