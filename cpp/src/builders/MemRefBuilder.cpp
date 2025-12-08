// cpp/src/builders/MemRefBuilder.cpp
#include "mlir_edsl/MemRefBuilder.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir_edsl/SCFBuilder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir_edsl/MLIRBuilder.h"

namespace mlir_edsl {

MemRefBuilder::MemRefBuilder(mlir::OpBuilder &builder,
                             mlir::MLIRContext *context, MLIRBuilder *parent,
                             ArithBuilder *arithBuilder,
                             SCFBuilder *scfBuilder)
    : builder(builder), context(context), parent(parent),
      arithBuilder(arithBuilder), scfBuilder(scfBuilder) {}

mlir::MemRefType MemRefBuilder::buildMemRefType(const ArrayTypeSpec &spec) {
  // Reuse parent's type converter - no duplication!
  mlir::Type elementType = parent->protoTypeToMLIRType(spec.element_type());

  int64_t size = spec.size();
  return mlir::MemRefType::get({size}, elementType);
}

mlir::Value MemRefBuilder::buildArrayLiteral(const ArrayLiteral &arrayLit) {
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

mlir::Value MemRefBuilder::buildArrayAccess(const ArrayAccess &access) {
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

mlir::Value MemRefBuilder::buildArrayStore(const ArrayStore &store) {
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

mlir::Value MemRefBuilder::buildArrayBinaryOp(const ArrayBinaryOp &op) {
  auto loc = builder.getUnknownLoc();

  // 1. Build operands
  mlir::Value left = parent->buildFromProtobufNode(op.left());
  mlir::Value right = parent->buildFromProtobufNode(op.right());

  // 2. Get broadcast mode and operation type
  auto broadcastMode = op.broadcast();
  auto opType = op.op_type();

  // 3. Allocate result memref
  int64_t arraySize = op.result_type().size();
  mlir::MemRefType resultType = buildMemRefType(op.result_type());
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, resultType);
  mlir::Value resultArray = allocOp.getResult();

  // 4. Build element-wise loop with clean lambda
  mlir::Value c0 = parent->buildIndexConstant(0);
  mlir::Value cSize = parent->buildIndexConstant(arraySize);
  mlir::Value c1 = parent->buildIndexConstant(1);

  auto loopBody = [&](mlir::OpBuilder& loopBuilder, mlir::Location loc, mlir::Value iv) {
    buildArrayBinaryOpElement(loopBuilder, loc, iv, left, right,
                              broadcastMode, opType, resultArray);
  };

  scfBuilder->buildForEach(c0, cSize, c1, loopBody);

  return resultArray;
}

// Helper function: handles single iteration of array binary op
void MemRefBuilder::buildArrayBinaryOpElement(
    mlir::OpBuilder& loopBuilder,
    mlir::Location loc,
    mlir::Value iv,
    mlir::Value left,
    mlir::Value right,
    BroadcastMode broadcastMode,
    mlir_edsl::BinaryOpType opType,
    mlir::Value resultArray) {

  // Load elements based on broadcast mode
  mlir::Value leftElem, rightElem;

  switch (broadcastMode) {
  case mlir_edsl::NONE:
    leftElem = loopBuilder.create<mlir::memref::LoadOp>(loc, left, iv);
    rightElem = loopBuilder.create<mlir::memref::LoadOp>(loc, right, iv);
    break;

  case mlir_edsl::SCALAR_LEFT:
    leftElem = left;
    rightElem = loopBuilder.create<mlir::memref::LoadOp>(loc, right, iv);
    break;

  case mlir_edsl::SCALAR_RIGHT:
    leftElem = loopBuilder.create<mlir::memref::LoadOp>(loc, left, iv);
    rightElem = right;
    break;

  default:
    throw std::runtime_error("Unknown broadcast mode");
  }

  // Perform element-wise operation using ArithBuilder
  mlir::Value result;
  switch (opType) {
  case mlir_edsl::BinaryOpType::ADD:
    result = arithBuilder->buildAdd(leftElem, rightElem);
    break;

  case mlir_edsl::BinaryOpType::SUB:
    result = arithBuilder->buildSub(leftElem, rightElem);
    break;

  case mlir_edsl::BinaryOpType::MUL:
    result = arithBuilder->buildMul(leftElem, rightElem);
    break;

  case mlir_edsl::BinaryOpType::DIV:
    result = arithBuilder->buildDiv(leftElem, rightElem);
    break;

  default:
    throw std::runtime_error("Unknown binary operation");
  }

  // Store result
  loopBuilder.create<mlir::memref::StoreOp>(loc, result, resultArray, iv);
}

} // namespace mlir_edsl
