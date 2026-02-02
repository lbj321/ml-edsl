// cpp/src/builders/MemRefBuilder.cpp
#include "mlir_edsl/MemRefBuilder.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/SCFBuilder.h"

namespace mlir_edsl {

MemRefBuilder::MemRefBuilder(mlir::OpBuilder &builder,
                             mlir::MLIRContext *context, MLIRBuilder *parent,
                             ArithBuilder *arithBuilder, SCFBuilder *scfBuilder)
    : builder(builder), context(context), parent(parent),
      arithBuilder(arithBuilder), scfBuilder(scfBuilder) {}

mlir::MemRefType MemRefBuilder::buildMemRefType(const MemRefTypeSpec &spec) {
  // Validate element type is scalar (memref-of-memref not supported)
  if (!spec.element_type().has_scalar()) {
    throw std::runtime_error(
        "Array element type must be scalar (i32, f32, or bool)");
  }

  mlir::Type elementType = parent->convertType(spec.element_type());

  // Build shape from protobuf repeated field
  llvm::SmallVector<int64_t, 3> shape;
  for (int i = 0; i < spec.shape_size(); ++i) {
    shape.push_back(spec.shape(i));
  }

  // Validation: only 1D, 2D, 3D supported
  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("Only 1D, 2D, and 3D arrays supported");
  }

  return mlir::MemRefType::get(shape, elementType);
}

mlir::Value MemRefBuilder::buildArrayLiteral(const ArrayLiteral &arrayLit) {
  auto loc = builder.getUnknownLoc();

  // 1. Build memref type from TypeSpec
  if (!arrayLit.type().has_memref()) {
    throw std::runtime_error("ArrayLiteral must have memref type");
  }
  mlir::MemRefType memrefType = buildMemRefType(arrayLit.type().memref());

  // 2. Allocate with memref.alloca (stack allocation)
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, memrefType);
  mlir::Value memref = allocOp.getResult();

  // 3. Get shape and initialize elements (flat iteration, row-major order)
  llvm::ArrayRef<int64_t> shape = memrefType.getShape();

  for (int flatIdx = 0; flatIdx < arrayLit.elements_size(); ++flatIdx) {
    mlir::Value element =
        parent->buildFromProtobufNode(arrayLit.elements(flatIdx));

    llvm::SmallVector<int64_t, 4> multiIdx = flatToMultiIndex(flatIdx, shape);
    llvm::SmallVector<mlir::Value, 4> indices;
    for (int64_t idx : multiIdx) {
      indices.push_back(parent->buildIndexConstant(idx));
    }

    builder.create<mlir::memref::StoreOp>(loc, element, memref, indices);
  }

  return memref;
}

mlir::Value MemRefBuilder::buildArrayAccess(const ArrayAccess &access) {
  auto loc = builder.getUnknownLoc();

  // Get memref (might be ValueReference to cached array)
  mlir::Value memref = parent->buildFromProtobufNode(access.array());

  // Build indices array from protobuf repeated field
  llvm::SmallVector<mlir::Value, 3> indices;
  for (int i = 0; i < access.indices_size(); ++i) {
    mlir::Value indexRaw = parent->buildFromProtobufNode(access.indices(i));
    mlir::Value index = parent->castToIndexType(indexRaw);
    indices.push_back(index);
  }

  // Load from memref with multi-dimensional indices
  auto loadOp = builder.create<mlir::memref::LoadOp>(loc, memref, indices);
  return loadOp.getResult();
}

mlir::Value MemRefBuilder::buildArrayStore(const ArrayStore &store) {
  auto loc = builder.getUnknownLoc();

  // Get memref and value
  mlir::Value memref = parent->buildFromProtobufNode(store.array());
  mlir::Value value = parent->buildFromProtobufNode(store.value());

  // Build indices array from protobuf repeated field
  llvm::SmallVector<mlir::Value, 3> indices;
  for (int i = 0; i < store.indices_size(); ++i) {
    mlir::Value indexRaw = parent->buildFromProtobufNode(store.indices(i));
    mlir::Value index = parent->castToIndexType(indexRaw);
    indices.push_back(index);
  }

  // Store to memref with multi-dimensional indices
  builder.create<mlir::memref::StoreOp>(loc, value, memref, indices);

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
  if (!op.result_type().has_memref()) {
    throw std::runtime_error("ArrayBinaryOp must have memref result type");
  }
  mlir::MemRefType resultType = buildMemRefType(op.result_type().memref());
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, resultType);
  mlir::Value resultArray = allocOp.getResult();

  // 4. Build element-wise nested loops over result shape
  llvm::ArrayRef<int64_t> shape = resultType.getShape();
  llvm::SmallVector<mlir::Value, 4> indices;

  auto bodyFn = [&](mlir::OpBuilder &loopBuilder, mlir::Location loc,
                    llvm::ArrayRef<mlir::Value> loopIndices) {
    buildArrayBinaryOpElement(loopBuilder, loc, loopIndices, left, right,
                              broadcastMode, opType, resultArray);
  };

  buildNestedForLoops(shape, /*dim=*/0, indices, bodyFn);

  return resultArray;
}

llvm::SmallVector<int64_t, 4>
MemRefBuilder::flatToMultiIndex(int64_t flatIndex,
                                llvm::ArrayRef<int64_t> shape) {
  int ndim = shape.size();
  llvm::SmallVector<int64_t, 4> indices(ndim);
  int64_t remaining = flatIndex;
  for (int64_t d = ndim - 1; d >= 0; --d) {
    indices[d] = remaining % shape[d];
    remaining /= shape[d];
  }
  return indices;
}

void MemRefBuilder::buildNestedForLoops(
    llvm::ArrayRef<int64_t> shape, int dim,
    llvm::SmallVectorImpl<mlir::Value> &indices,
    std::function<void(mlir::OpBuilder &, mlir::Location,
                       llvm::ArrayRef<mlir::Value>)>
        bodyFn) {
  mlir::Value c0 = parent->buildIndexConstant(0);
  mlir::Value cDimSize = parent->buildIndexConstant(shape[dim]);
  mlir::Value c1 = parent->buildIndexConstant(1);

  bool isInnermost = (dim == static_cast<int>(shape.size()) - 1);

  auto loopBody = [&](mlir::OpBuilder &loopBuilder, mlir::Location loc,
                      mlir::Value iv) {
    indices.push_back(iv);
    if (isInnermost) {
      bodyFn(loopBuilder, loc, indices);
    } else {
      buildNestedForLoops(shape, dim + 1, indices, bodyFn);
    }
    indices.pop_back();
  };

  scfBuilder->buildForEach(c0, cDimSize, c1, loopBody);
}

// Helper function: handles single iteration of array binary op
void MemRefBuilder::buildArrayBinaryOpElement(
    mlir::OpBuilder &loopBuilder, mlir::Location loc,
    llvm::ArrayRef<mlir::Value> indices, mlir::Value left, mlir::Value right,
    BroadcastMode broadcastMode, mlir_edsl::BinaryOpType opType,
    mlir::Value resultArray) {

  // Load elements based on broadcast mode
  mlir::Value leftElem, rightElem;

  switch (broadcastMode) {
  case mlir_edsl::NONE:
    leftElem = loopBuilder.create<mlir::memref::LoadOp>(loc, left, indices);
    rightElem = loopBuilder.create<mlir::memref::LoadOp>(loc, right, indices);
    break;

  case mlir_edsl::SCALAR_LEFT:
    leftElem = left;
    rightElem = loopBuilder.create<mlir::memref::LoadOp>(loc, right, indices);
    break;

  case mlir_edsl::SCALAR_RIGHT:
    leftElem = loopBuilder.create<mlir::memref::LoadOp>(loc, left, indices);
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
  loopBuilder.create<mlir::memref::StoreOp>(loc, result, resultArray, indices);
}

} // namespace mlir_edsl
