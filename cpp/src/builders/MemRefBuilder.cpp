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

  // 1. Build memref type
  mlir::MemRefType memrefType = buildMemRefType(arrayLit.array_type());

  // 2. Allocate with memref.alloca (stack allocation)
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, memrefType);
  mlir::Value memref = allocOp.getResult();

  // 3. Get shape dimensions
  llvm::ArrayRef<int64_t> shape = memrefType.getShape();
  int ndim = shape.size();

  // 4. Initialize elements with memref.store
  if (ndim == 1) {
    // 1D: Single index per store
    for (int i = 0; i < arrayLit.elements_size(); ++i) {
      mlir::Value element = parent->buildFromProtobufNode(arrayLit.elements(i));
      mlir::Value index = parent->buildIndexConstant(i);
      builder.create<mlir::memref::StoreOp>(loc, element, memref, index);
    }
  } else if (ndim == 2) {
    // 2D: Nested loop initialization (row-major order)
    int rows = shape[0];
    int cols = shape[1];
    int flatIdx = 0;

    for (int i = 0; i < rows; ++i) {
      mlir::Value rowIdx = parent->buildIndexConstant(i);
      for (int j = 0; j < cols; ++j) {
        mlir::Value colIdx = parent->buildIndexConstant(j);
        mlir::Value element = parent->buildFromProtobufNode(arrayLit.elements(flatIdx++));

        llvm::SmallVector<mlir::Value, 2> indices = {rowIdx, colIdx};
        builder.create<mlir::memref::StoreOp>(loc, element, memref, indices);
      }
    }
  } else if (ndim == 3) {
    // 3D: Triple-nested loop initialization (row-major order)
    int dim0 = shape[0];
    int dim1 = shape[1];
    int dim2 = shape[2];
    int flatIdx = 0;

    for (int i = 0; i < dim0; ++i) {
      mlir::Value idx0 = parent->buildIndexConstant(i);
      for (int j = 0; j < dim1; ++j) {
        mlir::Value idx1 = parent->buildIndexConstant(j);
        for (int k = 0; k < dim2; ++k) {
          mlir::Value idx2 = parent->buildIndexConstant(k);
          mlir::Value element = parent->buildFromProtobufNode(arrayLit.elements(flatIdx++));

          llvm::SmallVector<mlir::Value, 3> indices = {idx0, idx1, idx2};
          builder.create<mlir::memref::StoreOp>(loc, element, memref, indices);
        }
      }
    }
  } else {
    throw std::runtime_error("Only 1D, 2D, and 3D arrays supported");
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
  mlir::MemRefType resultType = buildMemRefType(op.result_type());
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, resultType);
  mlir::Value resultArray = allocOp.getResult();

  // 4. Get shape dimensions
  llvm::ArrayRef<int64_t> shape = resultType.getShape();
  int ndim = shape.size();

  // 5. Build element-wise loops based on dimensionality
  if (ndim == 1) {
    // 1D: Single loop
    mlir::Value c0 = parent->buildIndexConstant(0);
    mlir::Value cSize = parent->buildIndexConstant(shape[0]);
    mlir::Value c1 = parent->buildIndexConstant(1);

    auto loopBody = [&](mlir::OpBuilder& loopBuilder, mlir::Location loc, mlir::Value i) {
      llvm::SmallVector<mlir::Value, 1> indices = {i};
      buildArrayBinaryOpElement(loopBuilder, loc, indices, left, right,
                                broadcastMode, opType, resultArray);
    };

    scfBuilder->buildForEach(c0, cSize, c1, loopBody);

  } else if (ndim == 2) {
    // 2D: Nested loops
    mlir::Value c0 = parent->buildIndexConstant(0);
    mlir::Value c1 = parent->buildIndexConstant(1);
    mlir::Value rows = parent->buildIndexConstant(shape[0]);
    mlir::Value cols = parent->buildIndexConstant(shape[1]);

    auto outerLoop = [&](mlir::OpBuilder& outerBuilder, mlir::Location loc, mlir::Value i) {
      auto innerLoop = [&](mlir::OpBuilder& innerBuilder, mlir::Location loc, mlir::Value j) {
        llvm::SmallVector<mlir::Value, 2> indices = {i, j};
        buildArrayBinaryOpElement(innerBuilder, loc, indices, left, right,
                                  broadcastMode, opType, resultArray);
      };

      scfBuilder->buildForEach(c0, cols, c1, innerLoop);
    };

    scfBuilder->buildForEach(c0, rows, c1, outerLoop);

  } else if (ndim == 3) {
    // 3D: Triple-nested loops
    mlir::Value c0 = parent->buildIndexConstant(0);
    mlir::Value c1 = parent->buildIndexConstant(1);
    mlir::Value dim0 = parent->buildIndexConstant(shape[0]);
    mlir::Value dim1 = parent->buildIndexConstant(shape[1]);
    mlir::Value dim2 = parent->buildIndexConstant(shape[2]);

    auto outerLoop = [&](mlir::OpBuilder& outerBuilder, mlir::Location loc, mlir::Value i) {
      auto middleLoop = [&](mlir::OpBuilder& middleBuilder, mlir::Location loc, mlir::Value j) {
        auto innerLoop = [&](mlir::OpBuilder& innerBuilder, mlir::Location loc, mlir::Value k) {
          llvm::SmallVector<mlir::Value, 3> indices = {i, j, k};
          buildArrayBinaryOpElement(innerBuilder, loc, indices, left, right,
                                    broadcastMode, opType, resultArray);
        };

        scfBuilder->buildForEach(c0, dim2, c1, innerLoop);
      };

      scfBuilder->buildForEach(c0, dim1, c1, middleLoop);
    };

    scfBuilder->buildForEach(c0, dim0, c1, outerLoop);

  } else {
    throw std::runtime_error("Only 1D, 2D, and 3D arrays supported");
  }

  return resultArray;
}

// Helper function: handles single iteration of array binary op
void MemRefBuilder::buildArrayBinaryOpElement(
    mlir::OpBuilder& loopBuilder,
    mlir::Location loc,
    llvm::ArrayRef<mlir::Value> indices,
    mlir::Value left,
    mlir::Value right,
    BroadcastMode broadcastMode,
    mlir_edsl::BinaryOpType opType,
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
