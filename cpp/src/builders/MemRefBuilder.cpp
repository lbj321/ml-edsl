// cpp/src/builders/MemRefBuilder.cpp
#include "mlir_edsl/MemRefBuilder.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir_edsl/MLIRBuilder.h"

namespace mlir_edsl {

MemRefBuilder::MemRefBuilder(mlir::OpBuilder &builder,
                             mlir::MLIRContext *context, MLIRBuilder *parent)
    : builder(builder), context(context), parent(parent) {}

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

  // 1. Build operands (recursively - handles ValueReferences, etc.)
  mlir::Value left = parent->buildFromProtobufNode(op.left());
  mlir::Value right = parent->buildFromProtobufNode(op.right());

  // 2. Determine broadcast mode
  auto broadcastMode = op.broadcast();

  // 3. Get result array shape and allocate result memref
  int64_t arraySize = op.result_type().size();
  mlir::MemRefType resultType = buildMemRefType(op.result_type());
  auto allocOp = builder.create<mlir::memref::AllocaOp>(loc, resultType);
  mlir::Value resultArray = allocOp.getResult();

  // 4. Generate element-wise loop: for(i = 0; i < size; i++)
  mlir::Value c0 = parent->buildIndexConstant(0);
  mlir::Value cSize = parent->buildIndexConstant(arraySize);
  mlir::Value c1 = parent->buildIndexConstant(1);

  auto forOp = builder.create<mlir::scf::ForOp>(
      loc,
      c0,                 // lower bound
      cSize,              // upper bound
      c1,                 // step
      mlir::ValueRange{}, // no loop-carried values
      [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iv,
          mlir::ValueRange iterArgs) {
        // Build loop body inside this lambda
        mlir::Value leftElem, rightElem;
        switch (broadcastMode) {
        case mlir_edsl::NONE: {
          leftElem = builder.create<mlir::memref::LoadOp>(loc, left, iv);
          rightElem = builder.create<mlir::memref::LoadOp>(loc, right, iv);
          break;
        }
        case mlir_edsl::SCALAR_LEFT: {
          leftElem = left;
          rightElem = builder.create<mlir::memref::LoadOp>(loc, right, iv);
          break;
        }
        case mlir_edsl::SCALAR_RIGHT: {
          leftElem = builder.create<mlir::memref::LoadOp>(loc, left, iv);
          rightElem = right;
          break;
        }
        default:
          throw std::runtime_error("Unknown broadcast mode");
        }

        mlir::Value result;
        mlir::Type elementType = resultType.getElementType();
        bool isFloat = elementType.isF32();

        switch (op.op_type()) {
        case mlir_edsl::BinaryOpType::ADD:
          result =
              isFloat
                  ? builder
                        .create<mlir::arith::AddFOp>(loc, leftElem, rightElem)
                        .getResult()
                  : builder
                        .create<mlir::arith::AddIOp>(loc, leftElem, rightElem)
                        .getResult();
          break;
        case mlir_edsl::BinaryOpType::SUB:
          result =
              isFloat
                  ? builder
                        .create<mlir::arith::SubFOp>(loc, leftElem, rightElem)
                        .getResult()
                  : builder
                        .create<mlir::arith::SubIOp>(loc, leftElem, rightElem)
                        .getResult();
          break;
        case mlir_edsl::BinaryOpType::MUL:
          result =
              isFloat
                  ? builder
                        .create<mlir::arith::MulFOp>(loc, leftElem, rightElem)
                        .getResult()
                  : builder
                        .create<mlir::arith::MulIOp>(loc, leftElem, rightElem)
                        .getResult();
          break;
        case mlir_edsl::BinaryOpType::DIV:
          result =
              isFloat
                  ? builder
                        .create<mlir::arith::DivFOp>(loc, leftElem, rightElem)
                        .getResult()
                  : builder
                        .create<mlir::arith::DivSIOp>(loc, leftElem, rightElem)
                        .getResult();
          break;
        default:
          throw std::runtime_error("Unknown binary operation");
        }

        builder.create<mlir::memref::StoreOp>(loc, result, resultArray, iv);

        // Yield with no values (since we have no iter_args)
        builder.create<mlir::scf::YieldOp>(loc);
      });

  builder.setInsertionPointAfter(forOp);
  return resultArray;
}

} // namespace mlir_edsl
