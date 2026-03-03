// cpp/src/builders/LinalgBuilder.cpp
#include "mlir_edsl/LinalgBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"

#include <stdexcept>

namespace {

/// Build arith.constant 0 for any numeric type, using ConstantOp + explicit
/// attribute consistently (same pattern as ArithBuilder::buildConstant).
mlir::Value buildZero(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type elemType) {
  if (mlir::isa<mlir::FloatType>(elemType))
    return builder.create<mlir::arith::ConstantOp>(
        loc, mlir::FloatAttr::get(elemType, 0.0));
  if (mlir::isa<mlir::IntegerType>(elemType))
    return builder.create<mlir::arith::ConstantOp>(
        loc, mlir::IntegerAttr::get(elemType, 0));
  throw std::runtime_error("buildZero: unsupported element type");
}

} // anonymous namespace

namespace mlir_edsl {

LinalgBuilder::LinalgBuilder(mlir::OpBuilder &builder,
                             mlir::MLIRContext *context, MLIRBuilder *parent)
    : builder(builder), context(context), parent(parent) {}

mlir::Value LinalgBuilder::buildDot(const mlir_edsl::LinalgDot &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the 1D memref inputs
  mlir::Value lhs = parent->buildFromProtobufNode(node.lhs());
  mlir::Value rhs = parent->buildFromProtobufNode(node.rhs());

  // 2. Determine element type from lhs memref
  auto lhsMemRefType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  if (!lhsMemRefType) {
    throw std::runtime_error("linalg.dot: lhs must be a memref type");
  }
  mlir::Type elemType = lhsMemRefType.getElementType();

  // 3. Allocate a 0-D memref (scalar accumulator) and zero-initialise
  mlir::MemRefType outType = mlir::MemRefType::get({}, elemType);
  mlir::Value out = builder.create<mlir::memref::AllocaOp>(loc, outType);

  mlir::Value zero = buildZero(builder, loc, elemType);
  builder.create<mlir::memref::StoreOp>(loc, zero, out,
                                        mlir::ValueRange{} /*no indices*/);

  // 4. Emit linalg.dot ins(%lhs, %rhs) outs(%out)
  builder.create<mlir::linalg::DotOp>(loc, mlir::ValueRange{lhs, rhs},
                                      mlir::ValueRange{out});

  // 5. Load the scalar result from the 0-D accumulator
  return builder.create<mlir::memref::LoadOp>(loc, out,
                                              mlir::ValueRange{} /*no indices*/);
}

mlir::Value LinalgBuilder::buildMatmul(const mlir_edsl::LinalgMatmul &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the 2D memref inputs
  mlir::Value lhs = parent->buildFromProtobufNode(node.lhs());
  mlir::Value rhs = parent->buildFromProtobufNode(node.rhs());

  // 2. Determine output type from protobuf TypeSpec
  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outMemRefType = mlir::dyn_cast<mlir::MemRefType>(outMLIRType);
  if (!outMemRefType) {
    throw std::runtime_error("linalg.matmul: out_type must be a memref type");
  }
  mlir::Type elemType = outMemRefType.getElementType();

  // 3. Allocate and zero-initialise the output memref
  mlir::Value tmp = builder.create<mlir::memref::AllocaOp>(loc, outMemRefType);

  mlir::Value zero = buildZero(builder, loc, elemType);

  // 4. Zero-fill via linalg.fill
  builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero},
                                       mlir::ValueRange{tmp});

  // 5. Emit linalg.matmul ins(%lhs, %rhs) outs(%tmp)
  builder.create<mlir::linalg::MatmulOp>(loc, mlir::ValueRange{lhs, rhs},
                                         mlir::ValueRange{tmp});

  // 6. Return the output memref; finalizeFunction will copy it to out-param
  return tmp;
}

mlir::Value LinalgBuilder::buildMap(const mlir_edsl::LinalgMap &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the 1D memref input
  mlir::Value input = parent->buildFromProtobufNode(node.input());
  auto inputType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  if (!inputType)
    throw std::runtime_error("tensor_map: input must be a memref type");

  // 2. Determine output type and allocate output memref
  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outType = mlir::dyn_cast<mlir::MemRefType>(outMLIRType);
  if (!outType)
    throw std::runtime_error("tensor_map: out_type must be a memref type");

  mlir::Value output = builder.create<mlir::memref::AllocaOp>(loc, outType);

  // 3. Build affine maps and iterator types for linalg.generic
  int64_t ndim = inputType.getRank();
  auto identityMap = mlir::AffineMap::getMultiDimIdentityMap(ndim, context);
  llvm::SmallVector<mlir::AffineMap> indexingMaps = {identityMap, identityMap};
  llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
      ndim, mlir::utils::IteratorType::parallel);

  int64_t elementNodeId = node.element_node_id();
  const auto &bodyProto = node.body();

  // 4. Emit linalg.generic with a parallel body.
  // GenericOp::build does InsertionGuard(builder) + setInsertionPointToStart on
  // the outer builder before invoking the region callback — same as scf::ForOp.
  // So capturing 'builder' by [&] inside the lambda is correct.
  builder.create<mlir::linalg::GenericOp>(
      loc,
      /*resultTensorTypes=*/mlir::TypeRange{},
      /*inputs=*/mlir::ValueRange{input},
      /*outputs=*/mlir::ValueRange{output},
      indexingMaps,
      iteratorTypes,
      [&](mlir::OpBuilder &, mlir::Location innerLoc, mlir::ValueRange blockArgs) {
        // blockArgs[0] = current input element; blockArgs[1] = current output
        // element (unused — we compute and yield a fresh value).
        parent->setValueCacheEntry(elementNodeId, blockArgs[0]);
        mlir::Value result = parent->buildFromProtobufNode(bodyProto);
        builder.create<mlir::linalg::YieldOp>(innerLoc, result);
      });

  return output;
}

} // namespace mlir_edsl
