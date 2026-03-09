// cpp/src/builders/LinalgBuilder.cpp
#include "mlir_edsl/LinalgBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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

mlir::Value LinalgBuilder::buildMatmul(const mlir_edsl::LinalgMatmul &node,
                                       mlir::Value outParam) {
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

  // 3. Use Python-allocated out-param directly, or fall back to alloca
  mlir::Value out = outParam
      ? outParam
      : builder.create<mlir::memref::AllocaOp>(loc, outMemRefType);

  mlir::Value zero = buildZero(builder, loc, elemType);

  // 4. Zero-fill via linalg.fill
  builder.create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero},
                                       mlir::ValueRange{out});

  // 5. Emit linalg.matmul ins(%lhs, %rhs) outs(%out)
  builder.create<mlir::linalg::MatmulOp>(loc, mlir::ValueRange{lhs, rhs},
                                         mlir::ValueRange{out});

  return out;
}

mlir::Value LinalgBuilder::buildMap(const mlir_edsl::LinalgMap &node,
                                    mlir::Value outParam) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the 1D memref input
  mlir::Value input = parent->buildFromProtobufNode(node.input());
  if (!mlir::isa<mlir::MemRefType>(input.getType()))
    throw std::runtime_error("tensor_map: input must be a memref type");

  // 2. Determine output type; use Python-allocated out-param or fall back to alloca
  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outType = mlir::dyn_cast<mlir::MemRefType>(outMLIRType);
  if (!outType)
    throw std::runtime_error("tensor_map: out_type must be a memref type");

  mlir::Value output = outParam
      ? outParam
      : builder.create<mlir::memref::AllocaOp>(loc, outType);

  int64_t elementNodeId = node.element_node_id();
  const auto &bodyProto = node.body();

  // 3. Emit linalg.map — the named element-wise op (cleaner than linalg.generic:
  // no affine maps or iterator types needed, and blockArgs contains only the
  // input element, not the output).
  // MapOp::build calls buildGenericRegion which does InsertionGuard(builder) +
  // createBlock, so capturing 'builder' by [&] is correct.
  builder.create<mlir::linalg::MapOp>(
      loc,
      /*inputs=*/mlir::ValueRange{input},
      /*init=*/output,
      [&](mlir::OpBuilder &, mlir::Location innerLoc, mlir::ValueRange blockArgs) {
        // blockArgs[0] = current input element only (MapOp omits the output arg)
        parent->setValueCacheEntry(elementNodeId, blockArgs[0]);
        mlir::Value result = parent->buildFromProtobufNode(bodyProto);
        builder.create<mlir::linalg::YieldOp>(innerLoc, result);
      });

  return output;
}

mlir::Value LinalgBuilder::buildReduce(const mlir_edsl::LinalgReduce &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the 1D memref input
  mlir::Value input = parent->buildFromProtobufNode(node.input());
  auto inputType = mlir::dyn_cast<mlir::MemRefType>(input.getType());
  if (!inputType)
    throw std::runtime_error("linalg.reduce: input must be a memref type");
  mlir::Type elemType = inputType.getElementType();

  // 2. Allocate a 0-D memref and initialize it with the init value
  mlir::MemRefType outType = mlir::MemRefType::get({}, elemType);
  mlir::Value out = builder.create<mlir::memref::AllocaOp>(loc, outType);

  mlir::Value initVal = parent->buildFromProtobufNode(node.init());
  builder.create<mlir::memref::StoreOp>(loc, initVal, out,
                                        mlir::ValueRange{} /*no indices*/);

  int64_t elementNodeId = node.element_node_id();
  int64_t accumNodeId = node.accum_node_id();
  const auto &bodyProto = node.body();

  // 3. Emit linalg.reduce — body receives (input_element, accumulator)
  builder.create<mlir::linalg::ReduceOp>(
      loc,
      /*inputs=*/mlir::ValueRange{input},
      /*inits=*/mlir::ValueRange{out},
      /*dimensions=*/llvm::ArrayRef<int64_t>{0},
      [&](mlir::OpBuilder &, mlir::Location innerLoc,
          mlir::ValueRange blockArgs) {
        // blockArgs[0] = input element, blockArgs[1] = accumulator
        parent->setValueCacheEntry(elementNodeId, blockArgs[0]);
        parent->setValueCacheEntry(accumNodeId, blockArgs[1]);
        mlir::Value result = parent->buildFromProtobufNode(bodyProto);
        builder.create<mlir::linalg::YieldOp>(innerLoc, result);
      });

  // 4. Load and return the scalar result
  return builder.create<mlir::memref::LoadOp>(loc, out,
                                              mlir::ValueRange{} /*no indices*/);
}

} // namespace mlir_edsl
