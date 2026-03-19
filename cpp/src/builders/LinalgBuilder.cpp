// cpp/src/builders/LinalgBuilder.cpp
#include "mlir_edsl/LinalgBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include <stdexcept>

namespace {

/// Build arith.constant 0 for any numeric type.
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

  mlir::Value lhs = parent->buildFromProtobufNode(node.lhs());
  mlir::Value rhs = parent->buildFromProtobufNode(node.rhs());

  auto lhsTensorType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  if (!lhsTensorType)
    throw std::runtime_error("linalg.dot: lhs must be a tensor type");
  if (lhsTensorType.getRank() != 1)
    throw std::runtime_error("linalg.dot: lhs must be a 1D tensor");
  mlir::Type elemType = lhsTensorType.getElementType();

  auto rhsTensorType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!rhsTensorType)
    throw std::runtime_error("linalg.dot: rhs must be a tensor type");
  if (rhsTensorType.getRank() != 1)
    throw std::runtime_error("linalg.dot: rhs must be a 1D tensor");
  if (rhsTensorType.getElementType() != elemType)
    throw std::runtime_error("linalg.dot: lhs and rhs element types must match");

  // 0-D tensor accumulator: empty() → fill(zero) → dot → extract scalar
  auto initType = mlir::RankedTensorType::get({}, elemType);
  mlir::Value zero = buildZero(builder, loc, elemType);
  mlir::Value emptyInit =
      builder.create<mlir::tensor::EmptyOp>(loc, initType, mlir::ValueRange{});
  mlir::Value filledInit =
      builder
          .create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero},
                                        mlir::ValueRange{emptyInit})
          .result();
  mlir::Value dotResult =
      builder
          .create<mlir::linalg::DotOp>(loc, mlir::ValueRange{lhs, rhs},
                                       mlir::ValueRange{filledInit})
          .getResult(0);
  return builder.create<mlir::tensor::ExtractOp>(loc, dotResult,
                                                 mlir::ValueRange{});
}

mlir::Value LinalgBuilder::buildMatmul(const mlir_edsl::LinalgMatmul &node,
                                       mlir::Value outParam) {
  auto loc = builder.getUnknownLoc();

  mlir::Value lhs = parent->buildFromProtobufNode(node.lhs());
  mlir::Value rhs = parent->buildFromProtobufNode(node.rhs());

  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outTensorType = mlir::dyn_cast<mlir::RankedTensorType>(outMLIRType);
  if (!outTensorType)
    throw std::runtime_error("linalg.matmul: out_type must be a tensor type");
  mlir::Type elemType = outTensorType.getElementType();
  mlir::Value zero = buildZero(builder, loc, elemType);

  // Use out-param as init tensor so linalg writes directly into Python's
  // buffer. Fall back to tensor.empty for sub-expression use (no out-param).
  mlir::Value dest;
  if (outParam)
    dest = builder
               .create<mlir::bufferization::ToTensorOp>(
                   loc, outTensorType, outParam, /*restrict=*/true,
                   /*writable=*/true)
               .getResult();
  else
    dest = builder.create<mlir::tensor::EmptyOp>(loc, outTensorType,
                                                 mlir::ValueRange{});
  mlir::Value init =
      builder
          .create<mlir::linalg::FillOp>(loc, mlir::ValueRange{zero},
                                        mlir::ValueRange{dest})
          .result();
  return builder
      .create<mlir::linalg::MatmulOp>(loc, mlir::TypeRange{outTensorType},
                                      mlir::ValueRange{lhs, rhs},
                                      mlir::ValueRange{init})
      .getResult(0);
}

mlir::Value LinalgBuilder::buildMap(const mlir_edsl::LinalgMap &node,
                                    mlir::Value outParam) {
  auto loc = builder.getUnknownLoc();

  mlir::Value input = parent->buildFromProtobufNode(node.input());
  if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
    throw std::runtime_error("tensor_map: input must be a tensor type");

  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outTensorType = mlir::dyn_cast<mlir::RankedTensorType>(outMLIRType);
  if (!outTensorType)
    throw std::runtime_error("tensor_map: out_type must be a tensor type");

  int64_t elementNodeId = node.element_node_id();
  const auto &bodyProto = node.body();

  // Use out-param as init tensor so linalg writes directly into Python's
  // buffer.
  mlir::Value init;
  if (outParam)
    init = builder
               .create<mlir::bufferization::ToTensorOp>(
                   loc, outTensorType, outParam, /*restrict=*/true,
                   /*writable=*/true)
               .getResult();
  else
    init = builder.create<mlir::tensor::EmptyOp>(loc, outTensorType,
                                                 mlir::ValueRange{});

  auto mapOp = builder.create<mlir::linalg::MapOp>(
      loc,
      /*inputs=*/mlir::ValueRange{input},
      /*init=*/init,
      [&](mlir::OpBuilder &, mlir::Location innerLoc,
          mlir::ValueRange blockArgs) {
        parent->setValueCacheEntry(elementNodeId, blockArgs[0]);
        mlir::Value result = parent->buildFromProtobufNode(bodyProto);
        builder.create<mlir::linalg::YieldOp>(innerLoc, result);
      });

  return mapOp->getResult(0);
}

mlir::Value LinalgBuilder::buildReduce(const mlir_edsl::LinalgReduce &node) {
  auto loc = builder.getUnknownLoc();

  mlir::Value input = parent->buildFromProtobufNode(node.input());
  auto inputTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputTensorType)
    throw std::runtime_error("linalg.reduce: input must be a tensor type");
  mlir::Type elemType = inputTensorType.getElementType();

  // 0-D tensor init from user-provided scalar value
  auto initType = mlir::RankedTensorType::get({}, elemType);
  mlir::Value initVal = parent->buildFromProtobufNode(node.init());
  mlir::Value initTensor = builder.create<mlir::tensor::FromElementsOp>(
      loc, initType, mlir::ValueRange{initVal});

  int64_t elementNodeId = node.element_node_id();
  int64_t accumNodeId = node.accum_node_id();
  const auto &bodyProto = node.body();

  mlir::Value reduceResult =
      builder
          .create<mlir::linalg::ReduceOp>(
              loc,
              /*inputs=*/mlir::ValueRange{input},
              /*inits=*/mlir::ValueRange{initTensor},
              /*dimensions=*/llvm::ArrayRef<int64_t>{0},
              [&](mlir::OpBuilder &, mlir::Location innerLoc,
                  mlir::ValueRange blockArgs) {
                parent->setValueCacheEntry(elementNodeId, blockArgs[0]);
                parent->setValueCacheEntry(accumNodeId, blockArgs[1]);
                mlir::Value result = parent->buildFromProtobufNode(bodyProto);
                builder.create<mlir::linalg::YieldOp>(innerLoc, result);
              })
          .getResult(0);

  return builder.create<mlir::tensor::ExtractOp>(loc, reduceResult,
                                                 mlir::ValueRange{});
}

} // namespace mlir_edsl
