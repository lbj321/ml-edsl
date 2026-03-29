// cpp/src/builders/LinalgBuilder.cpp
#include "mlir_edsl/LinalgBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"

#include <stdexcept>

namespace {

/// Apply a BinaryOpType arith operation to two scalar values.
mlir::Value applyArithOp(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir_edsl::BinaryOpType opType, mlir::Value lhs,
                         mlir::Value rhs) {
  bool isFloat = mlir::isa<mlir::FloatType>(lhs.getType());
  switch (opType) {
  case mlir_edsl::ADD:
    return isFloat ? builder.create<mlir::arith::AddFOp>(loc, lhs, rhs)
                         .getResult()
                   : builder.create<mlir::arith::AddIOp>(loc, lhs, rhs)
                         .getResult();
  case mlir_edsl::SUB:
    return isFloat ? builder.create<mlir::arith::SubFOp>(loc, lhs, rhs)
                         .getResult()
                   : builder.create<mlir::arith::SubIOp>(loc, lhs, rhs)
                         .getResult();
  case mlir_edsl::MUL:
    return isFloat ? builder.create<mlir::arith::MulFOp>(loc, lhs, rhs)
                         .getResult()
                   : builder.create<mlir::arith::MulIOp>(loc, lhs, rhs)
                         .getResult();
  case mlir_edsl::DIV:
    return isFloat ? builder.create<mlir::arith::DivFOp>(loc, lhs, rhs)
                         .getResult()
                   : builder.create<mlir::arith::DivSIOp>(loc, lhs, rhs)
                         .getResult();
  default:
    throw std::runtime_error("applyArithOp: unsupported op type");
  }
}

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
  // out-param is already a tensor (writable=true set on function arg).
  mlir::Value dest;
  if (outParam)
    dest = outParam;
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
  // buffer. out-param is already a tensor (writable=true set on function arg).
  mlir::Value init;
  if (outParam)
    init = outParam;
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

mlir::Value LinalgBuilder::buildBinaryOp(const mlir_edsl::LinalgBinaryOp &node,
                                          mlir::Value outParam) {
  auto loc = builder.getUnknownLoc();
  auto broadcastMode = node.broadcast();
  auto opType = node.op_type();

  mlir::Type outMLIRType = parent->convertType(node.out_type());
  auto outTensorType = mlir::dyn_cast<mlir::RankedTensorType>(outMLIRType);
  if (!outTensorType)
    throw std::runtime_error("LinalgBinaryOp: out_type must be a tensor type");

  // Allocate output tensor. out-param is already a tensor (writable=true set
  // on function arg); use it directly as linalg init.
  mlir::Value init;
  if (outParam)
    init = outParam;
  else
    init = builder.create<mlir::tensor::EmptyOp>(loc, outTensorType,
                                                 mlir::ValueRange{});

  if (broadcastMode == mlir_edsl::TENSOR_BIAS_RIGHT ||
      broadcastMode == mlir_edsl::TENSOR_BIAS_LEFT) {
    // ----------------------------------------------------------------
    // Bias broadcast: [M,N] op [N]  (or [N] op [M,N])
    // Step 1: linalg.broadcast [N] → [M,N]
    // Step 2: linalg.map(arith op) over the two [M,N] tensors
    // ----------------------------------------------------------------
    mlir::Value matrix = parent->buildFromProtobufNode(
        broadcastMode == mlir_edsl::TENSOR_BIAS_RIGHT ? node.lhs() : node.rhs());
    mlir::Value bias = parent->buildFromProtobufNode(
        broadcastMode == mlir_edsl::TENSOR_BIAS_RIGHT ? node.rhs() : node.lhs());

    // Broadcast bias [N] → [M,N] along dimension 0
    mlir::Value broadcastInit = builder.create<mlir::tensor::EmptyOp>(
        loc, outTensorType, mlir::ValueRange{});
    auto broadcastOp = builder.create<mlir::linalg::BroadcastOp>(
        loc, bias, broadcastInit, builder.getDenseI64ArrayAttr({0}));
    mlir::Value bias2D = broadcastOp->getResult(0);

    // linalg.map with two inputs: matrix and bias2D
    auto mapOp = builder.create<mlir::linalg::MapOp>(
        loc,
        /*inputs=*/mlir::ValueRange{matrix, bias2D},
        /*init=*/init,
        [&](mlir::OpBuilder &, mlir::Location innerLoc,
            mlir::ValueRange blockArgs) {
          mlir::Value result =
              applyArithOp(builder, innerLoc, opType, blockArgs[0], blockArgs[1]);
          builder.create<mlir::linalg::YieldOp>(innerLoc, result);
        });
    return mapOp->getResult(0);
  }

  // ----------------------------------------------------------------
  // NONE: same-shape element-wise
  // SCALAR_LEFT / SCALAR_RIGHT: one operand is a scalar
  // ----------------------------------------------------------------
  mlir::Value lhs = parent->buildFromProtobufNode(node.lhs());
  mlir::Value rhs = parent->buildFromProtobufNode(node.rhs());

  if (broadcastMode == mlir_edsl::NONE) {
    auto mapOp = builder.create<mlir::linalg::MapOp>(
        loc,
        /*inputs=*/mlir::ValueRange{lhs, rhs},
        /*init=*/init,
        [&](mlir::OpBuilder &, mlir::Location innerLoc,
            mlir::ValueRange blockArgs) {
          mlir::Value result =
              applyArithOp(builder, innerLoc, opType, blockArgs[0], blockArgs[1]);
          builder.create<mlir::linalg::YieldOp>(innerLoc, result);
        });
    return mapOp->getResult(0);
  }

  // SCALAR_LEFT or SCALAR_RIGHT: scalar captured directly from outer scope
  mlir::Value tensorInput =
      (broadcastMode == mlir_edsl::SCALAR_RIGHT) ? lhs : rhs;
  mlir::Value scalarInput =
      (broadcastMode == mlir_edsl::SCALAR_RIGHT) ? rhs : lhs;

  auto mapOp = builder.create<mlir::linalg::MapOp>(
      loc,
      /*inputs=*/mlir::ValueRange{tensorInput},
      /*init=*/init,
      [&](mlir::OpBuilder &, mlir::Location innerLoc,
          mlir::ValueRange blockArgs) {
        mlir::Value elemVal = blockArgs[0];
        mlir::Value result =
            (broadcastMode == mlir_edsl::SCALAR_RIGHT)
                ? applyArithOp(builder, innerLoc, opType, elemVal, scalarInput)
                : applyArithOp(builder, innerLoc, opType, scalarInput, elemVal);
        builder.create<mlir::linalg::YieldOp>(innerLoc, result);
      });
  return mapOp->getResult(0);
}

} // namespace mlir_edsl
