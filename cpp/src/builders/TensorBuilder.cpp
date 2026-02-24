// cpp/src/builders/TensorBuilder.cpp
#include "mlir_edsl/TensorBuilder.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir_edsl {

TensorBuilder::TensorBuilder(mlir::OpBuilder &builder,
                             mlir::MLIRContext *context,
                             MLIRBuilder *parent)
    : builder(builder), context(context), parent(parent) {}

mlir::Value TensorBuilder::buildFromElements(const TensorFromElements &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build element values
  llvm::SmallVector<mlir::Value> elements;
  for (int i = 0; i < node.elements_size(); ++i) {
    elements.push_back(parent->buildFromProtobufNode(node.elements(i)));
  }

  // 2. Determine tensor type from protobuf TypeSpec
  mlir::Type tensorType = parent->convertType(node.type());

  // 3. Create tensor.from_elements op
  return builder.create<mlir::tensor::FromElementsOp>(loc, tensorType, elements);
}

mlir::Value TensorBuilder::buildExtract(const TensorExtract &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the tensor value
  mlir::Value tensor = parent->buildFromProtobufNode(node.tensor());

  // 2. Build index values
  llvm::SmallVector<mlir::Value> indices;
  for (int i = 0; i < node.indices_size(); ++i) {
    mlir::Value indexRaw = parent->buildFromProtobufNode(node.indices(i));
    mlir::Value index = parent->castToIndexType(indexRaw);
    indices.push_back(index);
  }

  // 3. Create tensor.extract op
  return builder.create<mlir::tensor::ExtractOp>(loc, tensor, indices);
}

mlir::Value TensorBuilder::buildInsert(const TensorInsert &node) {
  auto loc = builder.getUnknownLoc();

  // 1. Build the tensor value FIRST (for SSA value reuse - LET before REF)
  mlir::Value tensor = parent->buildFromProtobufNode(node.tensor());

  // 2. Build index values
  llvm::SmallVector<mlir::Value> indices;
  for (int i = 0; i < node.indices_size(); ++i) {
    mlir::Value indexRaw = parent->buildFromProtobufNode(node.indices(i));
    mlir::Value index = parent->castToIndexType(indexRaw);
    indices.push_back(index);
  }

  // 3. Build the scalar value to insert (may reference tensor via REF)
  mlir::Value scalar = parent->buildFromProtobufNode(node.value());

  // 4. Create tensor.insert op (returns NEW tensor)
  return builder.create<mlir::tensor::InsertOp>(loc, scalar, tensor, indices);
}

mlir::Value TensorBuilder::buildEmpty(const TensorEmpty &node) {
  auto loc = builder.getUnknownLoc();

  // Extract shape and element type directly from protobuf
  const auto &tensorSpec = node.type().tensor();
  llvm::SmallVector<int64_t> shape(tensorSpec.shape().begin(),
                                   tensorSpec.shape().end());
  // Map protobuf sentinel (-1) to MLIR's kDynamic
  for (auto &d : shape) {
    if (d == kProtoDynamicDim) d = mlir::ShapedType::kDynamic;
  }
  mlir::Type elemType = parent->convertType(tensorSpec.element_type());

  // Build runtime values for dynamic (?) dimensions
  llvm::SmallVector<mlir::Value> dynamicSizes;
  for (int i = 0; i < node.dynamic_dims_size(); ++i) {
    mlir::Value dimVal = parent->buildFromProtobufNode(node.dynamic_dims(i));
    dynamicSizes.push_back(parent->castToIndexType(dimVal));
  }

  auto tensorType = mlir::RankedTensorType::get(shape, elemType);
  return builder.create<mlir::tensor::EmptyOp>(loc, tensorType, dynamicSizes);
}

} // namespace mlir_edsl
