#include "mlir_edsl/ShapeUtils.h"
#include "mlir_edsl/proto_fwd.h"

#include "ast.pb.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"

#include <stdexcept>
#include <string>

namespace mlir_edsl {

llvm::SmallVector<int64_t>
buildShapeFromProto(const google::protobuf::RepeatedField<int64_t> &protoShape) {
  llvm::SmallVector<int64_t> shape(protoShape.begin(), protoShape.end());
  for (auto &d : shape) {
    if (d == kProtoDynamicDim)
      d = mlir::ShapedType::kDynamic;
  }
  return shape;
}

void validateRank(llvm::ArrayRef<int64_t> shape, llvm::StringRef kindName) {
  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("Only 1D, 2D, and 3D " + kindName.str() +
                             "s supported, got " +
                             std::to_string(shape.size()) + "D");
  }
}

void validateScalarElement(const mlir_edsl::TypeSpec &elementType,
                           llvm::StringRef kindName) {
  if (!elementType.has_scalar()) {
    throw std::runtime_error(kindName.str() +
                             " element type must be scalar (i32, f32, or bool)");
  }
}

void rejectDynamicDims(llvm::ArrayRef<int64_t> shape, llvm::StringRef kindName) {
  for (auto d : shape) {
    if (d == mlir::ShapedType::kDynamic)
      throw std::runtime_error("Dynamic " + kindName.str() +
                               " dimensions not supported");
  }
}

} // namespace mlir_edsl
