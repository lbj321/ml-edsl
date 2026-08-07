#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <google/protobuf/repeated_field.h>
#include <cstdint>

#include "mlir_edsl/proto_fwd.h"

namespace mlir_edsl {

/// Builds a shape vector from a protobuf repeated int64 field, mapping the
/// protobuf dynamic-dim sentinel (kProtoDynamicDim) to mlir::ShapedType::kDynamic.
/// Makes no policy decision about whether dynamic dims are allowed here.
llvm::SmallVector<int64_t>
buildShapeFromProto(const google::protobuf::RepeatedField<int64_t> &protoShape);

/// Throws if shape is empty or has more than 3 dims. kindName (e.g. "array",
/// "tensor") is used in the error message.
void validateRank(llvm::ArrayRef<int64_t> shape, llvm::StringRef kindName);

/// Throws "<kindName> element type must be scalar" if elementType is not a
/// ScalarTypeSpec (nested memref-of-memref/tensor-of-tensor unsupported).
void validateScalarElement(const mlir_edsl::TypeSpec &elementType,
                           llvm::StringRef kindName);

/// Throws if any dim in shape is the dynamic-dim sentinel. Used by callers
/// that don't support dynamic dimensions at this boundary.
void rejectDynamicDims(llvm::ArrayRef<int64_t> shape, llvm::StringRef kindName);

} // namespace mlir_edsl
