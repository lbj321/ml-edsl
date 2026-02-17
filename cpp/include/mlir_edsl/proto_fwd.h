#pragma once

#include <cstdint>

namespace mlir_edsl {

// Protobuf uses -1 as the sentinel for dynamic tensor dimensions.
// C++ code should compare against this constant, then map to
// mlir::ShapedType::kDynamic before creating MLIR types.
constexpr int64_t kProtoDynamicDim = -1;

// Enums
enum ComparisonPredicate : int;
enum BinaryOpType : int;

// Core AST nodes
class ASTNode;
class ScalarNode;
class ArrayNode;
class ControlFlowNode;
class FunctionNode;
class BindingNode;
class TensorNode;
class FunctionDef;

// Type specifications
class TypeSpec;
class ScalarTypeSpec;
class MemRefTypeSpec;
class TensorTypeSpec;

// Scalar operations
class Constant;
class BinaryOp;
class CompareOp;
class CastOp;

// Control flow operations
class IfOp;
class ForLoopOp;

// Function operations
class Parameter;
class CallOp;

// Tensor operations
class TensorFromElements;
class TensorExtract;
class TensorInsert;

// Binding operations
class LetBinding;
class ValueReference;

} // namespace mlir_edsl
