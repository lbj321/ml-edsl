#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include <pybind11/pybind11.h>

// Forward declarations for dialect builders
namespace mlir_edsl {
class ArithBuilder;
class SCFBuilder;
class MemRefBuilder;
class TensorBuilder;
}

// Forward declarations for protobuf classes
#include "mlir_edsl/proto_fwd.h"

namespace mlir_edsl {

/// Low-level IR builder: converts AST nodes to MLIR values.
/// Does not own the MLIR context, OpBuilder, or module.
/// Function-level state (parameterMap, functionTable) is injected by MLIRCompiler.
class MLIRBuilder {
public:
  MLIRBuilder(mlir::MLIRContext *context, mlir::OpBuilder *builder);
  ~MLIRBuilder();

  // ==================== DEPENDENCY INJECTION ====================
  void setParameterMap(std::unordered_map<std::string, mlir::Value> *paramMap);
  void setFunctionTable(std::unordered_map<std::string, mlir::func::FuncOp> *funcTable);
  void clearValueCache();

  // ==================== CORE BUILDING ====================
  mlir::Value buildFromProtobufNode(const mlir_edsl::ASTNode &node);

  // ==================== PUBLIC UTILITIES (for dialect builders) ====================
  mlir::Type convertType(const mlir_edsl::TypeSpec &typeSpec) const;
  mlir::Value buildIndexConstant(int64_t value);
  mlir::Value castToIndexType(mlir::Value value);

private:
  // Non-owning references (owned by MLIRCompiler)
  mlir::MLIRContext *context;
  mlir::OpBuilder *builder;

  // Injected state (non-owning pointers, owned by MLIRCompiler)
  std::unordered_map<std::string, mlir::Value> *parameterMap = nullptr;
  std::unordered_map<std::string, mlir::func::FuncOp> *functionTable = nullptr;

  // Internal state (owned by MLIRBuilder)
  std::unordered_map<int64_t, mlir::Value> valueCache;

  // Dialect builders
  std::unique_ptr<mlir_edsl::ArithBuilder> arithBuilder;
  std::unique_ptr<mlir_edsl::SCFBuilder> scfBuilder;
  std::unique_ptr<mlir_edsl::MemRefBuilder> memrefBuilder;
  std::unique_ptr<mlir_edsl::TensorBuilder> tensorBuilder;

  // Helper methods
  bool isIntegerType(mlir::Type type) const;
  bool isFloatType(mlir::Type type) const;

  // Type conversion helpers
  mlir::Type convertScalarType(const mlir_edsl::ScalarTypeSpec &scalarSpec) const;
  mlir::Type convertMemRefType(const mlir_edsl::MemRefTypeSpec &memrefSpec) const;
  mlir::Type convertTensorType(const mlir_edsl::TensorTypeSpec &tensorSpec) const;

  // Type promotion helper
  std::pair<mlir::Value, mlir::Value>
  promoteToMatchDataType(mlir::Value lhs, mlir::Value rhs, mlir::Type targetType);

  // AST node category dispatchers
  mlir::Value buildFromScalarNode(const mlir_edsl::ScalarNode &node);
  mlir::Value buildFromArrayNode(const mlir_edsl::ArrayNode &node);
  mlir::Value buildFromControlFlowNode(const mlir_edsl::ControlFlowNode &node);
  mlir::Value buildFromFunctionNode(const mlir_edsl::FunctionNode &node);
  mlir::Value buildFromTensorNode(const mlir_edsl::TensorNode &node);
  mlir::Value buildFromBindingNode(const mlir_edsl::BindingNode &node);

  // Node handlers
  mlir::Value handleConstant(const mlir_edsl::Constant &constant);
  mlir::Value handleBinaryOp(const mlir_edsl::BinaryOp &op);
  mlir::Value handleCompareOp(const mlir_edsl::CompareOp &op);
  mlir::Value handleCastOp(const mlir_edsl::CastOp &op);
  mlir::Value handleIfOp(const mlir_edsl::IfOp &op);
  mlir::Value handleParameter(const mlir_edsl::Parameter &param);
  mlir::Value handleCallOp(const mlir_edsl::CallOp &op);
  mlir::Value handleLetBinding(const mlir_edsl::LetBinding &binding);
  mlir::Value handleValueRef(const mlir_edsl::ValueReference &ref);

  // Internal helpers
  mlir::Value getParameter(const std::string &name);
  mlir::Value callFunction(const std::string &name,
                           const std::vector<mlir::Value> &args);
};

} // namespace mlir_edsl
