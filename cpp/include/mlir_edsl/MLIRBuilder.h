#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include <pybind11/pybind11.h>
#include <unordered_set>

// Forward declarations for dialect builders
namespace mlir_edsl {
class ArithBuilder;
class SCFBuilder;
class MemRefBuilder;
}

// Forward declarations for protobuf classes
#include "mlir_edsl/proto_fwd.h"

namespace mlir_edsl {

class MLIRBuilder {
public:
  MLIRBuilder();
  ~MLIRBuilder();

  // ==================== INITIALIZATION ====================
  void initializeModule();

  // ==================== CORE API (Exposed to Python) ====================
  // Main compilation entry point - takes FunctionDef protobuf object
  void compileFunctionFromDef(const mlir_edsl::FunctionDef &func_def);

  // ==================== INSPECTION ====================
  std::string getMLIRString();
  std::string getLLVMIRString();

  // ==================== MANAGEMENT ====================
  bool hasFunction(const std::string &name) const;
  void clearModule();
  std::vector<std::string> listFunctions() const;

  // ==================== PUBLIC UTILITIES (for dialect builders) ====================
  // Unified type conversion (algebraic type system)
  mlir::Type convertType(const mlir_edsl::TypeSpec &typeSpec) const;

  mlir::Value buildFromProtobufNode(const mlir_edsl::ASTNode &node);

  // Infrastructure utilities (used by multiple dialect builders)
  mlir::Value buildIndexConstant(int64_t value);
  mlir::Value castToIndexType(mlir::Value value);

private:
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::func::FuncOp currentFunction;

  // Helper methods
  bool isIntegerType(mlir::Type type) const;
  bool isFloatType(mlir::Type type) const;

  // Type conversion helpers (algebraic type system)
  mlir::Type convertScalarType(const mlir_edsl::ScalarTypeSpec &scalarSpec) const;
  mlir::Type convertMemRefType(const mlir_edsl::MemRefTypeSpec &memrefSpec) const;

  // Type validation helpers
  bool isValidParameterType(const mlir_edsl::TypeSpec &type) const;
  bool isValidReturnType(const mlir_edsl::TypeSpec &type) const;

  // Internal function building (not exposed to Python)
  void createFunction(
      const std::string &name,
      const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>> &params,
      mlir::Type returnType);
  void finalizeFunction(const std::string &name, mlir::Value result);
  mlir::Value callFunction(const std::string &name,
                           const std::vector<mlir::Value> &args);
  mlir::Value getParameter(const std::string &name);
  void reset();

  // Type promotion helper (explicit target type from Python)
  std::pair<mlir::Value, mlir::Value>
  promoteToMatchDataType(mlir::Value lhs, mlir::Value rhs, mlir::Type targetType);

  // AST node category dispatchers (two-tier dispatch for scalability)
  mlir::Value buildFromScalarNode(const mlir_edsl::ScalarNode &node);
  mlir::Value buildFromArrayNode(const mlir_edsl::ArrayNode &node);
  mlir::Value buildFromControlFlowNode(const mlir_edsl::ControlFlowNode &node);
  mlir::Value buildFromFunctionNode(const mlir_edsl::FunctionNode &node);
  mlir::Value buildFromBindingNode(const mlir_edsl::BindingNode &node);

  // Scalar node handlers
  mlir::Value handleConstant(const mlir_edsl::Constant &constant);
  mlir::Value handleBinaryOp(const mlir_edsl::BinaryOp &op);
  mlir::Value handleCompareOp(const mlir_edsl::CompareOp &op);
  mlir::Value handleCastOp(const mlir_edsl::CastOp &op);

  // Control flow node handlers
  mlir::Value handleIfOp(const mlir_edsl::IfOp &op);

  // Function node handlers
  mlir::Value handleParameter(const mlir_edsl::Parameter &param);
  mlir::Value handleCallOp(const mlir_edsl::CallOp &op);

  // Binding node handlers
  mlir::Value handleLetBinding(const mlir_edsl::LetBinding &binding);
  mlir::Value handleValueRef(const mlir_edsl::ValueReference &ref);

  // Dialect builders
  std::unique_ptr<mlir_edsl::ArithBuilder> arithBuilder;
  std::unique_ptr<mlir_edsl::SCFBuilder> scfBuilder;
  std::unique_ptr<mlir_edsl::MemRefBuilder> memrefBuilder;

  std::unordered_map<std::string, mlir::Value> parameterMap;
  std::unordered_map<std::string, mlir::func::FuncOp> functionTable;
  std::unordered_map<int64_t, mlir::Value> valueCache;  // SSA value cache for let bindings

  std::unordered_set<std::string> compiledFunctions;
};

} // namespace mlir_edsl