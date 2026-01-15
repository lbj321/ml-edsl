#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include <pybind11/pybind11.h>
#include <unordered_set>

// Forward declarations for dialect builders
namespace mlir_edsl {
class ArithBuilder;
class SCFBuilder;
class MemRefBuilder;
}

namespace mlir_edsl {

// Forward declarations for protobuf types
enum ValueType : int;
enum ComparisonPredicate : int;
enum BinaryOpType : int;
class ASTNode;
class ArrayTypeSpec;
class FunctionDef;

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
  mlir::Type protoTypeToMLIRType(mlir_edsl::ValueType protoType) const;
  mlir::Type arrayTypeSpecToMLIRType(const mlir_edsl::ArrayTypeSpec &arraySpec) const;
  mlir::Value buildFromProtobufNode(const mlir_edsl::ASTNode &node);

  // Infrastructure utilities (used by multiple dialect builders)
  mlir::Value buildIndexConstant(int64_t value);
  mlir::Value castToIndexType(mlir::Value value);

private:
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::ModuleOp module;
  mlir::func::FuncOp currentFunction;

  // Helper methods
  bool isIntegerType(mlir::Type type) const;
  bool isFloatType(mlir::Type type) const;

  // Internal function building (not exposed to Python)
  void createFunction(
      const std::string &name,
      const std::vector<std::pair<std::string, mlir_edsl::ValueType>> &params,
      mlir::Type returnType);
  void finalizeFunction(const std::string &name, mlir::Value result);
  mlir::Value callFunction(const std::string &name,
                           const std::vector<mlir::Value> &args);
  mlir::Value getParameter(const std::string &name);
  void reset();

  // Type promotion helper (explicit target type from Python)
  std::pair<mlir::Value, mlir::Value>
  promoteToType(mlir::Value lhs, mlir::Value rhs, mlir::Type targetType);

  // AST node handlers
  mlir::Value handleLetBinding(const mlir_edsl::ASTNode &node);
  mlir::Value handleValueRef(const mlir_edsl::ASTNode &node);
  mlir::Value handleConstant(const mlir_edsl::ASTNode &node);
  mlir::Value handleParameter(const mlir_edsl::ASTNode &node);
  mlir::Value handleBinaryOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleCompareOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleIfOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleCallOp(const mlir_edsl::ASTNode &node);
  // mlir::Value handleForLoopOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleWhileLoopOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleCastOp(const mlir_edsl::ASTNode &node);
  mlir::Value handleArrayLiteral(const mlir_edsl::ASTNode &node);
  mlir::Value handleArrayAccess(const mlir_edsl::ASTNode &node);
  mlir::Value handleArrayStore(const mlir_edsl::ASTNode &node);
  mlir::Value handleArrayBinaryOp(const mlir_edsl::ASTNode &node);

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