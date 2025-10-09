#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include <pybind11/pybind11.h>
#include <unordered_set>

namespace mlir_edsl {

// Forward declarations for protobuf types
enum ComparisonPredicate : int;
enum BinaryOpType : int;
class ASTNode;

class MLIRBuilder {
public:
  MLIRBuilder();
  ~MLIRBuilder();

  // Initialize a new MLIR module
  void initializeModule();

  mlir::Value buildFromProtoBuf(const std::string &buffer);

  // Create constants
  mlir::Value buildConstant(int32_t value);
  mlir::Value buildConstant(float value);

  // Create binary operations
  mlir::Value buildAdd(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildSub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildMul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildDiv(mlir::Value lhs, mlir::Value rhs);

  mlir::Value buildCompare(mlir_edsl::ComparisonPredicate predicate,
                           mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildIf(mlir::Value condition, mlir::Value thenValue,
                      mlir::Value elseValue);

  mlir::Value buildFor(
      mlir::Value start, mlir::Value end, mlir::Value step,
      mlir::Value init_value,
      std::function<mlir::Value(mlir::Value iv, mlir::Value iter_arg)> body_fn);

  // For loop with predefined operation
  mlir::Value buildForWithOp(mlir::Value start, mlir::Value end,
                             mlir::Value step, mlir::Value init_value,
                             mlir_edsl::BinaryOpType operation);

  mlir::Value buildWhileWithOp(mlir::Value init, mlir::Value target,
                               mlir_edsl::BinaryOpType operation,
                               mlir_edsl::ComparisonPredicate condition);

  mlir::Value buildWhile(mlir::Value init,
                         std::function<mlir::Value(mlir::Value)> condition_fn,
                         std::function<mlir::Value(mlir::Value)> body_fn);

  // Type conversion
  mlir::Value convertIntToFloat(mlir::Value intValue);

  void compileFunctionFromAST(
      const std::string &name,
      const std::vector<std::pair<std::string, std::string>> &params,
      const std::string &return_type, const std::string &ast_protobuf_bytes);

  // Function generation
  void
  createFunction(const std::string &name,
                 const std::vector<std::pair<std::string, std::string>> &params,
                 const std::string &return_type);

  void finalizeFunction(const std::string &name, mlir::Value result);

  mlir::Value callFunction(const std::string &name,
                           const std::vector<mlir::Value> &args);

  // Get generated MLIR as string
  std::string getMLIRString();

  // Get generated LLVM IR as string
  std::string getLLVMIRString();

  mlir::Value getParameter(const std::string &name);

  bool hasFunction(const std::string &name) const;
  void clearModule();
  std::vector<std::string> listFunctions() const;

  // Reset the builder for a new function
  void reset();

private:
  std::unique_ptr<mlir::MLIRContext> context;
  std::unique_ptr<mlir::OpBuilder> builder;
  mlir::ModuleOp module;
  mlir::func::FuncOp currentFunction;

  // Helper methods
  mlir::Type getIntegerType() const;
  mlir::Type getFloatType() const;
  mlir::Type getBoolType() const;
  bool isIntegerType(mlir::Type type) const;
  bool isFloatType(mlir::Type type) const;

  mlir::arith::CmpIPredicate
  protobufToIntPredicate(mlir_edsl::ComparisonPredicate pred) const;
  mlir::arith::CmpFPredicate
  protobufToFloatPredicate(mlir_edsl::ComparisonPredicate pred) const;

  mlir::Value buildFromProtobufNode(const mlir_edsl::ASTNode &node);

  std::string mapComparisonPredicate(mlir_edsl::ComparisonPredicate pred);

  // Type promotion helpers
  std::pair<mlir::Value, mlir::Value> promoteTypes(mlir::Value lhs,
                                                   mlir::Value rhs);
  mlir::Type getPromotedType(mlir::Type lhs, mlir::Type rhs) const;

  std::unordered_map<std::string, mlir::Value> parameterMap;
  std::unordered_map<std::string, mlir::func::FuncOp> functionTable;

  std::unordered_set<std::string> compiledFunctions;
};

} // namespace mlir_edsl