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

namespace mlir_edsl {

class MLIRBuilder {
public:
  MLIRBuilder();
  ~MLIRBuilder();

  // Initialize a new MLIR module
  void initializeModule();

  // Create constants
  mlir::Value buildConstant(int32_t value);
  mlir::Value buildConstant(float value);

  // Create binary operations
  mlir::Value buildAdd(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildSub(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildMul(mlir::Value lhs, mlir::Value rhs);
  mlir::Value buildDiv(mlir::Value lhs, mlir::Value rhs);

  // Type conversion
  mlir::Value convertIntToFloat(mlir::Value intValue);

  mlir::Value buildCompare(const std::string &predicate, mlir::Value lhs,
                           mlir::Value rhs);
  mlir::Value buildIf(mlir::Value condition, mlir::Value thenValue,
                      mlir::Value elseValue);

  mlir::Value buildFor(
      mlir::Value start, mlir::Value end, mlir::Value step,
      mlir::Value init_value,
      std::function<mlir::Value(mlir::Value iv, mlir::Value iter_arg)> body_fn);

  // For loop with predefined operation
  mlir::Value buildForWithOp(mlir::Value start, mlir::Value end,
                             mlir::Value step, mlir::Value init_value,
                             const std::string &operation);

  mlir::Value buildWhileWithOp(mlir::Value init, mlir::Value target,
                               const std::string &operation,
                               const std::string &condition);

  mlir::Value buildWhile(mlir::Value init,
                         std::function<mlir::Value(mlir::Value)> condition_fn,
                         std::function<mlir::Value(mlir::Value)> body_fn);

  // Function generation
  void createFunctionWithParamsSetup(
      const std::vector<std::pair<std::string, std::string>> &params);

  void finalizeFunctionWithParams(const std::string &name, mlir::Value result);

  // Get generated MLIR as string
  std::string getMLIRString();

  // Get generated LLVM IR as string
  std::string getLLVMIRString();

  mlir::Value getParameter(const std::string &name);

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

  mlir::arith::CmpIPredicate getIntegerPredicate(const std::string &pred) const;
  mlir::arith::CmpFPredicate getFloatPredicate(const std::string &pred) const;

  // Type promotion helpers
  std::pair<mlir::Value, mlir::Value> promoteTypes(mlir::Value lhs,
                                                   mlir::Value rhs);
  mlir::Type getPromotedType(mlir::Type lhs, mlir::Type rhs) const;

  std::unordered_map<std::string, mlir::Value> parameterMap;
};

} // namespace mlir_edsl