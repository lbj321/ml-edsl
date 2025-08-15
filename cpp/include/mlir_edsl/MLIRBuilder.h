#pragma once

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <memory>
#include <string>

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
    
    // Function generation
    void createFunction(const std::string& name, mlir::Value result);
    
    // Get generated MLIR as string
    std::string getMLIRString();
    
    // Get generated LLVM IR as string
    std::string getLLVMIRString();
    
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
    bool isIntegerType(mlir::Type type) const;
    bool isFloatType(mlir::Type type) const;
    
    // Type promotion helpers
    std::pair<mlir::Value, mlir::Value> promoteTypes(mlir::Value lhs, mlir::Value rhs);
    mlir::Type getPromotedType(mlir::Type lhs, mlir::Type rhs) const;
};

} // namespace mlir_edsl