#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"

namespace mlir_edsl {

class MLIRBuilder;
class MLIRExecutor;
class TypeSpec;

class MLIRCompiler {
public:
    enum class State { Building, Finalized };
    enum class OptLevel { O0, O2, O3 };

    MLIRCompiler();
    ~MLIRCompiler();

    // ==================== COMPILATION (Building state only) ====================
    void compileFunction(const std::string& functionDefBytes);

    // ==================== EXECUTION ====================
    uintptr_t getFunctionPointer(const std::string& name);
    std::string getFunctionSignature(const std::string& name) const;

    // ==================== STATE MANAGEMENT ====================
    void clear();

    State getState() const { return state; }
    bool isFinalized() const { return state == State::Finalized; }

    // ==================== INSPECTION ====================
    bool hasFunction(const std::string& name) const;
    std::vector<std::string> listFunctions() const;

    // ==================== CONFIGURATION ====================
    void setOptimizationLevel(OptLevel level);

private:
    State state;
    OptLevel optimizationLevel;

    // ==================== OWNED INFRASTRUCTURE ====================
    std::unique_ptr<mlir::MLIRContext> mlirContext;
    std::unique_ptr<mlir::OpBuilder> opBuilder;
    mlir::OwningOpRef<mlir::ModuleOp> module;

    // ==================== OWNED COMPONENTS ====================
    std::unique_ptr<MLIRBuilder> builder;
    std::unique_ptr<MLIRExecutor> executor;

    // ==================== FUNCTION STATE ====================
    mlir::func::FuncOp currentFunction;
    std::unordered_map<std::string, mlir::Value> parameterMap;
    std::unordered_map<std::string, mlir::func::FuncOp> functionTable;
    std::unordered_set<std::string> compiledFunctions;

    // Signatures for ctypes (stored as serialized protobuf bytes)
    std::unordered_map<std::string, std::string> signatures;

    // ==================== INTERNAL METHODS ====================
    void ensureFinalized();
    void resetFunctionState();

    // Function building (moved from MLIRBuilder)
    void createFunction(
        const std::string& name,
        const std::vector<std::pair<std::string, mlir_edsl::TypeSpec>>& params,
        mlir::Type returnType);
    void finalizeFunction(const std::string& name, mlir::Value result);

    // Type helpers
    mlir::Type convertType(const mlir_edsl::TypeSpec& typeSpec) const;
    bool isValidParameterType(const mlir_edsl::TypeSpec& type) const;
    bool isValidReturnType(const mlir_edsl::TypeSpec& type) const;
};

} // namespace mlir_edsl
