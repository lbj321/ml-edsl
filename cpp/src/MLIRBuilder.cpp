#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRLowering.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>
#include <iostream>

namespace mlir_edsl {

MLIRBuilder::MLIRBuilder() {
    context = std::make_unique<mlir::MLIRContext>();
    
    context->getOrLoadDialect<mlir::arith::ArithDialect>();
    context->getOrLoadDialect<mlir::func::FuncDialect>();
    
    builder = std::make_unique<mlir::OpBuilder>(context.get());
}

MLIRBuilder::~MLIRBuilder() {
    if (currentFunction) {
        currentFunction = nullptr;
    }
    if (module) {
        module = nullptr;
    }
    builder.reset();
    context.reset();
}

void MLIRBuilder::initializeModule() {
    module = mlir::ModuleOp::create(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(module.getBody());

    auto funcType = builder->getFunctionType({}, {});
    currentFunction = builder->create<mlir::func::FuncOp>(
        builder->getUnknownLoc(), "temp_function", funcType);
    
    auto* entryBlock = currentFunction.addEntryBlock();
    builder->setInsertionPointToStart(entryBlock);
}

mlir::Value MLIRBuilder::buildConstant(int32_t value) {
    auto loc = builder->getUnknownLoc();
    auto type = getIntegerType();
    auto attr = builder->getI32IntegerAttr(value);

    return builder->create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value MLIRBuilder::buildConstant(float value) {
    auto loc = builder->getUnknownLoc();
    auto type = getFloatType();
    auto attr = builder->getF32FloatAttr(value);

    return builder->create<mlir::arith::ConstantOp>(loc, type, attr);
}

mlir::Value MLIRBuilder::buildAdd(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

    if (isIntegerType(promotedLhs.getType())) {
        return builder->create<mlir::arith::AddIOp>(loc, promotedLhs, promotedRhs);
    }

    return builder->create<mlir::arith::AddFOp>(loc, promotedLhs, promotedRhs);    
}

mlir::Value MLIRBuilder::buildSub(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

    if (isIntegerType(promotedLhs.getType())) {
        return builder->create<mlir::arith::SubIOp>(loc, promotedLhs, promotedRhs);
    }

    return builder->create<mlir::arith::SubFOp>(loc, promotedLhs, promotedRhs);    
}

mlir::Value MLIRBuilder::buildMul(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

    if (isIntegerType(promotedLhs.getType())) {
        return builder->create<mlir::arith::MulIOp>(loc, promotedLhs, promotedRhs);
    }

    return builder->create<mlir::arith::MulFOp>(loc, promotedLhs, promotedRhs);    
}

mlir::Value MLIRBuilder::buildDiv(mlir::Value lhs, mlir::Value rhs) {
    auto loc = builder->getUnknownLoc();
    auto [promotedLhs, promotedRhs] = promoteTypes(lhs, rhs);

    if (isIntegerType(promotedLhs.getType())) {
        return builder->create<mlir::arith::DivSIOp>(loc, promotedLhs, promotedRhs);
    }

    return builder->create<mlir::arith::DivFOp>(loc, promotedLhs, promotedRhs);    
}

mlir::Value MLIRBuilder::convertIntToFloat(mlir::Value intValue) {
    auto loc = builder->getUnknownLoc();
    auto floatType = getFloatType();
    return builder->create<mlir::arith::SIToFPOp>(loc, floatType, intValue);
}

void MLIRBuilder::createFunction(const std::string& name, mlir::Value result) {
    std::cerr << "MLIRBuilder: Creating function " << name << "\n";
    
    if (!currentFunction) {
        std::cerr << "Error: No current function!\n";
        return;
    }
    
    currentFunction.setName(name);
    currentFunction.setFunctionType(builder->getFunctionType({}, {result.getType()}));
    
    auto& entryBlock = currentFunction.front();
    builder->setInsertionPointToEnd(&entryBlock);
    builder->create<mlir::func::ReturnOp>(builder->getUnknownLoc(), result);
}

std::string MLIRBuilder::getMLIRString() {
    std::string result;
    llvm::raw_string_ostream stream(result);
    mlir::OpPrintingFlags flags;
    flags.enableDebugInfo(false);
    flags.printGenericOpForm(false);
    module.print(stream, flags);
    return result;
}

std::string MLIRBuilder::getLLVMIRString() {
    MLIRLowering lowering(context.get());
    return lowering.lowerToLLVMIR(module);
}

void MLIRBuilder::reset() {
    initializeModule();
}

mlir::Type MLIRBuilder::getIntegerType() const {
    return builder->getI32Type();
}

mlir::Type MLIRBuilder::getFloatType() const {
    return builder->getF32Type();
}

bool MLIRBuilder::isIntegerType(mlir::Type type) const {
    return mlir::isa<mlir::IntegerType>(type);
}

bool MLIRBuilder::isFloatType(mlir::Type type) const {
    return mlir::isa<mlir::FloatType>(type);
}

std::pair<mlir::Value, mlir::Value> MLIRBuilder::promoteTypes(mlir::Value lhs, mlir::Value rhs) {
    mlir::Type lhsType = lhs.getType();
    mlir::Type rhsType = rhs.getType();
    mlir::Type promotedType = getPromotedType(lhsType, rhsType);

    if (lhsType != promotedType && isIntegerType(lhsType)) {
        lhs = convertIntToFloat(lhs);
    }

    if (rhsType != promotedType && isIntegerType(rhsType)) {
        rhs = convertIntToFloat(rhs);
    }

    return {lhs, rhs};
}

mlir::Type MLIRBuilder::getPromotedType(mlir::Type lhs, mlir::Type rhs) const {
    if (isFloatType(lhs) || isFloatType(rhs)) {
        return getFloatType();
    }
    return getIntegerType();
}

} // namespace mlir_edsl