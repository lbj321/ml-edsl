#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/ArithBuilder.h"
#include "mlir_edsl/SCFBuilder.h"
#include "mlir_edsl/MemRefBuilder.h"
#include "mlir_edsl/TensorBuilder.h"
#include "mlir_edsl/LinalgBuilder.h"

#include "ast.pb.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

namespace mlir_edsl {

MLIRBuilder::MLIRBuilder(mlir::MLIRContext *context, mlir::OpBuilder *builder)
    : context(context), builder(builder) {
  // Initialize dialect builders
  arithBuilder = std::make_unique<mlir_edsl::ArithBuilder>(*builder);
  scfBuilder = std::make_unique<mlir_edsl::SCFBuilder>(*builder);
  memrefBuilder = std::make_unique<mlir_edsl::MemRefBuilder>(
      *builder, context, this, arithBuilder.get(), scfBuilder.get());
  tensorBuilder = std::make_unique<mlir_edsl::TensorBuilder>(
      *builder, context, this);
  linalgBuilder = std::make_unique<mlir_edsl::LinalgBuilder>(
      *builder, context, this);
}

MLIRBuilder::~MLIRBuilder() = default;

// ==================== DEPENDENCY INJECTION ====================

void MLIRBuilder::setParameterMap(
    std::unordered_map<std::string, mlir::Value> *paramMap) {
  parameterMap = paramMap;
}

void MLIRBuilder::setFunctionTable(
    std::unordered_map<std::string, mlir::func::FuncOp> *funcTable) {
  functionTable = funcTable;
}

void MLIRBuilder::clearValueCache() {
  valueCache.clear();
}

void MLIRBuilder::setValueCacheEntry(int64_t nodeId, mlir::Value value) {
  valueCache[nodeId] = value;
}

// ==================== AST DISPATCH ====================

mlir::Value MLIRBuilder::buildFromProtobufNode(const mlir_edsl::ASTNode &node,
                                               mlir::Value outParam) {
  switch (node.node_case()) {
    case mlir_edsl::ASTNode::kScalar:
      return buildFromScalarNode(node.scalar());
    case mlir_edsl::ASTNode::kArray:
      return buildFromArrayNode(node.array(), outParam);
    case mlir_edsl::ASTNode::kControlFlow:
      return buildFromControlFlowNode(node.control_flow());
    case mlir_edsl::ASTNode::kFunction:
      return buildFromFunctionNode(node.function());
    case mlir_edsl::ASTNode::kTensor:
      return buildFromTensorNode(node.tensor());
    case mlir_edsl::ASTNode::kLinalg:
      return buildFromLinalgNode(node.linalg(), outParam);
    case mlir_edsl::ASTNode::kBinding:
      return buildFromBindingNode(node.binding());
    default:
      throw std::runtime_error("Unknown AST node category");
  }
}

mlir::Value MLIRBuilder::buildFromScalarNode(const mlir_edsl::ScalarNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::ScalarNode::kConstant:
      return handleConstant(node.constant());
    case mlir_edsl::ScalarNode::kBinaryOp:
      return handleBinaryOp(node.binary_op());
    case mlir_edsl::ScalarNode::kCompareOp:
      return handleCompareOp(node.compare_op());
    case mlir_edsl::ScalarNode::kCastOp:
      return handleCastOp(node.cast_op());
    default:
      throw std::runtime_error("Unknown scalar node type");
  }
}

mlir::Value MLIRBuilder::buildFromArrayNode(const mlir_edsl::ArrayNode &node,
                                            mlir::Value outParam) {
  switch (node.value_case()) {
    case mlir_edsl::ArrayNode::kLiteral:
      return memrefBuilder->buildArrayLiteral(node.literal(), outParam);
    case mlir_edsl::ArrayNode::kAccess:
      return memrefBuilder->buildArrayAccess(node.access());
    case mlir_edsl::ArrayNode::kStore:
      return memrefBuilder->buildArrayStore(node.store());
    case mlir_edsl::ArrayNode::kBinaryOp:
      return memrefBuilder->buildArrayBinaryOp(node.binary_op(), outParam);
    default:
      throw std::runtime_error("Unknown array node type");
  }
}

mlir::Value MLIRBuilder::buildFromTensorNode(const mlir_edsl::TensorNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::TensorNode::kFromElements:
      return tensorBuilder->buildFromElements(node.from_elements());
    case mlir_edsl::TensorNode::kExtract:
      return tensorBuilder->buildExtract(node.extract());
    case mlir_edsl::TensorNode::kInsert:
      return tensorBuilder->buildInsert(node.insert());
    case mlir_edsl::TensorNode::kEmpty:
      return tensorBuilder->buildEmpty(node.empty());
    default:
      throw std::runtime_error("Unknown tensor node type");
  }
}

mlir::Value MLIRBuilder::buildFromLinalgNode(const mlir_edsl::LinalgNode &node,
                                             mlir::Value outParam) {
  switch (node.value_case()) {
    case mlir_edsl::LinalgNode::kDot:
      return linalgBuilder->buildDot(node.dot());
    case mlir_edsl::LinalgNode::kMatmul:
      return linalgBuilder->buildMatmul(node.matmul(), outParam);
    case mlir_edsl::LinalgNode::kMap:
      return linalgBuilder->buildMap(node.map(), outParam);
    case mlir_edsl::LinalgNode::kMapElement:
      return handleLinalgMapElement(node.map_element());
    case mlir_edsl::LinalgNode::kReduce:
      return linalgBuilder->buildReduce(node.reduce());
    case mlir_edsl::LinalgNode::kBinaryOp:
      return linalgBuilder->buildBinaryOp(node.binary_op(), outParam);
    case mlir_edsl::LinalgNode::kReduceElement:
      return handleLinalgPlaceholder(node.reduce_element().node_id(),
                                     "LinalgReduceElement");
    case mlir_edsl::LinalgNode::kReduceAccum:
      return handleLinalgPlaceholder(node.reduce_accum().node_id(),
                                     "LinalgReduceAccumulator");
    default:
      throw std::runtime_error("Unknown linalg node type");
  }
}

mlir::Value MLIRBuilder::buildFromControlFlowNode(const mlir_edsl::ControlFlowNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::ControlFlowNode::kIfOp:
      return handleIfOp(node.if_op());
    case mlir_edsl::ControlFlowNode::kForLoop:
      return handleForLoopOp(node.for_loop());
    case mlir_edsl::ControlFlowNode::kForIndex:
      return handleForIndex(node.for_index());
    case mlir_edsl::ControlFlowNode::kForIterArg:
      return handleForIterArg(node.for_iter_arg());
    default:
      throw std::runtime_error("Unknown control flow node type");
  }
}

mlir::Value MLIRBuilder::buildFromFunctionNode(const mlir_edsl::FunctionNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::FunctionNode::kParameter:
      return handleParameter(node.parameter());
    case mlir_edsl::FunctionNode::kCall:
      return handleCallOp(node.call());
    default:
      throw std::runtime_error("Unknown function node type");
  }
}

mlir::Value MLIRBuilder::buildFromBindingNode(const mlir_edsl::BindingNode &node) {
  switch (node.value_case()) {
    case mlir_edsl::BindingNode::kLet:
      return handleLetBinding(node.let());
    case mlir_edsl::BindingNode::kRef:
      return handleValueRef(node.ref());
    default:
      throw std::runtime_error("Unknown binding node type");
  }
}

// ==================== NODE HANDLERS ====================

// Binding handlers
mlir::Value MLIRBuilder::handleLetBinding(const mlir_edsl::LetBinding &binding) {
  int64_t nodeId = binding.node_id();
  mlir::Value result = buildFromProtobufNode(binding.value());
  valueCache[nodeId] = result;
  return result;
}

mlir::Value MLIRBuilder::handleValueRef(const mlir_edsl::ValueReference &ref) {
  int64_t nodeId = ref.node_id();
  auto it = valueCache.find(nodeId);
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("Reference to unbound value ID: " + std::to_string(nodeId));
}

// Scalar handlers
mlir::Value MLIRBuilder::handleConstant(const mlir_edsl::Constant &constant) {
  const auto &typeSpec = constant.type();

  if (!typeSpec.has_scalar()) {
    throw std::runtime_error("Constant must have scalar type");
  }

  switch (typeSpec.scalar().kind()) {
  case mlir_edsl::ScalarTypeSpec::I32:
    return arithBuilder->buildConstant(constant.int_value());
  case mlir_edsl::ScalarTypeSpec::F32:
    return arithBuilder->buildConstant(constant.float_value());
  case mlir_edsl::ScalarTypeSpec::I1:
    return arithBuilder->buildConstant(constant.bool_value());
  case mlir_edsl::ScalarTypeSpec::INDEX:
    return arithBuilder->buildIndexConstant(constant.int_value());
  default:
    throw std::runtime_error("Unsupported constant type");
  }
}

mlir::Value MLIRBuilder::handleBinaryOp(const mlir_edsl::BinaryOp &binop) {
  mlir::Value left = buildFromProtobufNode(binop.left());
  mlir::Value right = buildFromProtobufNode(binop.right());

  mlir::Type targetType = convertType(binop.result_type());
  auto [promotedLeft, promotedRight] = promoteToMatchDataType(left, right, targetType);

  switch (binop.op_type()) {
  case mlir_edsl::BinaryOpType::ADD:
    return arithBuilder->buildAdd(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::SUB:
    return arithBuilder->buildSub(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::MUL:
    return arithBuilder->buildMul(promotedLeft, promotedRight);
  case mlir_edsl::BinaryOpType::DIV:
    return arithBuilder->buildDiv(promotedLeft, promotedRight);
  default:
    throw std::runtime_error("Unsupported binary operation");
  }
}

mlir::Value MLIRBuilder::handleCompareOp(const mlir_edsl::CompareOp &cmp) {
  mlir::Value left = buildFromProtobufNode(cmp.left());
  mlir::Value right = buildFromProtobufNode(cmp.right());

  mlir::Type targetType = convertType(cmp.operand_type());
  auto [promotedLhs, promotedRhs] = promoteToMatchDataType(left, right, targetType);

  return arithBuilder->buildCompare(cmp.predicate(), promotedLhs, promotedRhs);
}

mlir::Value MLIRBuilder::handleCastOp(const mlir_edsl::CastOp &cast) {
  mlir::Value sourceValue = buildFromProtobufNode(cast.value());
  mlir::Type targetType = convertType(cast.target_type());
  return arithBuilder->buildCast(sourceValue, targetType);
}

// Control flow handlers
mlir::Value MLIRBuilder::handleIfOp(const mlir_edsl::IfOp &ifop) {
  mlir::Value cond = buildFromProtobufNode(ifop.condition());
  mlir::Type resultType = convertType(ifop.result_type());

  auto buildThen = [this, &ifop]() {
    return buildFromProtobufNode(ifop.then_value());
  };
  auto buildElse = [this, &ifop]() {
    return buildFromProtobufNode(ifop.else_value());
  };

  return scfBuilder->buildIf(cond, buildThen, buildElse, resultType);
}

// For loop handlers
mlir::Value MLIRBuilder::handleForLoopOp(const mlir_edsl::ForLoopOp &op) {
  // Build bounds and cast to index type (scf.for requires index)
  mlir::Value start = castToIndexType(buildFromProtobufNode(op.start()));
  mlir::Value end = castToIndexType(buildFromProtobufNode(op.end()));
  mlir::Value step = castToIndexType(buildFromProtobufNode(op.step()));
  mlir::Value initValue = buildFromProtobufNode(op.init_value());

  const auto &bodyProto = op.body();
  int64_t indexId = op.index_node_id();
  int64_t iterArgId = op.iter_arg_node_id();

  auto results = scfBuilder->buildFor(
      start, end, step, mlir::ValueRange{initValue},
      [this, &bodyProto, indexId, iterArgId](mlir::Value iv, mlir::ValueRange iterArgs)
          -> llvm::SmallVector<mlir::Value> {
        auto loc = builder->getUnknownLoc();
        mlir::Value ivI32 = builder->create<mlir::arith::IndexCastOp>(
            loc, mlir::IntegerType::get(context, 32), iv);

        valueCache[indexId] = ivI32;
        valueCache[iterArgId] = iterArgs[0];

        return {buildFromProtobufNode(bodyProto)};
      });
  return results[0];
}

mlir::Value MLIRBuilder::handleForIndex(const mlir_edsl::ForIndex &node) {
  auto it = valueCache.find(node.node_id());
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("ForIndex: no value in cache for node_id " +
                           std::to_string(node.node_id()));
}

mlir::Value MLIRBuilder::handleForIterArg(const mlir_edsl::ForIterArg &node) {
  auto it = valueCache.find(node.node_id());
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("ForIterArg: no value in cache for node_id " +
                           std::to_string(node.node_id()));
}

mlir::Value MLIRBuilder::handleLinalgMapElement(const mlir_edsl::LinalgMapElement &node) {
  auto it = valueCache.find(node.node_id());
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error("LinalgMapElement: no value in cache for node_id " +
                           std::to_string(node.node_id()));
}

mlir::Value MLIRBuilder::handleLinalgPlaceholder(int64_t nodeId,
                                                  const char *name) {
  auto it = valueCache.find(nodeId);
  if (it != valueCache.end()) {
    return it->second;
  }
  throw std::runtime_error(std::string(name) + ": no value in cache for node_id " +
                           std::to_string(nodeId));
}

// Function node handlers
mlir::Value MLIRBuilder::handleParameter(const mlir_edsl::Parameter &param) {
  return getParameter(param.name());
}

mlir::Value MLIRBuilder::handleCallOp(const mlir_edsl::CallOp &call) {
  std::vector<mlir::Value> args;
  for (const auto &arg : call.args()) {
    args.push_back(buildFromProtobufNode(arg));
  }
  return callFunction(call.func_name(), args);
}

// ==================== INTERNAL HELPERS ====================

mlir::Value MLIRBuilder::getParameter(const std::string &name) {
  auto it = parameterMap->find(name);
  if (it != parameterMap->end()) {
    return it->second;
  }
  throw std::runtime_error("Parameter not found: " + name);
}

mlir::Value MLIRBuilder::callFunction(const std::string &name,
                                      const std::vector<mlir::Value> &args) {
  auto it = functionTable->find(name);
  if (it == functionTable->end()) {
    throw std::runtime_error("Function not found: " + name);
  }
  auto funcOp = it->second;
  return builder
      ->create<mlir::func::CallOp>(builder->getUnknownLoc(), funcOp, args)
      .getResult(0);
}

bool MLIRBuilder::isIntegerType(mlir::Type type) const {
  return mlir::isa<mlir::IntegerType>(type);
}

bool MLIRBuilder::isFloatType(mlir::Type type) const {
  return mlir::isa<mlir::FloatType>(type);
}

std::pair<mlir::Value, mlir::Value>
MLIRBuilder::promoteToMatchDataType(mlir::Value lhs, mlir::Value rhs,
                                    mlir::Type targetType) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();

  if (lhsType != targetType && isIntegerType(lhsType) &&
      isFloatType(targetType)) {
    lhs = arithBuilder->buildCast(lhs, targetType);
  }

  if (rhsType != targetType && isIntegerType(rhsType) &&
      isFloatType(targetType)) {
    rhs = arithBuilder->buildCast(rhs, targetType);
  }

  return {lhs, rhs};
}

// ==================== TYPE SYSTEM ====================

mlir::Type MLIRBuilder::convertType(const mlir_edsl::TypeSpec &typeSpec) const {
  switch (typeSpec.type_kind_case()) {
    case mlir_edsl::TypeSpec::kScalar:
      return convertScalarType(typeSpec.scalar());
    case mlir_edsl::TypeSpec::kMemref:
      return convertMemRefType(typeSpec.memref());
    case mlir_edsl::TypeSpec::kTensor:
      return convertTensorType(typeSpec.tensor());
    case mlir_edsl::TypeSpec::TYPE_KIND_NOT_SET:
      throw std::runtime_error("TypeSpec has no type set");
    default:
      throw std::runtime_error("Unknown TypeSpec kind: " +
                               std::to_string(typeSpec.type_kind_case()));
  }
}

mlir::Type MLIRBuilder::convertScalarType(
    const mlir_edsl::ScalarTypeSpec &scalarSpec) const {
  switch (scalarSpec.kind()) {
    case mlir_edsl::ScalarTypeSpec::I32:
      return mlir::IntegerType::get(context, 32);
    case mlir_edsl::ScalarTypeSpec::F32:
      return mlir::Float32Type::get(context);
    case mlir_edsl::ScalarTypeSpec::I1:
      return mlir::IntegerType::get(context, 1);
    default:
      throw std::runtime_error("Unknown ScalarTypeSpec kind: " +
                               std::to_string(static_cast<int>(scalarSpec.kind())));
  }
}

mlir::Type MLIRBuilder::convertMemRefType(
    const mlir_edsl::MemRefTypeSpec &memrefSpec) const {
  mlir::Type elementType = convertType(memrefSpec.element_type());

  llvm::SmallVector<int64_t> shape(
      memrefSpec.shape().begin(),
      memrefSpec.shape().end()
  );

  // Dynamic dims are not supported: shapes are fully resolved on the Python
  // frontend before protobuf serialization, and the memref pipeline has no
  // support for dynamic dimensions at function boundaries.
  for (auto d : shape) {
    if (d == kProtoDynamicDim)
      throw std::runtime_error("Dynamic memref dimensions not supported");
  }

  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("Only 1D, 2D, and 3D arrays supported, got " +
                             std::to_string(shape.size()) + "D");
  }

  return mlir::MemRefType::get(shape, elementType);
}

mlir::Type MLIRBuilder::convertTensorType(
    const mlir_edsl::TensorTypeSpec &tensorSpec) const {
  mlir::Type elementType = convertType(tensorSpec.element_type());

  llvm::SmallVector<int64_t> shape(
      tensorSpec.shape().begin(),
      tensorSpec.shape().end()
  );

  // Dynamic dims are not supported: shapes are fully resolved on the Python
  // frontend before protobuf serialization, and the memref pipeline has no
  // support for dynamic dimensions at function boundaries.
  for (auto d : shape) {
    if (d == kProtoDynamicDim)
      throw std::runtime_error("Dynamic tensor dimensions not supported");
  }

  if (shape.empty() || shape.size() > 3) {
    throw std::runtime_error("Only 1D, 2D, and 3D tensors supported, got " +
                             std::to_string(shape.size()) + "D");
  }

  return mlir::RankedTensorType::get(shape, elementType);
}

// ==================== INFRASTRUCTURE UTILITIES ====================

mlir::Value MLIRBuilder::buildIndexConstant(int64_t value) {
  auto loc = builder->getUnknownLoc();
  return builder->create<mlir::arith::ConstantIndexOp>(loc, value);
}

mlir::Value MLIRBuilder::castToIndexType(mlir::Value value) {
  if (value.getType().isIndex()) {
    return value;
  }

  if (mlir::isa<mlir::IntegerType>(value.getType())) {
    auto loc = builder->getUnknownLoc();
    return builder->create<mlir::arith::IndexCastOp>(
        loc, builder->getIndexType(), value);
  }

  throw std::runtime_error("Cannot cast to index type: unsupported source type");
}

} // namespace mlir_edsl
