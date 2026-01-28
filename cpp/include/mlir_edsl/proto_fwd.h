#pragma once

namespace mlir_edsl {

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
class FunctionDef;

// Type specifications
class TypeSpec;
class ScalarTypeSpec;
class MemRefTypeSpec;

// Scalar operations
class Constant;
class BinaryOp;
class CompareOp;
class CastOp;

// Control flow operations
class IfOp;
class ForLoopOp;
class WhileLoopOp;

// Function operations
class Parameter;
class CallOp;

// Binding operations
class LetBinding;
class ValueReference;

} // namespace mlir_edsl
