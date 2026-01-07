#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11_protobuf/native_proto_caster.h"
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRExecutor.h"
#include "mlir/IR/Value.h"
#include "ast.pb.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(mlir::Value);

PYBIND11_MODULE(_mlir_backend, m) {
    m.doc() = "MLIR EDSL C++ Backend";
    
    py::class_<mlir::Value>(m, "Value")
        .def("__bool__", [](const mlir::Value &v) { return static_cast<bool>(v); })
        .def("__repr__", [](const mlir::Value &v) { 
            return "<mlir.Value>"; 
        });
    
    py::class_<mlir_edsl::MLIRBuilder>(m, "MLIRBuilder")
        .def(py::init<>())
        .def("initialize_module", &mlir_edsl::MLIRBuilder::initializeModule,
             "Initialize a new MLIR module")

        // ==================== CORE COMPILATION ====================
        .def("compile_function", &mlir_edsl::MLIRBuilder::compileFunctionFromDef,
             py::arg("func_def"),
             "Compile complete function from FunctionDef protobuf")

        // ==================== INSPECTION ====================
        .def("get_mlir_string", &mlir_edsl::MLIRBuilder::getMLIRString,
             "Get generated MLIR IR as string")
        .def("get_llvm_ir_string", &mlir_edsl::MLIRBuilder::getLLVMIRString,
             "Get generated LLVM IR as string")

        // ==================== MANAGEMENT ====================
        .def("has_function", &mlir_edsl::MLIRBuilder::hasFunction,
             "Check if function is already compiled")
        .def("list_functions", &mlir_edsl::MLIRBuilder::listFunctions,
             "List all compiled function names")
        .def("clear_module", &mlir_edsl::MLIRBuilder::clearModule,
             "Clear all compiled functions");

    py::class_<mlir_edsl::MLIRExecutor>(m, "MLIRExecutor")
        .def(py::init<>())
        .def("initialize", &mlir_edsl::MLIRExecutor::initialize,
             "Initialize the JIT execution engine")
        .def("compile_function", &mlir_edsl::MLIRExecutor::compileFunction,
             "Compile LLVM IR string to executable function",
             py::return_value_policy::reference)
        .def("register_function_signature", &mlir_edsl::MLIRExecutor::registerFunctionSignature,
             py::arg("signature"),
             "Register function signature from FunctionSignature protobuf")
        .def("get_function_pointer", &mlir_edsl::MLIRExecutor::getFunctionPointer,
             py::arg("name"),
             "Get JIT-compiled function pointer as integer for ctypes")
        .def("get_function_signature", [](mlir_edsl::MLIRExecutor& self, const std::string& name) {
            std::string result = self.getFunctionSignature(name);
            return py::bytes(result);  // Return as bytes, not str
        },
             py::arg("name"),
             "Get function signature as FunctionSignature protobuf")
        .def("is_initialized", &mlir_edsl::MLIRExecutor::isInitialized,
             "Check if executor is initialized")
        .def("get_last_error", &mlir_edsl::MLIRExecutor::getLastError,
             "Get last error message")
        .def("clear", &mlir_edsl::MLIRExecutor::clear,
             "Clear JIT engine")
        .def("set_optimization_level", [](mlir_edsl::MLIRExecutor& self, int level) {
            mlir_edsl::MLIRExecutor::OptLevel opt;
            if (level == 0) opt = mlir_edsl::MLIRExecutor::OptLevel::O0;
            else if (level == 2) opt = mlir_edsl::MLIRExecutor::OptLevel::O2;
            else if (level == 3) opt = mlir_edsl::MLIRExecutor::OptLevel::O3;
            else opt = mlir_edsl::MLIRExecutor::OptLevel::O2; // default
            self.setOptimizationLevel(opt);
        }, "Set optimization level (0=O0, 2=O2, 3=O3)");
}