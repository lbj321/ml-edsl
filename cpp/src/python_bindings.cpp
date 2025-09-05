#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mlir_edsl/MLIRBuilder.h"
#include "mlir_edsl/MLIRExecutor.h"
#include "mlir/IR/Value.h"

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
        .def("build_constant", 
             py::overload_cast<int32_t>(&mlir_edsl::MLIRBuilder::buildConstant),
             "Create integer constant")
        .def("build_constant", 
             py::overload_cast<float>(&mlir_edsl::MLIRBuilder::buildConstant),
             "Create float constant")
        .def("build_add", &mlir_edsl::MLIRBuilder::buildAdd,
             "Create addition operation")
        .def("build_sub", &mlir_edsl::MLIRBuilder::buildSub,
             "Create subtraction operation")
        .def("build_mul", &mlir_edsl::MLIRBuilder::buildMul,
             "Create multiplication operation")
        .def("build_div", &mlir_edsl::MLIRBuilder::buildDiv,
             "Create division operation")
        .def("build_compare", &mlir_edsl::MLIRBuilder::buildCompare,
             "Create comparison operation")
        .def("build_if", &mlir_edsl::MLIRBuilder::buildIf,
             "Create if-else conditional operation")
        .def("build_for_with_op", &mlir_edsl::MLIRBuilder::buildForWithOp,
             "Create for loop with predefined operation")
        .def("build_while_with_op", &mlir_edsl::MLIRBuilder::buildWhileWithOp,
             "Create while loop with predefined operation")
        .def("create_function_with_params_setup", &mlir_edsl::MLIRBuilder::createFunctionWithParamsSetup,
             "Set up function parameters without finalizing")
        .def("finalize_function_with_params", &mlir_edsl::MLIRBuilder::finalizeFunctionWithParams,
             "Finalize function with parameters")
        .def("get_parameter", &mlir_edsl::MLIRBuilder::getParameter,
             "Get parameter by name")
        .def("get_mlir_string", &mlir_edsl::MLIRBuilder::getMLIRString,
             "Get generated MLIR as string")
        .def("get_llvm_ir_string", &mlir_edsl::MLIRBuilder::getLLVMIRString,
             "Get generated LLVM IR as string")
        .def("reset", &mlir_edsl::MLIRBuilder::reset,
             "Reset builder for new function");
    
    py::class_<mlir_edsl::MLIRExecutor>(m, "MLIRExecutor")
        .def(py::init<>())
        .def("initialize", &mlir_edsl::MLIRExecutor::initialize,
             "Initialize the JIT execution engine")
        .def("compile_function", &mlir_edsl::MLIRExecutor::compileFunction,
             "Compile LLVM IR string to executable function",
             py::return_value_policy::reference)
        .def("call_int32_function", &mlir_edsl::MLIRExecutor::callInt32Function,
             "Execute compiled function returning int32",
             py::arg("funcPtr"), py::arg("intArgs") = std::vector<int32_t>(), py::arg("floatArgs") = std::vector<float>())
        .def("call_float_function", &mlir_edsl::MLIRExecutor::callFloatFunction,
             "Execute compiled function returning float",
             py::arg("funcPtr"), py::arg("intArgs") = std::vector<int32_t>(), py::arg("floatArgs") = std::vector<float>())
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