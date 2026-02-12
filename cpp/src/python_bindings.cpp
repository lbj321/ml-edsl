#include "mlir_edsl/MLIRCompiler.h"
#include "ast.pb.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_mlir_backend, m) {
  m.doc() = "MLIR EDSL C++ Backend";

  py::class_<mlir_edsl::MLIRCompiler>(m, "MLIRCompiler")
      .def(py::init<>())
      .def(
          "compile_function",
          [](mlir_edsl::MLIRCompiler &self, const std::string &bytes) {
            mlir_edsl::FunctionDef funcDef;
            if (!funcDef.ParseFromString(bytes)) {
              throw std::runtime_error("Failed to parse FunctionDef protobuf");
            }
            self.compileFunction(funcDef);
          },
          py::arg("function_def_bytes"),
          "Compile function from protobuf FunctionDef bytes")
      .def("get_function_pointer", &mlir_edsl::MLIRCompiler::getFunctionPointer,
           py::arg("name"),
           "Get JIT-compiled function pointer (auto-finalizes on first call)")
      .def("has_function", &mlir_edsl::MLIRCompiler::hasFunction,
           py::arg("name"), "Check if function is already compiled")
      .def("list_functions", &mlir_edsl::MLIRCompiler::listFunctions,
           "List all compiled function names")
      .def("get_module_ir", &mlir_edsl::MLIRCompiler::getModuleIR,
           "Get current MLIR module IR as string")
      .def("clear", &mlir_edsl::MLIRCompiler::clear,
           "Reset to Building state, clear all compiled functions")
      .def("get_lowering_snapshots",
           &mlir_edsl::MLIRCompiler::getLoweringSnapshots,
           "Get IR snapshots after each lowering pass (populated when "
           "SAVE_IR is set)")
      .def(
          "set_optimization_level",
          [](mlir_edsl::MLIRCompiler &self, int level) {
            mlir_edsl::MLIRCompiler::OptLevel opt;
            if (level == 0)
              opt = mlir_edsl::MLIRCompiler::OptLevel::O0;
            else if (level == 2)
              opt = mlir_edsl::MLIRCompiler::OptLevel::O2;
            else if (level == 3)
              opt = mlir_edsl::MLIRCompiler::OptLevel::O3;
            else
              opt = mlir_edsl::MLIRCompiler::OptLevel::O2;
            self.setOptimizationLevel(opt);
          },
          "Set optimization level (0=O0, 2=O2, 3=O3)");
}
