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
           "Get IR snapshots after each lowering pass")
      .def("enable_snapshot_capture",
           &mlir_edsl::MLIRCompiler::enableSnapshotCapture,
           "Enable IR snapshot capture for lowering passes")
      .def("get_failure_ir", &mlir_edsl::MLIRCompiler::getFailureIR,
           "IR at lowering failure point; empty if no failure since last clear()")
      .def("inject_test_failure", &mlir_edsl::MLIRCompiler::injectTestFailure,
           "Testing only: injects a type-mismatched function to trigger verifier failure")
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
          "Set optimization level (0=O0, 2=O2, 3=O3)")
      .def(
          "set_target",
          [](mlir_edsl::MLIRCompiler &self, const std::string &target) {
            if (target == "gpu")
              self.setTarget(mlir_edsl::MLIRCompiler::Target::GPU);
            else
              self.setTarget(mlir_edsl::MLIRCompiler::Target::CPU);
          },
          py::arg("target"), "Set compilation target ('cpu' or 'gpu')")
      .def(
          "execute_gpu_function",
          [](mlir_edsl::MLIRCompiler &self, const std::string &name,
             py::list inputs, py::list output_shape, size_t element_size)
              -> py::bytes {
            // inputs: list of (data_ptr_int, shape_list) tuples
            std::vector<std::pair<const void *, std::vector<int64_t>>> cpp_inputs;
            for (auto item : inputs) {
              auto tup = item.cast<py::tuple>();
              auto ptr = reinterpret_cast<const void *>(tup[0].cast<uintptr_t>());
              auto shape_list = tup[1].cast<std::vector<int64_t>>();
              cpp_inputs.push_back({ptr, shape_list});
            }

            std::vector<int64_t> out_shape = output_shape.cast<std::vector<int64_t>>();
            size_t out_elems = 1;
            for (auto d : out_shape) out_elems *= (size_t)d;
            std::string out_buf(out_elems * element_size, '\0');

            {
              py::gil_scoped_release release;
              self.executeGPUFunction(name, cpp_inputs,
                                      out_buf.data(), out_shape, element_size);
            }
            return py::bytes(out_buf);
          },
          py::arg("name"), py::arg("inputs"), py::arg("output_shape"),
          py::arg("element_size"),
          "Execute GPU-compiled function; returns raw output bytes");
}
