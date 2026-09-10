//===- standalone-opt.cpp -------------------------------------------------===//
//
// A plain mlir-opt clone (all standard dialects/passes registered) plus
// LowerUnpackDirectPass, so run.sh can exercise
// --linalg-lower-unpack-direct alongside every other stage without needing
// to touch the real project's build.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Declared (not mlir::-namespaced) in LowerUnpackDirectPass.cpp.
std::unique_ptr<mlir::Pass> createLowerUnpackDirectPass();

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::registerPass(createLowerUnpackDirectPass);

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "standalone-opt\n", registry));
}
