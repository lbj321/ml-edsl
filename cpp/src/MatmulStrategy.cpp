#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/IR/BuiltinTypes.h"

#include <algorithm>
#include <cstdlib>
#include <thread>

namespace mlir_edsl {

namespace {

// Number of f32 lanes in one target vector register (AVX2 ymm). Used both for
// the register-budget arithmetic and as the granularity the k cache block must
// respect.
constexpr int64_t kVectorWidth = 8;

// Total ymm registers available to the microkernel.
constexpr int64_t kNumVectorRegisters = 16;

/// Largest b <= min(cap, dim) such that b % multiple == 0 and dim % b == 0.
///
/// This is what makes the cache blocking follow the microkernel rather than
/// being hardcoded: raising mr from 4 to 6 moves MC from 128 to 96 on M=768
/// with no other input. It also subsumes a separate divisibility check —
/// "there is no remainder at any loop level" is exactly "this search
/// succeeded for all three dimensions".
///
/// Linear scan downward: `dim` is a matmul extent (thousands at most) and this
/// runs once per matmul at compile time, so there is no reason to be clever.
mlir::FailureOr<int64_t> chooseBlock(int64_t dim, int64_t cap,
                                     int64_t multiple) {
  if (dim <= 0 || cap <= 0 || multiple <= 0)
    return mlir::failure();

  for (int64_t b = std::min(cap, dim); b >= multiple; --b) {
    if (b % multiple == 0 && dim % b == 0)
      return b;
  }
  return mlir::failure();
}

/// True if an MR x NR f32 register tile fits in the vector register file:
/// MR*NR/8 accumulators, NR/8 registers holding the current B row, and one
/// register for the broadcast A element.
bool fitsRegisterBudget(int64_t mr, int64_t nr) {
  if (mr < 1 || nr < kVectorWidth || nr % kVectorWidth != 0)
    return false;
  int64_t used = (mr * nr) / kVectorWidth + nr / kVectorWidth + 1;
  return used <= kNumVectorRegisters;
}

/// Threads to size the jc bands for.
///
/// OMP_NUM_THREADS first, because that is what libomp actually honours at run
/// time; hardware concurrency otherwise. Read when the function is compiled,
/// so raising OMP_NUM_THREADS afterwards leaves the extra threads with no
/// band to take — set it before the first call.
int64_t detectNumThreads() {
  if (const char *env = std::getenv("OMP_NUM_THREADS")) {
    char *end = nullptr;
    long parsed = std::strtol(env, &end, 10);
    if (end != env && parsed > 0)
      return static_cast<int64_t>(parsed);
  }
  unsigned hw = std::thread::hardware_concurrency();
  return hw > 0 ? static_cast<int64_t>(hw) : 1;
}

/// Static f32 ranked tensor, or failure. Matches the operand constraints
/// LinalgMatmulToContractPass applies further down the pipeline, so a matmul
/// this accepts is one that pass can also turn into a vector.contract.
mlir::FailureOr<mlir::ArrayRef<int64_t>> staticF32Shape(mlir::Value v) {
  auto tensorType = llvm::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!tensorType || !tensorType.hasStaticShape())
    return mlir::failure();
  if (!tensorType.getElementType().isF32())
    return mlir::failure();
  return tensorType.getShape();
}

} // namespace

mlir::FailureOr<MatmulStrategy> chooseStrategy(mlir::linalg::MatmulOp op,
                                               const StrategyOverrides &ov) {
  // Already blocked by a previous run of the pass.
  if (op->hasAttr(kBlockedAttrName))
    return mlir::failure();

  if (op.getInputs().size() != 2 || op.getOutputs().size() != 1)
    return mlir::failure();

  auto aShape = staticF32Shape(op.getInputs()[0]);
  auto bShape = staticF32Shape(op.getInputs()[1]);
  auto cShape = staticF32Shape(op.getOutputs()[0]);
  if (mlir::failed(aShape) || mlir::failed(bShape) || mlir::failed(cShape))
    return mlir::failure();

  if (aShape->size() != 2 || bShape->size() != 2 || cShape->size() != 2)
    return mlir::failure();

  const int64_t m = (*aShape)[0];
  const int64_t k = (*aShape)[1];
  const int64_t n = (*bShape)[1];
  if ((*bShape)[0] != k || (*cShape)[0] != m || (*cShape)[1] != n)
    return mlir::failure();

  // An epilogue-fused matmul (matmul → bias_add → relu) stays on
  // LinalgOuterTileAndFusePass, which fuses the whole producer chain onto one
  // tile. Blocking it here would tile the matmul away from its consumers and
  // lose that fusion; applying the epilogue to the MR x NR accumulator is
  // Step 4 work.
  if (op->getNumResults() != 1)
    return mlir::failure();
  for (mlir::Operation *user : op->getResult(0).getUsers()) {
    if (llvm::isa<mlir::linalg::LinalgOp>(user))
      return mlir::failure();
  }

  if (!fitsRegisterBudget(ov.mr, ov.nr))
    return mlir::failure();

  auto mc = chooseBlock(m, ov.mcTarget, ov.mr);
  auto nc = chooseBlock(n, ov.ncTarget, ov.nr);
  auto kc = chooseBlock(k, ov.kcTarget, kVectorWidth);
  if (mlir::failed(mc) || mlir::failed(nc) || mlir::failed(kc))
    return mlir::failure();

  MatmulStrategy strategy;
  strategy.mr = ov.mr;
  strategy.nr = ov.nr;
  strategy.mc = *mc;
  strategy.nc = *nc;
  strategy.kc = *kc;
  strategy.vectorize = ov.vectorize;
  return strategy;
}

} // namespace mlir_edsl
