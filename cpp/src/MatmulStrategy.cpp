#include "mlir_edsl/MatmulStrategy.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
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
  strategy.packA = ov.packA;
  strategy.packB = ov.packB;
  strategy.vectorize = ov.vectorize;
  return strategy;
}

llvm::StringRef stringifyBlockedStage(BlockedStage stage) {
  switch (stage) {
  case BlockedStage::Distributed:
    return "distributed";
  case BlockedStage::Tiled:
    return "tiled";
  case BlockedStage::Kernel:
    return "kernel";
  }
  llvm_unreachable("unknown BlockedStage");
}

std::optional<BlockedStage> symbolizeBlockedStage(llvm::StringRef str) {
  for (BlockedStage stage : {BlockedStage::Distributed, BlockedStage::Tiled,
                             BlockedStage::Kernel})
    if (str == stringifyBlockedStage(stage))
      return stage;
  return std::nullopt;
}

mlir::DictionaryAttr buildBlockedConfig(mlir::MLIRContext *ctx,
                                        const MatmulStrategy &s,
                                        BlockedStage stage) {
  mlir::Builder b(ctx);
  return b.getDictionaryAttr({
      b.getNamedAttr("mr", b.getI64IntegerAttr(s.mr)),
      b.getNamedAttr("nr", b.getI64IntegerAttr(s.nr)),
      b.getNamedAttr("mc", b.getI64IntegerAttr(s.mc)),
      b.getNamedAttr("nc", b.getI64IntegerAttr(s.nc)),
      b.getNamedAttr("kc", b.getI64IntegerAttr(s.kc)),
      b.getNamedAttr("pack_a", b.getBoolAttr(s.packA)),
      b.getNamedAttr("pack_b", b.getBoolAttr(s.packB)),
      b.getNamedAttr("vectorize", b.getBoolAttr(s.vectorize)),
      b.getNamedAttr("stage", b.getStringAttr(stringifyBlockedStage(stage))),
  });
}

mlir::DictionaryAttr withStage(mlir::DictionaryAttr config,
                               BlockedStage stage) {
  mlir::NamedAttrList attrs(config);
  attrs.set("stage", mlir::StringAttr::get(config.getContext(),
                                           stringifyBlockedStage(stage)));
  return attrs.getDictionary(config.getContext());
}

mlir::FailureOr<BlockedConfig> readBlockedConfig(mlir::Operation *op) {
  auto config = op->getAttrOfType<mlir::DictionaryAttr>(kBlockedAttrName);
  if (!config)
    return mlir::failure();

  auto getInt = [&](llvm::StringRef name, int64_t &out) {
    auto attr = config.getAs<mlir::IntegerAttr>(name);
    if (attr)
      out = attr.getInt();
    return static_cast<bool>(attr);
  };
  auto getBool = [&](llvm::StringRef name, bool &out) {
    auto attr = config.getAs<mlir::BoolAttr>(name);
    if (attr)
      out = attr.getValue();
    return static_cast<bool>(attr);
  };

  BlockedConfig result;
  MatmulStrategy &s = result.strategy;
  if (!getInt("mr", s.mr) || !getInt("nr", s.nr) || !getInt("mc", s.mc) ||
      !getInt("nc", s.nc) || !getInt("kc", s.kc) ||
      !getBool("pack_a", s.packA) || !getBool("pack_b", s.packB) ||
      !getBool("vectorize", s.vectorize))
    return mlir::failure();

  auto stageAttr = config.getAs<mlir::StringAttr>("stage");
  if (!stageAttr)
    return mlir::failure();
  auto stage = symbolizeBlockedStage(stageAttr.getValue());
  if (!stage)
    return mlir::failure();
  result.stage = *stage;
  return result;
}

bool isBlockedWithoutVectorize(mlir::Operation *op) {
  auto config = readBlockedConfig(op);
  return mlir::succeeded(config) && !config->strategy.vectorize;
}

mlir::LogicalResult collectBlockedTilesAtStage(
    mlir::Operation *root, BlockedStage stage,
    llvm::SmallVectorImpl<std::pair<mlir::linalg::MatmulOp, MatmulStrategy>>
        &tiles) {
  auto result = root->walk([&](mlir::linalg::MatmulOp op) {
    if (!op->hasAttr(kBlockedAttrName))
      return mlir::WalkResult::advance();
    auto config = readBlockedConfig(op);
    if (mlir::failed(config)) {
      op->emitError("malformed ") << kBlockedAttrName;
      return mlir::WalkResult::interrupt();
    }
    if (config->stage == stage)
      tiles.emplace_back(op, config->strategy);
    return mlir::WalkResult::advance();
  });
  return mlir::failure(result.wasInterrupted());
}

void setBlockedStage(mlir::Operation *tile, BlockedStage stage) {
  auto config = tile->getAttrOfType<mlir::DictionaryAttr>(kBlockedAttrName);
  tile->setAttr(kBlockedAttrName, withStage(config, stage));
}

} // namespace mlir_edsl
