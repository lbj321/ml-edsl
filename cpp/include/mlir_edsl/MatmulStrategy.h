#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir_edsl {

/// Discardable attribute set by the blocked matmul passes on every
/// linalg.matmul tile they produce. Its value is a dictionary of the strategy
/// that produced the tile and its stage (see buildBlockedConfig), so IR dumps
/// show which blocking was used and each pass can pick up the previous one's
/// tiles.
///
/// The blocked passes leave MR x NR x 1 tiles behind that the older matmul
/// passes would happily re-tile: LinalgOuterTileAndFusePass would wrap one in
/// another scf.forall, the 8x8x8 LinalgMatmulTilingPass would split a 4x16
/// tile into 4x8 halves, and the 64x64 parallel tiling would grab them too.
/// Those three passes skip any op carrying this attribute; the passes that
/// *build* the microkernel (LinalgMatmulToContractPass,
/// LinalgVectorizationPass) deliberately do not, unless the strategy turned
/// vectorization off (see isBlockedWithoutVectorize).
constexpr llvm::StringLiteral kBlockedAttrName = "mlir_edsl.blocked";

/// A complete blocking decision for one linalg.matmul: the register tile (the
/// microkernel), the cache blocks derived from it, and the feature toggles.
///
/// Produced by chooseStrategy(), never constructed directly by the pass — the
/// cache blocks are only meaningful once validated against the op's shape.
struct MatmulStrategy {
  /// Register tile: the MR x NR accumulator held in vector registers across
  /// the innermost k-loop. This is "the microkernel".
  int64_t mr = 4;
  int64_t nr = 16;

  /// Cache blocks, resolved by chooseStrategy from the register tile and the
  /// op's shape. Always satisfy mc % mr == 0, nc % nr == 0, kc % 8 == 0, and
  /// divide M, N and K respectively.
  int64_t mc = 0;
  int64_t nc = 0;
  int64_t kc = 0;

  /// Operand packing. When set, the MR x NR x KC tile's operand is padded and
  /// the pad hoisted out of the ir/jr loops into a contiguous panel rebuilt
  /// once per pc iteration: B~ = (NC/NR) x KC x NR, and A~ = (MC/MR) x KC x MR
  /// transposed to k-major so each k step reads a contiguous MR-element row.
  bool packA = false;
  bool packB = false;

  /// When false LinalgMatmulToContractPass and LinalgVectorizationPass skip
  /// the produced tiles (see isBlockedWithoutVectorize) and they fall through to
  /// convert-linalg-to-loops. For a meaningful scalar comparison LLVM's own
  /// loop/SLP vectorizers must be disabled too, or O2/O3 re-vectorizes them.
  bool vectorize = true;

  /// jc as scf.forall rather than scf.for. The forall becomes omp.parallel via
  /// the existing ForallToParallelLoop + ConvertSCFToOpenMP passes, so each
  /// thread owns a disjoint band of N columns and no reduction is split.
  bool parallel = true;
};

/// Caller-supplied knobs, populated from the blocked distribute pass's
/// options. The mc/nc/kc entries are *upper bounds* on the corresponding
/// cache block, not the block itself: chooseStrategy searches downward from
/// them for a size that is both a multiple of the register tile and a divisor
/// of the matmul's extent.
struct StrategyOverrides {
  int64_t mr = 4;
  int64_t nr = 16;
  int64_t mcTarget = 128;
  int64_t ncTarget = 256;
  int64_t kcTarget = 256;
  bool packA = true;
  bool packB = true;
  bool vectorize = true;
};

/// Decide how (or whether) to block `op`.
///
/// This is the fast-path guard: failure is the normal way a matmul opts out
/// and stays on the existing 64x64 / 8x8x8 pipeline. There is no remainder
/// handling, so every loop level must divide evenly.
///
/// Rejects, in order: non-f32 element types; non-tensor or dynamically shaped
/// operands; ops already marked kBlockedAttrName; ops whose result feeds
/// another linalg op (epilogue-fused matmuls belong to
/// LinalgOuterTileAndFusePass); register tiles that overflow the 16 available
/// ymm registers; and shapes for which no valid cache block exists.
mlir::FailureOr<MatmulStrategy> chooseStrategy(mlir::linalg::MatmulOp op,
                                               const StrategyOverrides &ov);

/// How far the blocked passes have taken a tile. Each pass picks up tiles at
/// the stage the previous one left them in:
///   linalg-matmul-blocked-distribute     → Distributed (inside the ic x jc forall)
///   linalg-matmul-blocked-tile-and-pack  → Tiled (MR x NR x KC, operands packed)
///   linalg-matmul-blocked-kernel         → Kernel (MR x NR x 1)
enum class BlockedStage { Distributed, Tiled, Kernel };

llvm::StringRef stringifyBlockedStage(BlockedStage stage);
std::optional<BlockedStage> symbolizeBlockedStage(llvm::StringRef str);

/// The kBlockedAttrName value recording `s` at `stage`: mr, nr, mc, nc, kc
/// (i64), pack_a, pack_b, vectorize (bool) and stage (string).
mlir::DictionaryAttr buildBlockedConfig(mlir::MLIRContext *ctx,
                                        const MatmulStrategy &s,
                                        BlockedStage stage);

/// `config` with its stage replaced.
mlir::DictionaryAttr withStage(mlir::DictionaryAttr config, BlockedStage stage);

/// A kBlockedAttrName value read back.
struct BlockedConfig {
  MatmulStrategy strategy;
  BlockedStage stage;
};

/// Reads `op`'s kBlockedAttrName. Fails when the attribute is absent, not a
/// dictionary, missing a field, or carries an unknown stage.
mlir::FailureOr<BlockedConfig> readBlockedConfig(mlir::Operation *op);

/// True when `op` is a blocked tile whose strategy has vectorize = false, so
/// it must be left for convert-linalg-to-loops as scalar code.
bool isBlockedWithoutVectorize(mlir::Operation *op);

} // namespace mlir_edsl
