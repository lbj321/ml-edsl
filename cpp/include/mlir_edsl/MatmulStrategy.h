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
/// It is also what the passes that build the microkernel
/// (LinalgMatmulToContractPass, LinalgVectorizationPass) key on: a matmul
/// without it, or with vectorization turned off, is left for scalar loops
/// (see isLeftForScalarLowering).
constexpr llvm::StringLiteral kBlockedAttrName = "mlir_edsl.blocked";

/// Unit attribute the tile pass sets on the register loops (jr, ir) of a
/// blocked tile: the loops the pack pass hoists the packed panels out of.
constexpr llvm::StringLiteral kBlockedHoistAttrName = "mlir_edsl.blocked_hoist";

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
  /// op's shape. Always satisfy mc % mr == 0, nc % nr == 0 and kc % 8 == 0.
  /// They need not divide M, N and K: the distribute pass pads each extent up
  /// to a multiple of its block.
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
  /// the produced tiles (see isLeftForScalarLowering) and they fall through to
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
/// cache block, not the block itself: chooseStrategy picks a multiple of the
/// register tile at or below them that pads the matmul's extent least.
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
/// Failure is the way a matmul opts out and stays on the existing 64x64 /
/// 8x8x8 pipeline. Extents that no block divides are padded rather than
/// rejected (see paddedExtent), so every static f32 matmul is accepted unless
/// the overrides themselves are invalid.
///
/// Rejects, in order: non-f32 element types; non-tensor or dynamically shaped
/// operands; ops already marked kBlockedAttrName; register tiles that overflow
/// the 16 available ymm registers; and cache-block caps below the register
/// tile. A matmul's consumers do not matter: an epilogue (bias, relu) runs as
/// its own linalg op after the blocked matmul, unfused.
mlir::FailureOr<MatmulStrategy> chooseStrategy(mlir::linalg::MatmulOp op,
                                               const StrategyOverrides &ov);

/// `dim` rounded up to a multiple of `block`: the extent the blocked passes
/// tile once the distribute pass has padded the matmul.
inline int64_t paddedExtent(int64_t dim, int64_t block) {
  return (dim + block - 1) / block * block;
}

/// How far the blocked passes have taken a tile.
///
/// The fast path for f32 matmuls chooseStrategy accepts is four passes
/// (cpp/src/passes/LinalgMatmulBlocked*.cpp), run in this order and each
/// followed by canonicalize in buildCPUPipeline. Each picks up the tiles at
/// the stage the previous one left them in:
///
///   linalg-matmul-blocked-distribute  → Distributed (inside the ic x jc forall)
///   linalg-matmul-blocked-tile        → Tiled (MR x NR x KC)
///   linalg-matmul-blocked-pack        → Packed (A and B operands packed)
///   linalg-matmul-blocked-kernel      → Kernel (MR x NR x 1)
///
/// Together they produce the BLIS loop nest
///
///   jc (N/NC) → pc (K/KC) → ic (M/MC) → jr (NC/NR) → ir (MC/MR) → k (KC/1)
///
/// with jc and ic fused into one forall. The MR x NR x 1 linalg.matmul left
/// at the bottom is the microkernel, but these passes do not build it:
/// LinalgMatmulToContractPass turns it into a vector.contract,
/// LinalgVectorizationPass handles the fill, LoopInvariantSubsetHoisting lifts
/// the C accumulator into the k-loop's iter_args, and
/// VectorContractToOuterProductPass lowers the contract to FMAs.
///
/// Tiles pass from one stage to the next only through kBlockedAttrName: the
/// distribute pass records the strategy there, and each pass advances the
/// stage. Canonicalize removes single-iteration loops, forall included, in
/// between, so no pass holds on to a loop an earlier one created. The one
/// loop-level hand-over is kBlockedHoistAttrName: the pack pass hoists out of
/// the marked loops that survived, and a missing marker means that loop was
/// folded away.
enum class BlockedStage { Distributed, Tiled, Packed, Kernel };

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

/// True when `op` is a linalg.matmul that must be left for
/// convert-linalg-to-loops as scalar code: one the blocked passes did not
/// take (a non-f32 matmul, or invalid overrides), or a blocked tile whose
/// strategy has vectorize = false. Vectorizing an unblocked matmul whole
/// would build one vector the size of the matrix.
bool isLeftForScalarLowering(mlir::Operation *op);

/// Appends the blocked tiles under `root` at `stage`, with their strategy.
/// Emits an error and fails on a malformed kBlockedAttrName.
mlir::LogicalResult collectBlockedTilesAtStage(
    mlir::Operation *root, BlockedStage stage,
    llvm::SmallVectorImpl<std::pair<mlir::linalg::MatmulOp, MatmulStrategy>>
        &tiles);

/// Advances `tile`'s kBlockedAttrName to `stage`, keeping the strategy.
void setBlockedStage(mlir::Operation *tile, BlockedStage stage);

} // namespace mlir_edsl
