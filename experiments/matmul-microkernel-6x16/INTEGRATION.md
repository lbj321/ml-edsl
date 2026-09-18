# Integrating the blocked matmul into the compiler

How to move this experiment's result (PLAN.md Stages 0–6) into
`buildCPUPipeline` (`cpp/src/MLIRLowering.cpp`) one step at a time, reusing
existing passes wherever possible and removing the ones it supersedes.

**Numbers.** Production today: 18 GFLOPS at 1024³ on 1 core. The 8-thread
number isn't measured yet; Step 0 does that. Experiment: ~100 GFLOPS on 1
thread and ~585 on 8 (Stage 6), correct for every power-of-2 shape.

## Key finding: the new pass only tiles and packs; existing passes build the microkernel

Checked 2026-09-18 at 1024³:
1. Ran the Stage 6 transform only up to "tiled + packed + k-tiled to MR×NR×1,
   un-transpose fused into the k-loop".
2. Handed the rest to existing passes:
   `mlir-edsl-opt -linalg-matmul-to-contract -canonicalize -linalg-vectorize
   -canonicalize -loop-invariant-subset-hoisting -vector-contract-to-outerproduct`,
   then the Stage 6 bufferize/OpenMP lowering.

Results:
- **The k-loop comes out exactly as intended:**
  - C is hoisted into `iter_args` as a `vector<4x16xf32>`;
  - each k step is one `vector.outerproduct` of `vector<4>` × `vector<16>`;
  - asm: 8 `vfmadd231ps`, 4 `vbroadcastss`, 0 spills.
- **Same performance, correct in every run:** 87–88 GFLOPS on 1 thread and 582
  on 8, vs. the Stage 6 script's 84–86 / 578–605.

**Why this works where the script needed workarounds.**
`LinalgMatmulToContractPass` builds the `vector.contract` directly with the
standard (m,k)×(k,n) maps. There's never a `vector.multi_reduction` for
canonicalize to fold away, so the script's `reduction_to_contract` phase
ordering (Stage 3 pitfall 1) and its "vectorize only these ops" targeting
(pitfall 6) aren't needed. `LinalgVectorizationPass` only walks linalg ops, so
unlike `vectorize_children_and_apply_patterns` it never touches
`tensor.insert_slice`.

**The only new pipeline pass:** upstream `createLoopInvariantSubsetHoistingPass()`
(`mlir/Transforms/Passes.h`, already exposed in `mlir-edsl-opt` as
`-loop-invariant-subset-hoisting`). It moves the C-tile read/write out of the
k-loop.
- Production doesn't run it today, so the current 8×8×8 path loads and stores
  its C tile on every k step. Try adding it for the old path too; it may be
  a free win for the fallback (Step 0).

## Existing passes: reuse, supersede, keep

| Pass | Verdict | Notes |
|---|---|---|
| `LinalgMatmulToContractPass` | **reuse** | builds the microkernel's `vector.contract` from the MR×NR×1 tiles |
| `VectorContractToOuterProductPass` | **reuse** | contract → `vector.outerproduct` → FMA |
| `LinalgVectorizationPass` | **reuse** | vectorizes the A-pack transpose tiles, the fused 1×MR un-transpose, and the fill tiles |
| `AllocaScopeCleanupPass` | **reuse** | after `convert-scf-to-openmp` (`-canonicalize` also works, Stage 6, but the narrow pass is deliberate, see its comment) |
| `LinalgGenericTilingPass` | keep | elementwise/epilogue generics; unrelated |
| `VectorCleanupPass` | keep | serves the fallback path and reductions (`linalg.reduce`/dot); the new path doesn't need it |
| `LinalgMatmulParallelTilingPass` (64×64 forall, CPU) | **superseded** for guarded matmuls | delete once nothing falls back (see "Endgame") |
| `LinalgMatmulKTilingPass` | **superseded** | same |
| `LinalgMatmulTilingPass` 8×8×8 (CPU instance) | **superseded** | same. The struct stays: the GPU pipeline uses it (`createLinalgGPUMatmulTilingPass`, 32×32) |
| `LinalgOuterTileAndFusePass` (CPU use) | bare-matmul branch **superseded**; relu-epilogue branch until Step 4 | the struct stays: the GPU pipeline uses it (`MLIRLowering.cpp:498/500`) |

### Marking: keep the superseded passes away from our tiles until they're deleted

The new pass runs first and leaves MR×NR×1 `linalg.matmul` tiles behind. Left
alone:
- `LinalgOuterTileAndFusePass`'s bare-matmul branch would wrap one of them in
  another forall;
- the 8×8×8 `LinalgMatmulTilingPass` would split a 4×16 tile into 4×8 halves;
- with `parallel=false`, the 64×64 parallel tiling would grab them too.

Put a discardable attribute (e.g. `mlir_edsl.blocked`) on every op the new
pass produces, and skip it in those three passes. That's a one-line check
each; the same idea as `LinalgMatmulTilingPass`'s existing "skip if inside an
`scf.forall`" guard. (`LinalgMatmulKTilingPass` is naturally a no-op on K=1
tiles.) `LinalgMatmulToContractPass` and `LinalgVectorizationPass` must *not*
skip them, except when the `vectorize` toggle is off.

## The new pass: `LinalgMatmulBlockedPass`

Runs first in `buildCPUPipeline`, on tensors. Per standalone `linalg.matmul`
that passes the guard:

| Phase | What | C++ API (the script op it replaces) |
|---|---|---|
| 1. Tile | jc (forall if `parallel`) → pc → ic → jr → ir, one level per call to fix the order; fuse the C `linalg.fill` into the jc forall | `scf::tileUsingSCF` ×5 (`tile_using_for[all]`); `scf::tileConsumerAndFuseProducersUsingSCF` at the jc level for the fill (same technique as `LinalgOuterTileAndFusePass`) |
| 2. Pack (per operand, if `packA`/`packB`) | merge slice chains; pad the MR×NR×KC tile (nofold A/B, not C); hoist B above ir, jr, ic and A above ir, jr with transpose `[1,0]` | `tensor::populateMergeConsecutiveInsertExtractSlicePatterns` + canonicalize; `linalg::rewriteAsPaddedOp`; `linalg::hoistPaddingOnTensors` |
| 3. Pack loops | B: tile the pad `[8,0]`, vectorize with `{8, NR}`, and **immediately** fold into B̃. A: decompose the zero-width pad, tile the transpose `[8,0]` (vectorized later by `LinalgVectorizationPass`) | `scf::tileUsingSCF`, `linalg::vectorize(rewriter, pad, {8, nr})`, `tensor::populateFoldTensorSubsetIntoVectorTransferPatterns`, `linalg::populateDecomposePadPatterns` |
| 4. k-tile | MR×NR×KC → MR×NR×1 k-loop; fuse A's un-transpose into it if present; mark | `scf::tileConsumerAndFuseProducersUsingSCF` (`tile_using_for` + `fuse_into_containing_op`) |

The script's contract formation, contract lowering, transpose lowering and
accumulator hoisting all disappear into the reused passes.

**Pitfalls that still apply** (PLAN.md Stage 3, approach 2):
- Merge the slice chains before padding, or hoisting fails with
  "Source not defined outside of loops".
- Fold the B-pack transfer into B̃ before any canonicalize, or it turns back
  into a strided `memref.copy` → `memrefCopy` runtime call.
- `tensor.pad` is not destination-style, so don't try to fuse it with the
  fusion utilities (that's why A's pad gets decomposed instead).
- **The fill:** unfused, `LinalgVectorizationPass` turns a 1024×1024
  `linalg.fill` into one `vector<1024x1024xf32>` (seen in the check above).
  With production's default `convert-vector-to-scf` that becomes a 4 MB stack
  temporary. Phase 1 fuses it into the jc forall, which also parallelizes
  Stage 6's serial `memset`; then tile it to `[8, nr]` so it vectorizes into
  small row stores.

### `MatmulStrategy`: blocking, guard and toggles in one struct

```cpp
struct MatmulStrategy {
  int64_t mr = 4, nr = 16;              // register tile
  int64_t mc = 128, nc = 256, kc = 256; // cache blocks (clamped per shape)
  bool packA = true, packB = true;
  bool vectorize = true;
  bool parallel = true;                 // jc: scf.forall vs scf.for
};
FailureOr<MatmulStrategy> chooseStrategy(linalg::MatmulOp op,
                                         const StrategyOverrides &ov);
```

**`chooseStrategy` is the guard**: failure means the matmul stays on the
existing path.
- **Applies to:** f32; static shapes; result not consumed by a linalg op, so
  epilogue-fused matmuls stay on `LinalgOuterTileAndFusePass` until Step 4.
- **Blocking:** MC = min(mc, M), KC = min(kc, K).
  - NC = N / numThreads when `parallel`, else min(nc, N).
  - Stage 6's 1-thread sweep: NC=128 → ~85 GFLOPS, NC ≥ 256 → ~100–106,
    because every jc block re-packs A. So prefer NC ≥ 256 when there are few
    threads.
- **Divisibility** (no remainder handling, PLAN.md Stage 5): M % MC, MC % MR,
  N % NC, NC % NR, K % KC and KC % 8 must all be 0. With `parallel`, the jc
  loop needs ≥ 2 iterations, or the forall folds away (Stage 6).
- **Register budget:** NR % 8 == 0 and `MR·NR/8 + NR/8 + 1 ≤ 16`
  (4×16 = 11 ✓, 6×16 = 15 ✓, 8×16 = 19 ✗).

| Toggle | Cost | Effect |
|---|---|---|
| `mr`/`nr` | easy | tile sizes; the guard checks the register budget |
| `parallel` | trivial | `LoopType::ForallOp` vs `ForOp` for jc |
| `vectorize` | easy | the marker also makes `LinalgMatmulToContractPass`/`LinalgVectorizationPass` skip our ops, so they go to `convert-linalg-to-loops`. For a clean comparison also turn off LLVM's loop/SLP vectorizers, or O2/O3 will auto-vectorize the loops |
| `packA`/`packB` | medium | skip phases 2–3 for that operand. Phase 4 must then not assume an un-transpose exists (fuse it only if present) |

- **Exposure:**
  - Pass options, so `mlir-edsl-opt` can compare strategies from the command
    line.
  - For the Python JIT, one env var parsed in C++, e.g.
    `MLIR_EDSL_MATMUL="on,mr=6,pack_a=0"` (default: off).
  - **Include the strategy in the JIT cache key**, or changing it inside one
    process returns stale code (cf. `tests/core/test_cache_key_collision.py`).
- **Test presets, not the whole grid:** default, `no-pack`, `no-vectorize`,
  `serial`, `6x16`, plus the guard's rejections.

### Pipeline changes outside the new pass

| Change | Where | Notes |
|---|---|---|
| `LinalgMatmulBlockedPass` | first in `buildCPUPipeline` | off by default until Step 3 |
| `createLoopInvariantSubsetHoistingPass()` | after `LinalgVectorizationPass`, before `VectorContractToOuterProductPass` (the position used in the check above) | affects the old path too (probably positively); run the full suite |
| `BufferHoisting` + `BufferLoopHoisting` | right after one-shot-bufferize, **before** ownership-based dealloc | allocates Ã/B̃ once per thread instead of per ic iteration (Stage 4 measured the page-fault cost of the alternative). Stage 6 confirmed it never lifts allocs out of an `scf.forall`. Global change: run the full suite |
| Scalarization pass (wraps `vector::populateScalarVectorTransferLoweringPatterns`, `allowMultipleUses`) | after bufferization | replaces the test-only `-test-scalar-vector-transfer-lowering`; turns the packed Ã row read into MR scalar loads → `vbroadcastss mem` |
| 2-D transfers vs `convert-vector-to-scf` | production uses default options; the experiment used `full-unroll=true` (Stage 3 Fix 7: otherwise alloca + `memcpy`) | check the asm for `memcpy` in Step 1. Fix locally (unroll 2-D transfers to rows inside the new pass or right after vectorization, `vector::populateVectorUnrollPatterns`) before reaching for a global `fullUnroll` |

## Iterative plan: microkernel first

Each step ships on its own, adds exactly one thing the experiment added, and
comes with a toggle, so you can always measure what that step contributed.

### Step 0: scaffolding and baselines (no behavior change)
- **Code:**
  - `MatmulStrategy` + `chooseStrategy`;
  - an empty `LinalgMatmulBlockedPass` with pass options, registered in
    `mlir-edsl-opt`;
  - the `mlir_edsl.blocked` skip in the three superseded passes;
  - the env var, included in the JIT cache key.
- **Baselines:** a Python benchmark over 256³/512³/1024³ at 1 and 8 threads for
  the *current* pipeline, both with and without
  `LoopInvariantSubsetHoistingPass`. That measures what hoisting alone does
  for the old path.
- **Tests:** guard decisions (power-of-2 accepted; odd shapes, non-f32 and
  epilogue-fused rejected); the full suite unchanged.

### Step 1: microkernel (serial, no packing)
- **Code:** phase 1 with `parallel=false` + phase 4 (without the un-transpose)
  + the fill handling, plus `LoopInvariantSubsetHoistingPass` in the pipeline.
  That's the whole microkernel, because the reused passes do the vectorizing.
- **Toggles:** `mr`/`nr`, `vectorize`.
- **Checks:**
  - FileCheck: one `vector.outerproduct` per k step, a `vector<MRxNR>`
    `iter_args` on the k-loop, and no `linalg.matmul` left;
  - pytest correctness over power-of-2 shapes plus fallback shapes;
  - an asm spot check (2·MR FMAs, 0 spills, no `memcpy`).
- **Expect:** somewhere between 18 and ~100. A is unpacked (MR scalars from MR
  rows) and B has stride N between k steps. Measure; this step is about
  kernel code quality.

### Step 2: packing (B, then A)
- **Code:** phases 2 and 3; B first (a plain row copy), then A (transpose +
  un-transpose fusion in phase 4). Also the hoisting-before-dealloc change and
  the scalarization pass.
- **Toggles:** `packA`, `packB`.
- **Checks:** the Stage 4 structural checks as tests:
  - one alloc/dealloc per packed buffer, at function (later forall-body) top
    level;
  - no `memref.copy`/`memrefCopy`/`memcpy`;
  - flat memory over repeated calls.
- **Expect:** ~100 GFLOPS single-thread at 1024³.

### Step 3: threads, and switch it on
- **Code:** `parallel=true`, the NC heuristic, and the fill fused into the jc
  forall. The OpenMP lowering and libomp loading already exist.
- **Checks:** `test_multicore.py`-style correctness at 1/2/8 threads; allocs
  inside the forall body, never above it.
- **Expect:** ~585 GFLOPS at 1024³ on 8 threads.
- **Then** make the pass on by default for guarded shapes.

### Step 4 (by need): coverage, then deletion
- **Epilogue fusion:** apply bias/ReLU to the C tile before its store (after
  the k-loop, on the `vector<MRxNR>` accumulator), so fused matmuls leave
  `LinalgOuterTileAndFusePass` too. The main piece of genuinely new work.
- **Remainders:** PLAN.md Stage 5 has the dry-run findings; masked edge copies
  for A/B/C are the direction.
- **Stage 7 tuning:** A-pack transpose via shuffles, the 512³ dip, KC/MC sweep,
  β=0 instead of the fill.

### Endgame: what can be deleted, and when
- **After Step 3, if the fallback is still needed** (odd shapes, non-f32):
  delete nothing. The superseded passes *are* the fallback.
- **After Step 4 covers epilogues and remainders:** delete the CPU instances
  of `LinalgMatmulParallelTilingPass` and the 8×8×8 `LinalgMatmulTilingPass`,
  the whole `LinalgMatmulKTilingPass`, and the CPU call of
  `LinalgOuterTileAndFusePass`, plus the `mlir_edsl.blocked` skips that
  protected against them.
  - Keep the `LinalgMatmulTilingPass` and `LinalgOuterTileAndFusePass`
    structs: the GPU pipeline uses them.
  - Revisit whether `VectorCleanupPass` still has users (reductions) before
    touching it.
- **Alternative if remainders never matter:** point the fallback at the new
  pass with `vectorize=false` (scalar but correct) and delete the old matmul
  path anyway. That trades speed on odd shapes for less code.

## Testing per step
1. **IR:** `mlir-edsl-opt` + FileCheck per phase, with the experiment's
   `out/stage6_*` dumps as the reference shape.
2. **Correctness:** pytest over power-of-2 shapes, fallback shapes, and thread
   counts.
3. **Performance:** a benchmark script, not in CI. Record numbers per step in
   this file, the same way PLAN.md does for each stage.
