# Integrating the blocked matmul into the compiler

How to move this experiment's result (PLAN.md Stages 0–6) into
`buildCPUPipeline` (`cpp/src/MLIRLowering.cpp`) one step at a time, reusing
existing passes wherever possible and removing the ones it supersedes.

**Numbers.** Production today: 18 GFLOPS at 1024³ on 1 core. The 8-thread
number isn't measured yet; Step 0 does that. Experiment: ~100 GFLOPS on 1
thread and ~585 on 8 (Stage 6), correct for every power-of-2 shape.

> **Superseded — see the implementation log at the end of this file.** That 18
> GFLOPS is a CLI-toolchain figure (`opt -O3 -mcpu=skylake`). Measured through
> the JIT, the old pipeline does **25.0 GFLOPS on 1 thread and 191.6 on 8** at
> 1024³, which is the baseline any comparison should use.

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

---

# Implementation log

What actually landed, and where it diverged from the plan above.

## Steps 0 + 1 + parallel outer tiling (DONE)

`LinalgMatmulBlockedPass` runs first in `buildCPUPipeline`, unconditionally.
Packing (Step 2) is **not** done, so A is read as MR scalars from MR rows and
B has stride N between k steps.

**Through `@ml_function` at 1024^3** (9700KF, f32, best of 25 samples — see
the measurement note below):

| threads | old pipeline | blocked path | |
|---|---|---|---|
| 1 | 25.0 | **44.9** | 1.80x |
| 4 | 98.6 | **146.7** | 1.49x |
| 8 | 191.6 | **193.4** | parity |

A clear win at low thread counts, parity on the full machine: at 8 threads
both are memory-bound, which is what packing addresses.

Note the old pipeline's real JIT baseline is **25.0 GFLOPS single-thread**,
not the 18.25 quoted at the top of this file — that figure came from the CLI
toolchain with a fixed `opt -O3 -mcpu=skylake`, and the JIT's own O2/O3 does
better. Compare like with like.

**Single core via `repro/run_step1.sh`** (CLI toolchain, `taskset -c 0`, all
passing `bench_matmul.c`'s naive reference check):

| shape | mr=4 | mr=6 |
|---|---|---|
| 256^3 | 48.2 | — |
| 512^3 | 46.4 | — |
| 768^3 | 49.0 | **58.7** |
| 1024^3 | 40.1 | rejected |

The microkernel is exactly what the "key finding" section predicted: the hot
loop at 1024^3 is 8 `vfmadd231ps`, 4 `vbroadcastss`, 2 `vmovups` (the B tile),
**0 spills**, no `memcpy`, with C held in `ymm0`-`ymm7` across the whole
k-loop and stored only after it exits.

### Divergences from the plan

- **Cache blocks are derived, not fixed.** `mc`/`nc`/`kc` are *upper bounds*;
  `chooseStrategy` searches downward for the largest block that is both a
  multiple of the register tile and a divisor of the extent. The blocking
  follows the microkernel — mr=6 on M=768 picks MC=96 by itself. This replaces
  the plan's separate divisibility checklist: every `%` condition in it is
  implied by a successful search.
- **6x16 is selectable but cannot be the default.** `MC % 6 == 0` forces a
  factor of 3 into MC, and no such MC divides a power-of-2 M, so 1024^3 is
  rejected outright. 4x16 is the default. Where 6x16 is legal it is worth
  having: 768^3 measured **58.7 vs 49.0**, a 1.20x gain.
- **No env var.** The pass is wired into `buildCPUPipeline` unconditionally,
  so `mr`/`nr` are reachable only as `mlir-edsl-opt` pass options; the JIT
  always uses the derived defaults.
- **The parallel level is a 2-D forall over ic x jc**, not jc alone as the
  plan assumed — see below.

## Pitfall: parallelize M and N, not N alone

The plan's Stage 6 shape (jc forall only) is wrong without packing. Splitting
only N makes every thread sweep the whole of A, so total A traffic grows with
the thread count. Measured at 1024^3 on 8 threads:

| jc bands | NC | GFLOPS |
|---|---|---|
| 4 | 256 | 121 |
| 8 | 128 | 104 |

More parallelism, *less* throughput. Sizing NC by thread count — the obvious
fix — makes it worse, not better.

Tiling both dimensions into one `scf.forall` over MC x NC tiles bounds each
thread's working set to an MC-row band of A and an NC-column band of B, and
took 8 threads from 121 to ~193. This is what the old 64x64 forall did, and
what BLIS does: jc and ic are both parallel loops. A sweep of MC/NC targets
(64/64, 64/128, 128/128, 128/256) found nothing better than the 128/256
default, so the remaining gap is not tile tuning.

## Pitfall: LoopInvariantSubsetHoisting has no globally correct position

The microkernel needs this pass to lift the C tile into the k-loop's
`iter_args` — worth **39.8 vs 16.7 GFLOPS** at 1024^3 single-core. Without it
the loop reloads and stores all 8 accumulator registers every k step
(`vmovups` in the hot loop goes 2 -> 10).

It only achieves that **pre-bufferization**, while C is still tensor SSA;
moved after bufferization it hoists nothing for the blocked path.

But it is a whole-function pass, and pre-bufferization it also rewrites the
fused matmul+bias+relu epilogue's `extract_slice`/`insert_slice` chain into
tensor iter_args. That defeats one-shot-bufferize's in-place analysis: the
dense chain lowers with two `memrefCopy` calls the base pipeline does not
emit, and **aborts at run time**.

One phase is right for the blocked matmul and wrong for fusion. It is
currently run globally and the breakage accepted.

**The fix** is to hoist only the blocked loops: mark the k-loop `scf.for` with
`mlir_edsl.blocked` (the marker currently goes on the `linalg.matmul`, which
`LinalgMatmulToContractPass` erases — the loop survives) and run
`hoistLoopInvariantSubsets` on those alone. Watch for the two canonicalizer
runs in between dropping the discardable attribute.

## Pitfall: measure with enough samples

This machine's numbers swing ~1.5x with sample count. The same binary and
shape measured 157, 292 and 196 GFLOPS under three protocols — short runs sit
in turbo, sustained AVX2 FMA load settles at the offset clock. Use at least
25 timed calls and report the spread; several conclusions in this file were
wrong the first time for exactly this reason.

## Known breakage: 13 skipped tests

Ten **crashes**, all epilogue chains. `chooseStrategy` rejects matmuls with a
linalg consumer, so they stay on the old path, which the global hoisting
miscompiles: `test_epilogue_fusion.py` (8), `test_multicore.py::TestMulticoreDenseLayer`
(3), `test_lowering_ir.py::test_dense_layer_large_k_cache_blocked`.

Three **stale IR assertions**, failing because the new path works — bare
matmuls no longer produce a 64x64 `scf.forall`, 8x8 `extract_slice`s or an
`omp.parallel`: `test_bare_matmul_fused`, `test_bare_matmul_not_retiled`,
`test_large_matmul_produces_extract_slices`,
`test_omp_loop_body_has_no_alloca_scope`. These want rewriting against the
blocked path, not un-skipping.

Everything else passes: **707 passed, 16 skipped**.

## Fusion inventory (context for Step 4)

There is exactly one fusion call site in the project:
`tileConsumerAndFuseProducersUsingSCF` in `LinalgOuterTileAndFusePass`. Its
root selection is two-tier — a generic with `library_call == "relu"`, else the
last bare `linalg.matmul` — and it is narrower than it looks:

| chain | fused into the tile | left outside |
|---|---|---|
| matmul -> bias -> **relu** | fill, matmul, bias_add, relu | — |
| matmul -> bias -> **leaky_relu** | fill, matmul | bias_add, leaky_relu |
| matmul -> **bias** only | fill, matmul | bias_add |
| matmul -> `tensor_map(fn)` | fill, matmul | the map generic |

`leaky_relu` is tagged `library_call = "leaky_relu"` by `LinalgBuilder.cpp`
but the pass matches only `"relu"`, so it silently loses epilogue fusion and
does two extra full-array round-trips. Step 4's epilogue-on-accumulator would
cover all four rows uniformly.

## Step 2a: packing B (DONE)

`pack_b` on by default. B's operand slice is padded (`nofold`) and the pad
hoisted out of the ir and jr loops, giving `B~ : tensor<(NC/NR) x KC x NR>`
built once per pc iteration, so the microkernel reads contiguous NR-wide rows
instead of striding by N. A is still unpacked (Step 2b).

**Through `@ml_function` at 1024^3** (9700KF, f32, median of 60, spread in
brackets):

| threads | old pipeline | Step 1 | + B packing |
|---|---|---|---|
| 1 | 25.0 | 44.9 | **84** [1.07] |
| 4 | 98.6 | 146.7 | **300** [1.14] |
| 8 | 191.6 | 193.4 | **~470-500** [1.31-2.60] |

The 8-thread cell is the point of the exercise: Step 1 sat at parity with the
old pipeline there because both were memory-bound. Its spread stays above the
1.3 that `benchmarks/common.py` treats as trustworthy, so read it as a range;
four runs gave 464, 474, 482, 499.

Single core via `repro/run_step1.sh -- pack_b=1`: **85.1** at 1024^3, against
40.1 for Step 1.

### Divergences from the plan

- **Hoist depth is 2, not 3.** The plan assumed the experiment's nest, where
  jc was the only parallel loop. This pass fuses ic and jc into one 2-D
  `scf.forall`, and `hoistPaddingOnTensors` only hoists across `scf.for`, so
  the loops between the tile and the pc body are just ir and jr. Both panels
  land in the pc body, which is where BLIS wants them.
- **The `memcpy` gate was wrong and is not met.** The plan required 0
  `memcpy` in the output; there is 1, in the B-packing loop. Removing it (by
  unrolling the 8 x NR transfers to NR-wide rows) costs **10%** — interleaved,
  4 pairs, zero overlap:

  | | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|
  | with memcpy | 86.06 | 86.62 | 85.88 | 86.90 |
  | unrolled | 77.79 | 78.16 | 77.01 | 78.60 |

  512 contiguous bytes is a size glibc's `memcpy` handles better than eight
  separate row transfers, and this copy is amortized over the NC/NR microtiles
  that read the panel. Stage 3 Fix 7 chased `memcpy` out of the *microkernel*,
  where 12 calls per microtile really did cost; that finding does not carry
  over. If ever revisited: `vector::populateVectorUnrollPatterns` with
  nativeShape `{1, nr}`, filtered to transfers under the pack loop, and
  strictly *after* the subset fold — before it, the round trip collapses back
  to the strided `memrefCopy` the fold exists to prevent.
- **`MLIRLoweringPasses.cpp` was split.** It had reached 1126 lines and 11
  passes; the blocked path is now `cpp/src/passes/LinalgMatmulBlockedPass.cpp`
  (496 lines), leaving 655. Clean cut — nothing else referenced the moved code.
- **Two registrations were missing.** `tensor::registerTilingInterfaceExternalModels`
  and `registerInferTypeOpInterfaceExternalModels`, without which
  tiling the hoisted pad fails with no diagnostic (the `dyn_cast<TilingInterface>`
  just returns null). Both are in `registerCPUDialects`, so `mlir-edsl-opt`
  gets them too.

### What this pass is, precisely

Not a port of `stage6_transform.mlir`. Roughly a third of that script is
reproduced, a third is deliberately replaced by passes the pipeline already
had (the "key finding" above), and a third is Step 2b. The transform-dialect
ops and this pass call the *same* upstream entry points one layer apart:

| script op | upstream call | used here |
|---|---|---|
| `structured.tile_using_for` | `scf::tileUsingSCF` | yes |
| `structured.pad` | `linalg::rewriteAsPaddedOp` | yes |
| `structured.hoist_pad` | `linalg::hoistPaddingOnTensors` | yes |
| `structured.vectorize` | `linalg::vectorize` | yes |
| `apply_patterns.tensor.merge_consecutive_...` | `tensor::populateMergeConsecutive...` | yes |
| `apply_patterns.tensor.fold_tensor_subset_...` | `tensor::populateFoldTensorSubset...` | yes |

There is no hand-written rewriting logic in the pass: it is a shape guard,
handle bookkeeping, and calls into these. The structural differences from the
script are the 2-D forall (measured: 121 -> 193 GFLOPS on 8 threads) and
leaving microkernel construction to the existing passes.

### Checks

- `B~` is `tensor<16x256x16xf32>` at 256^3 — static, (NC/NR) x KC x NR.
- One `malloc`/`free` pair; no `memrefCopy`; max RSS flat over 10 vs 1000
  calls (163.6 vs 163.3 MB).
- Microkernel unchanged: 8 `vfmadd231ps`, 4 `vbroadcastss`, 0 spills.
- `tests/linalg/test_blocked_matmul_ir.py` covers the 2-D forall, the marked
  4x16x1 tile, the packed panel, the outerproduct kernel, and that a rejected
  shape (24^3) is left alone. Suite: **712 passed, 16 skipped**.

### Unrelated flake found

`test_binary_op_execution.py::test_three_layer_net` fails about 1 run in 9.
Nothing to do with this work — the blocked pass leaves that function entirely
untouched (all 3 matmuls have linalg consumers, so the guard rejects them;
verified in the `linalg-matmul-blocked` snapshot). It is unseeded random data
with `rtol=1e-4` and no `atol`, through a relu whose output can land near
zero: absolute error is ~2e-6 in every trial, relative error explodes only
when the expected value does not. One-line fix, not applied here.

## Step 2b: packing A (DONE)

`pack_a` on by default. A's pad is hoisted with a `[1, 0]` transpose into
`A~ : tensor<(MC/MR) x KC x MR>`, the zero-width pad is decomposed so the
transpose reads A directly, the packing transpose is tiled to 8 rows, and the
un-transpose `hoistPaddingOnTensors` leaves behind is fused into the k-loop by
`tileConsumerAndFuseProducersUsingSCF` — so each k step transposes a 1 x MR
row rather than copying MR x KC per microtile.

**Through `@ml_function` at 1024^3**, median of 60:

| threads | old | Step 1 | + B pack | + A pack |
|---|---|---|---|---|
| 1 | 25.0 | 44.9 | 84 | **92.8** |
| 4 | 98.6 | 146.7 | 300 | **330.5** |
| 8 | 191.6 | 193.4 | ~470-500 | **539-549** |

3.7x / 3.4x / 2.85x over the old pipeline, and within reach of the
experiment's ~100 / ~585.

### A-packing needs the transposing transfer lowered, or it is a net loss

First attempt measured *slower* than B-only — 83 vs 86 single core, 443 vs
~470-500 on 8 threads. The packing loop was a `vinsertps` chain:

    18 vmovups  15 vinsertps  9 vbroadcastss  8 vmovaps  5 vblendps  2 vshufps

The cause was not that the transpose is unvectorized. `LinalgVectorizationPass`
does vectorize it — into a `vector.transfer_read` carrying a `permutation_map`,
i.e. a transposing load, which `convert-vector-to-llvm` can only do
element-wise.

Fix: `VectorTransposeLoweringPass`, new, after `LinalgVectorizationPass`. It
runs `populateVectorTransferPermutationMapLoweringPatterns` (splitting the
permuting transfer into a plain transfer plus an explicit `vector.transpose`)
and `populateVectorTransposeLoweringPatterns` with `Shuffle16x16`. These are
the script's `transfer_permutation_patterns` and `lower_transpose` phases,
which the "key finding" section had written off as unnecessary — they are not,
once A is packed. The loop becomes a real in-register transpose:

    24 vmovups  9 vshufps  8 vmovaps  4 vunpcklps  4 vunpckhps
     4 vextractf128  2 vinsertps  2 vpermpd

and A-packing flips from -3% to +11%, interleaved single core:

| | 1 | 2 | 3 |
|---|---|---|---|
| B only | 81.58 | 86.70 | 86.21 |
| A + B | 95.01 | 95.86 | 94.88 |

### Notes

- The plan's scalarization pass
  (`vector::populateScalarVectorTransferLoweringPatterns`) is not implemented
  and is not needed: the microkernel reads A as MR `vbroadcastss` whether or
  not A is packed.
- `VectorTransposeLoweringPass` is a global pipeline change; the full suite
  is unaffected.
- Panels at 256^3: `B~ = tensor<16x256x16xf32>`, `A~ = tensor<32x256x4xf32>`.
  Microkernel unchanged at 8 `vfmadd231ps`, 4 `vbroadcastss`, 0 spills. Max
  RSS flat over 10 vs 1000 calls. **712 passed, 16 skipped.**

## Next

1. **Targeted hoisting**, which un-skips the ten crashers. Now the largest
   item: a dense layer with a relu is miscompiled today.
2. **Rewrite the three stale IR assertions** against the blocked path
   (`test_bare_matmul_fused`, `test_bare_matmul_not_retiled`,
   `test_large_matmul_produces_extract_slices`,
   `test_omp_loop_body_has_no_alloca_scope`). New coverage for the blocked
   path itself is in `tests/linalg/test_blocked_matmul_ir.py`.
3. **The `test_three_layer_net` flake** (see Step 2a), one line.
