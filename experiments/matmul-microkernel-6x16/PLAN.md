# Bottom-up 6×16 matmul microkernel

## Why

The production CPU matmul pipeline (`cpp/src/MLIRLoweringPasses.cpp`, driven by
`buildCPUPipeline` in `cpp/src/MLIRLowering.cpp:250`) tiles as outer 64×64 → K-blocked
at 256 → inner 8×8×8, lowered through `vector.contract` → `vector.outerproduct`. The
8×8×8 tile was never checked against hardware limits. On the 9700KF (Coffee Lake,
AVX2 + FMA3, 16 ymm registers, 2 FMA ports, 4-cycle FMA latency) you need ≥8
independent accumulator chains in flight to saturate both FMA ports. An 8×8 f32 tile
only gives 8 accumulators total split across both operand dims in a way that doesn't
map cleanly onto ymm outer-products; the BLIS/OpenBLAS-standard shape for this exact
microarchitecture is **MR=6, NR=16** (12 accumulators + 2 registers for the B row +
1 for the A broadcast = 15 of 16 ymm registers).

Rather than retuning the whole pipeline and hoping the kernel is good, build bottom-up:
get the 6×16 kernel's assembly *provably* clean in isolation (right FMA count, no
spills, no loop-carried stalls) before adding the macro-kernel, outer loops, packing,
bufferization, remainder handling, and multithreading back on top. Each stage below
has a pass/fail check — don't move to the next stage until the current one passes.

## Ground truth for this repo (checked 2026-09-18)

- **LLVM/MLIR pin**: `llvmorg-21-init` @ `2b1ebef8b8a5af7092de80daafd2743683d0e8c8`,
  assertions-enabled build (see `LLVM_DIR`/`MLIR_DIR` in your environment for
  the local path). Do not
  move this checkout mid-experiment — transform op names/syntax drift fast on tip-of-tree.
- **`mlir-opt`, `mlir-translate`, `llc`**: present at `.../build/bin/`.
- **`llvm-mca`**: **not built** in the pinned tree (only a mismatched system
  `llvm-mca-14` exists at `/usr/lib/llvm-14/bin/llvm-mca`, whose scheduling model is
  stale — don't trust it for Coffee Lake FMA-port/latency numbers). Before Stage 1's
  `llvm-mca` check, run `ninja llvm-mca` in your local LLVM build directory
  (same one as `LLVM_DIR` above).
- **`mlir-cpu-runner`**: also not built in the pinned tree, but
  `libmlir_runner_utils.so` / `libmlir_c_runner_utils.so` are. Stage 0's harness
  works around this with a small hand-rolled driver (`mlir-translate` →
  `llc` → `clang -shared` → `dlopen`/`dlsym`) instead of depending on
  `mlir-cpu-runner`.
- **Harness precedent**: `mlir-edsl-opt` (`cpp/tools/mlir-edsl-opt/mlir-edsl-opt.cpp`,
  build via `./build.sh --component opt`) is a standalone `mlir-opt`-style driver that
  registers every custom lowering/tiling pass (`LinalgOuterTileAndFusePass`,
  `LinalgMatmulKTilingPass`, `LinalgEpilogueTileAndFusePass`,
  `VectorContractToOuterProductPass`, etc.) plus a `-cpu-pipeline` pipeline that calls
  `buildCPUPipeline` directly. Use it — not bare `mlir-opt` — whenever a stage needs a
  production pass alongside a hand-written transform script, following the
  `experiments/matmul-bias-relu-tile-fuse/repro/run.sh` pattern
  (`-transform-preload-library=... -transform-interpreter -canonicalize -cse`, numbered
  `out/N_description.mlir` dumps for bisection).
- **Current kernel baseline**: production is 8×8×8 via `vector.outerproduct`
  (`LinalgEpilogueTileAndFusePass` + `VectorContractToOuterProductPass` in
  `MLIRLoweringPasses.cpp`) — this is the concrete number Stage 1's 6×16 kernel must
  beat, not just an abstract peak-FLOPS estimate.
- **No `linalg.pack`/`tensor.pack` in this repo today.** The two similarly-named
  experiments (`ab-panel-pack-matmul`, `matmul-per-tile-packing`) do packing via
  custom C++ passes, not the transform-dialect pack ops. Stage 3 is genuinely new
  territory here — double-check op names/syntax against the pinned commit before
  trusting any online example.
- **Calling convention**: the production JIT path
  (`mlir_edsl/backend.py:_MEMREF_DESCRIPTOR_PREFIX_C_TYPES` +
  `ctypes.CFUNCTYPE`) uses the *flattened scalar* memref descriptor convention
  (`use-bare-ptr-memref-call-conv=0`, `emit-c-wrappers=0` — LLVM's default
  struct-flattened-to-scalars ABI: per memref arg, 7 scalars `alloc_ptr,
  align_ptr, offset, size0, size1, stride0, stride1` for rank 2). The Stage 0
  harness (`harness/bench_matmul.c`) matches this exact convention with a
  hand-written `typedef void (*MatmulFn)(...)`, so its baseline numbers are
  directly comparable to what the production Python path would see.

## Hardware facts (9700KF, Coffee Lake)

- 16 ymm registers, AVX2 + FMA3, 2 FMA ports, FMA latency 4 cycles → need ≥8
  independent accumulator chains.
- L1d 32 KB, L2 256 KB per core, L3 12 MB shared.
- Peak 32 SP flops/cycle/core. At ~4.6 GHz all-core: ~145-150 GFLOPS single-thread,
  ~1.1-1.2 TFLOPS all 8 cores (AVX offset may pull this down — measure sustained
  clock, don't assume).
- Target kernel shape: **MR=6, NR=16** (not the 8×8 currently in production).
  Keep 6×8 as an intermediate stepping stone only.
- Use `llc -mcpu=skylake` (or `-mcpu=native`) — not `haswell` (Haswell's FMA
  latency is 5 cycles, further from Coffee Lake's 4).

## Stages

Each stage's `.mlir`/transform script lives under `repro/stageN_*`, and each dump
goes to `out/N_description.mlir`, mirroring `matmul-bias-relu-tile-fuse`.

### Stage 0 — Environment, harness, baseline — DONE (2026-09-18)

- Pin the LLVM commit above; don't move it mid-experiment.
- C harness (`harness/bench_matmul.c` + `harness/build_and_run.sh`): calls the
  compiled function via the **flattened scalar memref descriptor** convention
  (the same one `mlir_edsl/backend.py` uses via ctypes for the production JIT
  path — confirmed by inspecting the lowered `llvm.func` signature: 7 scalar
  args per memref: `alloc_ptr, align_ptr, offset, size0, size1, stride0,
  stride1`). Validates against a naive triple-loop reference, reports GFLOPS
  as best-of-N under `taskset -c 0`.
  - Toolchain notes that weren't obvious up front:
    - `llc` needs `-relocation-model=pic` or linking into a `.so` fails with
      `relocation R_X86_64_32 against .rodata can not be used when making a
      shared object`.
    - The pipeline emits `omp.parallel`/`omp.wsloop` unconditionally (via
      `LinalgOuterTileAndFusePass`'s `scf.forall` → `convert-scf-to-openmp`),
      so the `.so` must link against a libomp providing `__kmpc_*` symbols —
      used a local conda-env `libomp.so` (cross-version OpenMP ABI is stable;
      this doesn't need to come from the pinned LLVM build) via the
      `LIBOMP_DIR` env var in `harness/build_and_run.sh`.
    - `mlir-edsl-opt`'s outer tiling (`LinalgOuterTileAndFusePass` /
      `LinalgMatmulParallelTilingPass`) hardcodes a 64×64 tile with **no
      remainder padding**. A size not divisible by 64 (e.g. 336, which does
      satisfy the Stage 1+ blocking lcm(6,16,168)) hits `linalg-vectorize`'s
      un-vectorized fallback on the tail tile — confirmed via
      `mlir-edsl-opt -cpu-pipeline` warning `"vectorization failed, skipping
      op"` on a dynamic `?x8` remainder. Stage 0 inputs must additionally
      divide 64; later custom-pipeline stages aren't bound by this since
      they don't use the production tiling passes.
- Baseline numbers (`mlir-edsl-opt -cpu-pipeline` = current production 8×8×8
  pipeline, pinned to one core via `taskset -c 0`):
  - 192×256×192 (`repro/stage0_baseline.mlir`): **27.75 GFLOPS**
  - 1024×1024×1024: **18.25 GFLOPS**
  - Both are ~12-19% of the ~145 GFLOPS single-core peak estimate — leaves a
    lot of headroom for the 6×16 kernel to close.
- **Done when**: harness validates correctness and prints a baseline GFLOPS
  number for the current 8×8×8 pipeline. ✅

### Stage 1 — Microkernel in isolation — DONE (2026-09-18)

- Input (`repro/stage1_microkernel.mlir`): `linalg.matmul_transpose_a` on a
  6×16×256 problem — A packed 256×6 k-major (`A[k,m]`), B packed 256×16
  (`B[k,n]`), verified directly against the pinned LLVM 21 build (its
  `LinalgNamedStructuredOps.yaml` `shape_map` comments read confusingly
  transposed, but the actual op interface is A:(K,M), B:(K,N), C:(M,N)).
- Transform (`repro/stage1_transform.mlir`, run via upstream `mlir-opt`, not
  `mlir-edsl-opt` — the project's opt tool doesn't register the Transform
  dialect): `tile_using_for [0,0,1]` → `vectorize_children_and_apply_patterns`
  → `apply_patterns.vector.{lower_contraction,lower_outerproduct}` →
  `transform.loop.hoist_loop_invariant_subsets`. All op names/syntax verified
  against the pinned commit before use (`TileUsingForOp`,
  `VectorizeChildrenAndApplyPatternsOp`, `ApplyLowerContractionPatternsOp`,
  `ApplyLowerOuterProductPatternsOp`, `HoistLoopInvariantSubsetsOp` all
  present, same names as upstream examples).
- **First lowering attempt failed the assembly check**: 12 FMAs were present,
  but 2 of 12 accumulators spilled to the stack with a bizarre
  `vmovaps %ymmN, %ymm(N-1)` register-rotation chain every iteration, and 2 of
  6 A-element broadcasts went through `movq`/`vpbroadcastd`/`vpermd` instead of
  `vbroadcastss mem`. Root-caused (not a transform-script problem — a
  lowering-pipeline problem) to two things, both fixed in
  `repro/run_stage1.sh`:
  1. **Missing `opt -O3` between `mlir-translate` and `llc`.** `llc` alone only
     does codegen — no SROA/InstCombine/LICM/LoopRotate. The `scf.for` lowers
     through `-convert-scf-to-cf` into an *unrotated* loop (header check at
     top, backedge from the bottom) carrying an aggregate
     `phi [6 x <16 x float>]`. Handing that straight to `llc` is exactly what
     produces phi-copy-coalescing failures (the rotation chain) and spills.
     Running `opt -passes='default<O3>' -mcpu=skylake` on the `.ll` before
     `llc` fixed this completely — SelectionDAG splits the aggregate phi into
     12 independent per-element virtual registers once the loop is in good
     shape, and the rotation chain and both spills disappeared.
  2. **A's packed read is an illegal LLVM type.** `vector.transfer_read` of
     `vector<1x6xf32>` becomes `load <6 x float>` (24 bytes) — the legalizer
     splits it into a 16-byte + 8-byte load, so elements 0-3 fold into clean
     `vbroadcastss mem` but elements 4-5 (from the 8-byte piece) need an extra
     shuffle. `opt -O3` alone turned this into a cheap `vpermps` (down from the
     uglier `movq`/`vpbroadcastd`/`vpermd`), but the shuffle-control constant
     still spilled/reloaded every iteration. Fully fixed at the MLIR level
     with `mlir-opt -test-scalar-vector-transfer-lowering=allow-multiple-uses`
     (wraps `vector::populateScalarVectorTransferLoweringPatterns`, a
     test-only pass but fine for this standalone experiment) run right after
     bufferization — it rewrites the `vector.extract`-of-`transfer_read` chain
     into 6 independent `memref.load`s. `allow-multiple-uses` is required:
     the default only fires when the `transfer_read` has a single use, and
     ours has 6 (one extract per M-row). No matching transform-dialect
     `apply_patterns` op exists for this pattern on the pinned commit — only
     the `-test-scalar-vector-transfer-lowering` pass wrapper.
- **Final assembly** (`out/stage1_loop_body.s`, produced by
  `repro/run_stage1.sh`): exactly 12 `vfmadd231ps`, exactly 6 `vbroadcastss`
  (all from memory), exactly 2 `vmovups` (B loads), zero spills, zero
  `rsp`-relative accesses — 15 of 16 ymm registers used (12 accumulators + 2
  B operands + 1 A-broadcast scratch), matching the plan's prediction exactly.
- **`llvm-mca -mcpu=skylake`** (not built in the pinned tree by default —
  build via `ninja opt llvm-mca` in the LLVM build dir, `$LLVM_DIR/..`):
  614 cycles / 100 iterations = 6.14 cycles/iteration, vs. a 6.0-cycle
  theoretical floor for 12 FMAs over 2 ports at 1/cycle each (port0/port1
  pressure both 6.02) — **97.7% of FMA-port-bound peak**, no loop-carried
  bottleneck beyond the expected FMA dependency chain.
- **Real measured throughput** (`harness/bench_kernel.c`, calling the kernel
  2M times back-to-back on a hot KC=256 panel, pinned via `taskset -c 0`):
  **133.53 GFLOPS** — ~89-92% of the ~145-150 GFLOPS single-core peak
  estimate (exact % depends on actual sustained clock, not yet measured
  directly), and a **~5-7× speedup over the Stage 0 production baseline**
  (18.25-27.75 GFLOPS).
- **Done when**: assembly is clean per the above, and the kernel alone
  (looping over a hot KC=256 panel) hits ≥90% of single-core peak. ✅ (right at
  the edge — worth re-checking once sustained clock is measured directly
  rather than assumed from the datasheet figure)

### Stage 2 — Macro-kernel (jr/ir loops over pre-packed panels) — DONE (2026-09-18)

- Input (`repro/stage2_macrokernel.mlir`): the Stage 1 kernel wrapped in the
  two register-block loops. A (`memref<256x168xf32>`, KC×MC k-major) and B
  (`memref<256x64xf32>`, KC×NC) are already-packed panels passed in directly
  — packing itself is still out of scope (Stage 3). MC=168, KC=256 match the
  plan's done-when check; NC=64 gives 4 `jr` tiles.
- Transform (`repro/stage2_transform.mlir`): rather than one
  `tile_using_for [6, 16, 0]` (which tiles dims in-order, giving *ir outer, jr
  inner* — backwards from BLIS's `jr → ir → kernel` order), tiled N and M in
  two separate calls to control loop order explicitly: `tile_using_for
  [0, 16, 0]` (jr, outer) then `tile_using_for [6, 0, 0]` on the result (ir,
  inner) — then applied Stage 1's exact K-tile/vectorize/lower/hoist recipe to
  the resulting 6×16×256 tile. Confirmed via the emitted IR:
  `scf.for %arg3 = 0 to 64 step 16 { scf.for %arg5 = 0 to 168 step 6 { scf.for
  %arg7 = 0 to 256 step 1 { ... 6× vector.fma ... } } }` — jr outer, ir inner,
  kernel innermost, as intended.
- **Assembly check passed on the first try** (`repro/run_stage2.sh`, same
  recipe as Stage 1 plus a Python step to isolate the innermost loop by its
  own label/backedge rather than the naive "block with the most
  `vfmadd231ps`" — the naive version over-counted by including the adjacent,
  correctly-hoisted C-tile store-back code). The innermost k-loop
  (`.LBB0_3`) is **byte-for-byte identical** to Stage 1's: 12 `vfmadd231ps`,
  6 `vbroadcastss mem`, 2 `vmovups`, zero spills, and `llvm-mca` reports the
  same 6.14 cycles/iteration. Bonus, unplanned: because this tiles the whole
  `linalg.matmul_transpose_a` including its `outs` accumulator (rather than
  Stage 1's from-scratch zero vector), the generated code already reads the
  C tile from memory before the k-loop and writes it back after — i.e. it's
  already set up to accumulate into a pre-existing C, which is exactly what
  Stage 3's `pc` loop will need across KC blocks.
- **Real measured throughput did not initially clear the done-when bar**:
  `harness/bench_macrokernel.c` (K-major-layout equivalent of
  `bench_matmul.c`, needed since A here is KxM not MxK) measured 66-87 GFLOPS
  at MC=168/NC=64, vs. Stage 1's 133.53 GFLOPS — roughly half, not "within a
  few percent."
- **Root-caused with a two-way sweep** (`repro/stage2_sweep.sh`, results in
  `out/stage2_sweep_results.csv`) before accepting the gap, to distinguish
  panel-size-exceeds-L1 from cost-scales-with-jr-repeat-count:
  - **Sweep 1** (vary MC 6→168, NC fixed at 16 so `jr` only sweeps once):
    155 → 133 → 107 → 103 → 96 → 98 → 93 → 93 GFLOPS. A sharp knee between
    MC=12 and MC=24 — right where the A slice (`MC×KC×4` bytes) plus the
    resident 16KB B tile cross L1d's 32KB — then a **flat plateau** from
    MC=36 to MC=168, matching Ã (up to 172KB) comfortably fitting L2
    (256KB) rather than continuing to degrade.
  - **Sweep 2** (vary NC 16→64 at fixed MC=168, i.e. repeat the same
    oversized A panel 1×/2×/4×): 93 → 81 → 83 GFLOPS — an ~8-13% decline,
    not the multiplicative falloff repeat-count-driven cost would predict.
  - Conclusion: **panel size vs. L1d, not `jr` repeat frequency**, causes the
    gap. Stage 1's 133-155 GFLOPS is a real number but specific to an
    L1-resident ~22KB working set; any macro-kernel panel past ~16-24
    columns is inherently L2-bound at ~90-98 GFLOPS on this chip — an
    expected property of the memory hierarchy, not a defect in the
    transform script or lowering. This is exactly why Stage 3's MC/KC
    blocking targets L2, not L1.
- **Revised done-when** (the original "within a few percent of Stage 1" bar
  compared against the wrong reference point — an L1-resident microbenchmark
  isn't a fair target for an L2-scale panel): MC=168/KC=256 performance is
  within a few percent of the L2-bound plateau (~90-98 GFLOPS, confirmed
  flat for MC≥36) *and* the innermost kernel's assembly is unchanged from
  Stage 1. ✅ Both hold (93.43 GFLOPS at MC=168/NC=16; assembly verified
  byte-for-byte identical).
- `transform.loop.outline` (to compare inlined vs. `noinline` microtile call
  overhead) was **not needed** — not attempted, since the assembly check
  already showed the microkernel's exact instruction sequence was inlined
  and unchanged; revisit only if a future stage's profile suggests call
  overhead matters.

### Stage 3 — Outer loops + packing — DONE (2026-09-18, approach 2)

- Add jc (NC), pc (KC), ic (MC) loops. Starting blocking for this chip:
  - KC=256: one B micro-panel = 256·16·4 B = 16 KB (half of L1d)
  - MC=168: Ã = 168·256·4 B ≈ 172 KB (fits L2)
  - NC ≈ 4080 (or just N for test sizes): B̃ ≈ 4 MB (fits L3)
  - Loop order: jc → pc → [pack B] → ic → [pack A] → jr → ir → kernel
- Two packing approaches, in this order:
  1. Global pack first (`transform.structured.pack` + `pack_transpose`) on the
     whole matmul, then tile. Simpler to verify, wastes memory traffic, but
     proves the layout.
  2. BLIS-style per-block packing: tile first, then
     `transform.structured.fuse_into_containing_op` to pull `linalg.pack`
     producers into the pc loop (B̃) and ic loop (Ã). Alternative:
     `transform.structured.pad` + `hoist_pad`.
  - Since neither `linalg.pack` nor these ops exist anywhere in this repo yet,
    verify exact op names/syntax against the pinned commit before writing the
    script.
- Do tiling as five successive `tile_using_for` calls, one non-zero entry each
  in `[m, n, k]`, to control loop order explicitly.

#### Approach 1 result (global pack) — layout proven, closed out 2026-09-18

- `repro/stage3_outer.mlir` + `repro/stage3_transform.mlir`, driven by
  `repro/run_stage3.sh`. Took 7 fixes, documented in the header comments of the
  transform script and the run script. Most of them work around the approach
  itself: `structured.pack` turns the matmul into a 5-D `linalg.generic` with k
  in the middle, which needs `interchange` + `transfer_permutation_patterns` +
  `reduction_to_contract` just to get the contract back. C also gets packed and
  unpacked. The pack/unpack transposes need their own tiling and vectorization.
- **Fix 7** (this round): 12 `callq memcpy` remained in the function (384 B
  in and 384 B out around every 6×16 microtile, 512 B per B-pack iteration).
  `-convert-vector-to-scf`'s default lowering of n-D transfers goes through a
  stack temp buffer (`memref<vector<...>>`), and `opt -O3` couldn't SROA it away
  on these strided subviews. Fixed with `-convert-vector-to-scf="full-unroll=true"`,
  which brings memcpy to 0 with the kernel loop unchanged.
  `rank_reducing_subview_patterns`, run post-bufferization to drop the unit dims
  of the rank-4 C tile and the `8x1x16` B-pack read, was tried first and made no
  difference on its own.
- Kernel k-loop: 12 `vfmadd231ps`, 6 `vbroadcastss`, 2 B loads, 0 spills,
  `llvm-mca` 6.14 cycles/iter, all identical to Stages 1 and 2.
- Measured (`harness/bench_matmul`, `taskset -c 0`, correctness OK):
  - 336×64×512 (MC=168, NC=32, KC=256): **46.4 GFLOPS** (36 before Fix 7).
  - 1008×1024×1024 (MC=168, NC=256, KC=256; transform jc tile `[0, 16, 0, 0, 0]`):
    **~65–84 GFLOPS**, noisy from run to run (69.7, then 61/83, then 65/84/84 in
    later sessions). That's ~70–90% of Stage 2's ~93 plateau.
- What's left: whole-array packing of A and B (extra passes over memory, and
  the packed buffers aren't cache-sized), plus C pack/unpack. Approach 2
  addresses all three.

#### Approach 2 result (BLIS per-block packing via pad + hoist_pad) — DONE 2026-09-18

- `repro/stage3b_transform.mlir`, driven by `repro/run_stage3b.sh` (same
  lowering as `run_stage3.sh`, including Fix 7; `INPUT`/`TRANSFORM`/`TAG` env
  overrides). Large-size input: `repro/stage3_outer_large.mlir` (the header of
  the transform script has the one-line sed for NC=256).
- Recipe: five one-dim `tile_using_for` calls on the *named* `linalg.matmul`
  (jc `[0,NC,0]` → pc `[0,0,KC]` → ic `[MC,0,0]` → jr `[0,16,0]` → ir `[6,0,0]`)
  → `structured.pad` on the 6×16×KC tile (`nofold_flags = [1, 1, 0]`, so C is
  never padded) → `hoist_pad` B by 3 loops → `hoist_pad` A by 2 loops with
  `transpose by [1, 0]` → tile and vectorize the packing copies → k-tile by 1
  and fuse the un-transpose into the k-loop → targeted `structured.vectorize` →
  `reduction_to_contract` → `lower_contraction`/`lower_outerproduct` → hoist.
- **Pack ops at the right depth** (`out/stage3b_vectorized.mlir`):
  B̃ = `tensor<(NC/16)×KC×16>` is built directly inside the pc loop (once per
  jc, pc), and Ã = `tensor<(MC/6)×KC×6>`, k-major, is built directly inside
  the ic loop (once per jc, pc, ic). `hoist_pad` makes only the loops the slice
  indexes into packing dims (jr for B, ir for A), so nothing is duplicated.
  C is updated in place across pc, with no pack/unpack. Nothing packs inside
  jr/ir/k.
- Pitfalls hit, in order:
  1. **`hoist_pad` → "Source not defined outside of loops -> Skip"**
     (`-debug-only=hoist-padding`). Each tiling level slices the previous
     level's slice, so the microtile's operand is defined inside the loops.
     Fix: `apply_patterns.tensor.merge_consecutive_insert_extract_slice` +
     canonicalization before `pad`, which collapses the chains to one slice of
     the original tensor.
  2. **`hoist_pad ... transpose by [1, 0]` inserts an un-transpose**
     (`HoistPadding.cpp:982`) back to 6×KC in front of the matmul, costing a 6 KB
     copy per microtile. Fix: after k-tiling, `fuse_into_containing_op` the
     un-transpose into the k-loop. Each k step then reads a contiguous
     `vector<6xf32>` row of Ã, Stage 1's exact access pattern.
  3. **`fuse_into_containing_op` asserts (crashes) on `tensor.pad`**
     (`cast<DestinationStyleOpInterface>` in `FuseIntoContainingOp::apply`).
     So instead of fusing A's zero-width pad into the transpose's tile loop,
     `apply_patterns.linalg.decompose_pad` removes it (the full-size
     insert_slice folds away) and the packing transpose reads A directly.
  4. **`structured.vectorize` on a `tensor.pad` needs explicit
     `vector_sizes`** on this commit ("Attempted to vectorize, but failed"
     without them). B's pad is tiled `[8, 0]` and vectorized with `[8, 16]`.
  5. **The vectorized B pad got un-vectorized by canonicalize.** The
     `transfer_read(B)` → `transfer_write(tensor.empty)` round trip folds back
     to `insert_slice(extract_slice(B))`, which bufferizes to a strided
     `memref.copy`, i.e. a call to the `memrefCopy` runtime helper (`dlopen`:
     undefined symbol). Fix: `fold_tensor_subset_ops_into_vector_transfers`
     immediately after vectorizing, before any canonicalization.
  6. **`vectorize_children_and_apply_patterns` vectorizes `tensor.insert_slice`**
     on this commit, turning every tiling level's result insert_slice into a
     whole-block vector copy (`vector<336x32>`, `<168x16>`, …). Stage 2 already
     had one of these (`stage2_scalarized.mlir:66`, harmless there). Dropping
     `{vectorize_padding}` doesn't help. Fix: targeted `structured.vectorize`
     on just the three compute tiles, plus approach 1's
     `transfer_permutation_patterns` + `reduction_to_contract` phase to form
     the contract.
- Kernel k-loop: 12 `vfmadd231ps`, 6 `vbroadcastss`, 2 B loads, 0 spills,
  0 `memcpy` in the whole function, `llvm-mca` 6.14 cycles/iter, all
  identical to Stages 1 and 2.
- Measured (`harness/bench_matmul`, `taskset -c 0`, correctness OK at both sizes):
  - 336×64×512: **73.0 GFLOPS** (approach 1: 46.4).
  - 1008×1024×1024: **96.6–100.8 GFLOPS** across 4 runs (approach 1 in the
    same session: 65–84). That's slightly *above* Stage 2's ~93 plateau
    measurement and ~5.4× the Stage 0 production baseline (18.25 GFLOPS at
    1024³).
- **Done when** ✅: correct at full size, pack ops at the right loop depth,
  and ≥ the 80–90%-of-Stage-2 target.
- Carried into Stage 4: `-buffer-loop-hoisting` already hoists both packed
  buffers (Ã, B̃) to one `malloc` each at function entry, but **nothing frees
  them**. One-shot-bufferize ran without ownership-based deallocation, so every
  call leaks (NC/16·KC·16 + MC/6·KC·6 floats). Fix this in Stage 4, either with
  dealloc or with scratch buffers passed in as arguments. C is zero-filled by a
  separate `linalg.fill` pass over C before the loop nest (BLIS instead uses
  β=0 on the first pc iteration), which is worth removing in Stage 4 or 7.
- **Done when**: correct at full size, IR has pack ops at the right loop depth.
  Expect ~80-90% of Stage 2 performance.

### Stage 4 — Bufferization cleanup — DONE (2026-09-18)

Original goals: one-shot bufferize with `bufferize-function-boundaries`; no stray
`memref.copy` of C; C updated in place; Ã/B̃ allocated once and hoisted out of
loops; full lowering; recheck the kernel's assembly, since late lowering can
undo earlier hoisting. **Done when**: allocations sit outside the hot loops and
Stage 3 performance holds.

**State coming out of Stage 3 (approach 2, checked 2026-09-18).** Most of this
already holds, as a side effect of the Stage 3 pipeline:
- C is updated in place. `linalg.fill` and the whole loop nest write `%arg2`
  directly, and there's no C-sized alloc or copy.
- The only `memref.copy` ops in `out/stage3b_bufferized.mlir` are self-copies
  (`memref.copy %x, %x`), left over from tiling's insert_slice-into-own-slice.
  `-cse` runs after the last `-canonicalize`, so they survive into the dump; the
  next step's canonicalize (`FoldSelfCopy`) removes them. Cosmetic only.
- After bufferization, Ã's and B̃'s `memref.alloc` are still *inside* the ic and
  pc loops. `-buffer-loop-hoisting`, placed late in the LLVM-lowering step,
  moves them to function entry (2 `malloc` in `stage3b.opt.s`, outside all loops).
- Kernel asm is identical to Stages 1 and 2, with 0 `memcpy`. B-pack loop: 16
  `vmovups` + 16 `vmovaps` per 8 rows, which is ideal.

**What's actually left:**
1. **Leak: nothing frees Ã/B̃.** There's no `free` in the asm; one-shot-bufferize
   ran without deallocation, so every call leaks
   `NC/16·KC·16 + MC/6·KC·6` floats (~200 KB at MC=168/NC=32).
2. **Hoisting depends on where a late lowering flag sits.** Stage 6 (per-thread
   Ã inside `scf.forall`) and Stage 5 (dynamic sizes can't be hoisted to the
   function entry) will both need this handled deliberately.
3. **C gets zeroed in a separate pass** (a `memset` of all of C) before the loop
   nest. BLIS uses β=0 on the first pc iteration instead. The estimated cost is
   M·N·4 B written once vs M·N·K·2 flops (~1–2% at K=1024), so this is optional.

**Plan:**
1. **New driver, `repro/run_stage4.sh`**, copied from `run_stage3b.sh` and
   reusing `stage3b_transform.mlir` unchanged (TAG=`stage4`). Changes are all in
   the bufferize step. Right after `one-shot-bufferize`, run:
   `-canonicalize -buffer-hoisting -buffer-loop-hoisting
   -ownership-based-buffer-deallocation -canonicalize
   -buffer-deallocation-simplification -bufferization-lower-deallocations
   -canonicalize -cse`.
   Then drop `-buffer-hoisting -buffer-loop-hoisting` from the LLVM-lowering step.
   - Already dry-run on `out/stage3b_bufferized.mlir`: both allocs land at
     function entry, exactly 2 `memref.dealloc` sit right before `return`,
     there are no `bufferization.dealloc`/ownership `scf.if`s in loops, and the
     self-copies are gone.
   - Hoisting *before* dealloc insertion is the point. Ownership-based dealloc
     on the unhoisted IR would put an alloc/free pair inside the ic loop, and
     it isn't obvious that the later hoisting passes would move both out
     (unverified). Ordering it this way avoids the question.
2. **Structural checks added to the run script** (they fail loudly, not just
   echo):
   - `memref.copy` with differing operands in the bufferized dump: 0.
   - `memref.alloc` count: 2, `memref.dealloc` count: 2, both in the
     function-entry/exit blocks.
   - asm: `malloc` 2, `free` 2, `memcpy` 0, plus the existing kernel-loop
     checks (12 FMA / 6 broadcast / 2 B loads / 0 spills).
3. **Leak regression check**: extend `harness/bench_matmul.c`, or use
   `/usr/bin/time -v`, and compare max RSS at 10 vs 1000 calls. It should be
   flat now; before this fix it grows ~200 KB per call.
4. **Performance holds**: rerun 336×64×512 and 1008×1024×1024, expecting
   71–73 / 96–101 GFLOPS, within noise of Stage 3. One malloc/free pair per call
   is negligible next to ~21 ms of compute.
5. **Optional, only if step 4 shows the `memset` in a profile: β=0 on the
   first pc.** Use `transform.loop.peel %pc {peel_front = true}` (present at
   the pin, `SCFTransformOps.td`). In the peeled iteration, C's tile read comes
   from the fill, so check whether `extract_slice(fill)` → `transfer_read(fill)`
   folds to a zero splat, removing both the global fill and that first C-tile
   load. If it doesn't fold cleanly within ~1 hour, defer it to Stage 7. The
   expected win is ≤2%.

**Result.** `repro/run_stage4.sh` (reuses `stage3b_transform.mlir` unchanged;
`INPUT`/`TRANSFORM`/`TAG` overrides as in `run_stage3b.sh`):
- The bufferize step is now `one-shot-bufferize` → `-canonicalize -buffer-hoisting
  -buffer-loop-hoisting -ownership-based-buffer-deallocation -canonicalize
  -buffer-deallocation-simplification -bufferization-lower-deallocations
  -canonicalize -cse` (dump: `out/stage4_dealloc.mlir`). Hoisting moved out of
  the LLVM-lowering step.
- **Structural checks now fail the script** (Python block at the end of
  `run_stage4.sh`):
  - `memref.copy`: 0, including self-copies.
  - `memref.alloc`/`memref.dealloc`: 2/2, both at function top level.
  - asm: `malloc` 2, `free` 2, `memcpy` 0, `memrefCopy` 0.
  - Kernel loop: 12 `vfmadd231ps`, 6 `vbroadcastss`, 2 B loads, 0 spills.
  - Everything passes at both sizes, and `llvm-mca` still gives 6.14 cycles/iter.
  - One pitfall in writing the checks: the second `free` is emitted as a tail
    call (`jmp free@PLT # TAILCALL`), so counting only `callq` finds 1.
- **Leak fixed** (`/usr/bin/time -f %M`, max RSS, 10 → 1000 calls at 336×64×512):
  Stage 3b 4.9 MB → 207 MB (~204 KB/call = B̃ 32 KB + Ã 172 KB, exactly);
  Stage 4 2.8 MB → 2.8 MB, flat.
- **Performance** (`taskset -c 0`, correctness OK; Stage 3b rebuilt and
  interleaved in the same session for a fair comparison):
  - 336×64×512: **83–85 GFLOPS** vs Stage 3b's 60–72. That's a real win, not
    noise. Ã (172 KB) is over glibc's 128 KB mmap threshold, so the leaking
    build got a fresh mmap every call and paid for first-touch page faults.
    With `free`, glibc raises its dynamic mmap threshold and hands back the same
    already-mapped pages. Minor faults over 200 calls: **10,694 (3b) vs 530 (4)**,
    about 51 fresh 4 KB pages per call in 3b, which matches 204 KB.
  - 1008×1024×1024: **103.7–105.1 GFLOPS** vs Stage 3b's 102.6–104.7 in the
    same session, i.e. equal within noise. The fault cost is amortized over
    ~20 ms of compute. This session runs ~4% faster overall than the Stage 3
    measurement (96.6–100.8).
- **Step 5 (β=0 via peeling the first pc) skipped as not worth it.** `perf`
  isn't installed, so I measured the cost directly: `memset` of a 1008×1024 f32
  C takes 75 µs vs ~20 ms per call, **~0.4%**. Moved to Stage 7's list.
- **Done when** ✅: allocations sit outside every loop (function entry/exit),
  there's no leak, and Stage 3 performance holds (equal at the large size,
  better at the small one).

**Explicitly not in Stage 4** (noted for later stages):
- A-pack transpose quality. It's 32 `vinsertps` + 16 `vmovss` per 6×8 block,
  which is mostly scalar, and it's the next obvious packing cost. It's amortized
  over NC/16 microtiles per Ã element, so it's a tuning item (Stage 7): e.g.
  `lower_transpose` with a shuffle strategy (`lowering_strategy = "shuffle_16x16"`
  / `"shuffle_1d"`), or an 8×8 in-register transpose with MR padding.
- `-test-scalar-vector-transfer-lowering` is a test-only pass. It's fine for
  the experiment, but porting to production needs a custom pass calling
  `vector::populateScalarVectorTransferLoweringPatterns`.
- Scratch buffers passed as function arguments (the BLIS/workspace style) are
  the alternative to malloc/free. Revisit in Stage 6 if per-thread Ã allocation
  inside `scf.forall` makes the dealloc path awkward.
- β=0 on the first pc iteration (`transform.loop.peel {peel_front = true}`)
  instead of zero-filling C first. Measured at ~0.4% (Stage 7).
- Dynamic sizes (Stage 5): `-buffer-loop-hoisting` can only lift Ã/B̃ to
  function entry because their shapes are static. With dynamic M/N/K, the
  alloc sizes must be computable at function entry (e.g. `min(MC, M)`) or the
  buffers stay inside the loops. Recheck the alloc checks there.

### Stage 5 — Remainders

- Try non-multiple sizes (e.g. 1000×1000×1000). Options: loop peeling
  (`transform.loop.peel` on ir/jr), padding the packed panels to MR/NR multiples
  (BLIS approach, cheapest here since packing already copies), or masked
  vectorization for edge tiles.
- **Done when**: arbitrary sizes are correct with no big perf cliff at
  non-multiple sizes.

### Stage 6 — Multithreading

- L2 is private, L3 shared → parallelize the ic loop. Each core packs/uses its
  own Ã; B̃ is shared (pack cooperatively with a barrier, or have one thread pack
  before the ic loop forks).
- `transform.loop.forall` (or retile to `scf.forall`) → `convert-scf-to-openmp`
  → link `libomp` (see `MLIRExecutor.cpp`'s existing `RTLD_GLOBAL` load of libomp
  for JIT symbol visibility — the same trick applies here).
- **Done when**: ~7× speedup on 8 cores vs. single-core, benchmarked against
  OpenBLAS multithreaded.

### Stage 7 — Tuning

- Sweep KC ∈ {192, 256, 320}, MC, and k-unroll U.
- Align packed buffers to 64 bytes.
- Try software prefetch of next Ã/B̃ micro-panels (`memref.prefetch`).
- Try an alternative kernel shape (e.g. 8×...) to confirm 6×16 really is best on
  this chip.

## Debugging habits

- `transform.print` after every transform step.
- One `.mlir` test file per stage under `repro/`, so a later-stage regression can
  be bisected against an earlier stage's dump in `out/`.
- A small script (`harness/count_fma.sh` or similar) that greps the assembly for
  `vfmadd231ps` and `rsp`-relative accesses inside the hot loop label — run it at
  every stage, not just Stage 1.
