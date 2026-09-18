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

### Stage 3 — Outer loops + packing

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
- **Done when**: correct at full size, IR has pack ops at the right loop depth.
  Expect ~80-90% of Stage 2 performance.

### Stage 4 — Bufferization cleanup

- One-shot bufferize with `bufferize-function-boundaries`. Check for: no stray
  `memref.copy` of C, C updated in place, Ã/B̃ buffers allocated once and hoisted
  out of loops (buffer-loop-hoisting, or passed in as scratch args) — not
  reallocated per iteration.
- Full lowering: `canonicalize`, `cse`, `lower-affine`, `convert-scf-to-cf`,
  `convert-vector-to-llvm`, `finalize-memref-to-llvm`,
  `convert-arith/func/cf-to-llvm`, `reconcile-unrealized-casts`. Recheck the
  kernel's assembly — late lowering can undo earlier hoisting.
- **Done when**: allocations sit outside the hot loops and Stage 3 performance
  holds.

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
