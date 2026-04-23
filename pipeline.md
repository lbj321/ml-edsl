# Lowering Pipeline Reference

Source of truth: `cpp/src/MLIRLowering.cpp`.  
Both pipelines live in `MLIRLowering` and are built once at construction time.

---

## Shared prologue

Both CPU and GPU pipelines open with the same five steps on the tensor-level IR:

| # | Pass | Effect |
|---|------|--------|
| 1 | `linalg-fuse-elementwise-ops` | Fuse adjacent elementwise linalg ops before bufferization so DCE keeps the chain live |
| 2 | `one-shot-bufferize` (`bufferizeFunctionBoundaries=true`, `IdentityLayoutMap`) | Tensor ops → memref ops; function boundaries become plain `memref<NxT>` to match Python descriptors |
| 3 | `ownership-based-buffer-deallocation` | Insert `bufferization.dealloc` for buffers created during bufferization |
| 4 | `canonicalize` | Fold constants, erase dead ops |
| 5 | `buffer-deallocation-simplification` + `lower-deallocations` | Simplify and lower `bufferization.dealloc` → `memref.dealloc` |

**CPU only, inserted between steps 1 and 2:** `TensorReturnToOutParamPass` — rewrites tensor-returning functions to `void` + writable out-parameter so bufferization can write results in-place into the caller's buffer without a `memref.copy`.  GPU uses a different ownership model (explicit device alloc) so this pass is absent there.

---

## CPU pipeline (`addConversionPasses`)

Runs on the original `ModuleOp` in-place.  Output: LLVM IR with `omp.parallel` regions.

```
[shared prologue — includes TensorReturnToOutParamPass]

── tiling & parallelism ──
LinalgMatmulParallelTilingPass   64×64 outer tile, K=untiled → scf.forall (no mapping attr)
canonicalize
forall-to-parallel               scf.forall → scf.parallel  (one body per CPU thread)
LinalgMatmulTilingPass           8×8×8 inner tile → scf.for (sets up 8-wide vector lanes)
canonicalize

── vectorization ──
LinalgMatmulToContractPass       static 8×8 linalg.matmul → vector.contract (m,k)×(k,n)→(m,n)
LinalgVectorizationPass          remaining linalg structured ops → vector dialect
canonicalize
VectorCleanupPass                fuse mulf + multi_reduction → vector.contract
VectorContractToOuterProductPass vector.contract → vector.outerproduct on rank-1 slices
                                   (must precede convert-vector-to-scf)
convert-linalg-to-loops          fallback: any remaining linalg ops → scf.for

── vector lowering ──
lower-vector-multi-reduction     vector.multi_reduction → vector.reduce
convert-vector-to-scf            complex transfer ops (broadcasts, permutation maps) → SCF
convert-vector-to-llvm           vector ops → LLVM intrinsics  (x86Vector=true → AVX/FMA)
ub-to-llvm                       ub.poison (from VectorToSCF boundary padding) → LLVM

── parallelism → OpenMP ──
scf-to-control-flow              inner scf.for → CF  [MUST run before OMP conversion;
                                   scf.for cannot be lowered once inside omp.loop_nest]
convert-scf-to-openmp            outer scf.parallel → omp.parallel + omp.wsloop

── final lowering ──
expand-strided-metadata          memref.subview (dynamic offsets from tiling) → arith/affine
lower-affine                     affine.apply → arith ops
arith-to-llvm
finalize-memref-to-llvm
cf-to-llvm
func-to-llvm
convert-openmp-to-llvm
reconcile-unrealized-casts
```

**Tile sizes**

| Pass | Tile (M × N × K) | Loop type | Purpose |
|------|-----------------|-----------|---------|
| `LinalgMatmulParallelTilingPass` | 64 × 64 × 0 (K untiled) | `scf.forall` → `scf.parallel` | One tile per CPU thread; K stays as a full loop inside each thread |
| `LinalgMatmulTilingPass` | 8 × 8 × 8 | `scf.for` | Sets up 8-wide lanes for `vector.contract` / AVX |

---

## GPU pipeline (`lowerToGPUModule`)

Runs on a **cloned** `ModuleOp`.  Two separate `PassManager` runs.  Output: PTX string(s) + `GPULoweredModule` kernel descriptor.

### Phase 1 — outlining (`addGPUPreOutliningPasses`)

```
[shared prologue — no TensorReturnToOutParamPass]

── GPU tiling ──
LinalgGPUMatmulTilingPass        32×32 outer tile, K=untiled → scf.forall (no mapping attr)
canonicalize
forall-to-parallel               scf.forall → scf.parallel

── GPU mapping ──
convert-linalg-to-parallel-loops remaining linalg ops → scf.parallel
gpu-map-parallel-loops           annotate scf.parallel with blockIdx/threadIdx dimensions
                                   [nested inside func.func — OperationPass constraint]
convert-parallel-loop-to-gpu     scf.parallel → gpu.launch
gpu-kernel-outlining             gpu.launch → gpu.launch_func + outlined gpu.func in gpu.module
canonicalize
```

After phase 1, `analyzeKernelLaunches` walks the `gpu.launch_func` ops and records each kernel's grid/block dimensions and argument layout (input memrefs, output memref, scalars) into `GPULoweredModule`.  This snapshot is taken **before** phase 2 destroys the high-level type info.

**Tile size**

| Pass | Tile (M × N × K) | Loop type | Rationale |
|------|-----------------|-----------|-----------|
| `LinalgGPUMatmulTilingPass` | 32 × 32 × 0 | `scf.forall` → `scf.parallel` | 32×32 = 1024 threads/block (CUDA max); K untiled so each thread handles the full reduction |

### Phase 2 — NVVM lowering (`addGPUNVVMPasses`)

```
convert-gpu-ops-to-nvvm          [nested in gpu.module] gpu.func → nvvm dialect
scf-to-control-flow              host-side scf.for → CF
expand-strided-metadata
lower-affine
arith-to-llvm
finalize-memref-to-llvm          recurses into gpu.module contents
cf-to-llvm
reconcile-unrealized-casts
```

### PTX codegen (`gpuModuleToPTX`)

Each `gpu.module` is cloned into a plain `builtin.module`, translated to LLVM IR, then compiled via the NVPTX backend:

- Target triple: `nvptx64-nvidia-cuda`
- CPU: `sm_75` (RTX 2070, Turing)
- Features: `+ptx64`

---

## Side-by-side comparison

| Stage | CPU | GPU |
|-------|-----|-----|
| Entry IR | tensor dialect (linalg.matmul) | same |
| Shared prologue | yes (+ `TensorReturnToOutParamPass`) | yes (no out-param pass) |
| Outer tiling | 64×64, `scf.forall` → `scf.parallel` | 32×32, `scf.forall` → `scf.parallel` |
| Inner tiling | 8×8×8 `scf.for` for vectorization | none |
| Vectorization | `vector.contract` → `outerproduct` → LLVM AVX | none |
| Parallelism target | `omp.parallel` + `omp.wsloop` | `gpu.launch` → PTX |
| Final output | LLVM IR (JIT via LLJIT) | PTX image (loaded via `cuModuleLoadData`) |
| Module ownership | in-place | cloned copy |
| Pipeline runs | 1 | 2 (outlining + NVVM) |

---

## Pass ordering constraints

- **`scf-to-control-flow` before `convert-scf-to-openmp`** (CPU): `scf.for` cannot be lowered to multi-block CFG once it is nested inside `omp.loop_nest` due to OMP structural constraints.
- **`expand-strided-metadata` before `lower-affine` before `finalize-memref-to-llvm`** (both): tiling produces `memref.subview` with dynamic offsets that must be decomposed into `affine.apply` before the memref→LLVM conversion runs.
- **`VectorContractToOuterProductPass` before `convert-vector-to-scf`** (CPU): a rank-3 `vector.contract` surviving to `convert-vector-to-scf` expands into 3D transfer reads with broadcast+transpose+alloca loops, defeating vectorization entirely.
- **`LinalgMatmulToContractPass` before `LinalgVectorizationPass`** (CPU): `linalg::vectorize` always produces a 3D double-broadcast form for matmul that `VectorContractToOuterProduct` cannot decompose; the explicit contract pass produces the correct 2D `(m,k)×(k,n)→(m,n)` form.
- **`analyzeKernelLaunches` between phase 1 and phase 2** (GPU): the high-level memref types on `gpu.launch_func` operands are needed to build the `cuLaunchKernel` argument descriptors; phase 2 replaces them with opaque LLVM pointer types.
