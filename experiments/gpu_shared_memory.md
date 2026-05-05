# GPU Shared Memory Tiling

Design notes for adding shared memory tiling to the GPU matmul pipeline.

## Motivation

Benchmark sizes go up to 1024×1024 f32. At that scale A and B are 4 MB each,
well past GPU L2. Without shared memory, each thread block re-reads the same
rows of A and columns of B from global memory K times — O(N³) global reads
instead of O(N²). The current pipeline is memory-bound at these sizes and
likely 10–50× slower than a shared memory implementation.

## Tiling Hierarchy

```
Block tile  [32×32]          workgroup tiling — current pipeline
  └── Thread tile  [4×4]     each thread owns a 4×4 output submatrix
        └── K-strip loop     iterate over K in strips of width K_TILE (e.g. 8)
              └── micro-kernel  innermost: 4×4 outer-product per K-strip step
```

Each level builds on the one above. Thread tiling is a prerequisite for shared
memory — without it each thread owns one output element and there is nothing to
reuse A rows / B columns across.

## Step 1 — Thread-Level Output Tiling (no shared memory)

Each thread computes a 4×4 (or 8×8) output tile instead of one element.

**What it gives:**
- 4 values loaded from a row of A reused across 4 columns of output
- 4 values loaded from a column of B reused across 4 rows of output
- 16 multiply-accumulates from 8 loads → 2× arithmetic intensity vs 1-element-per-thread
- Still reads from global memory; bandwidth not fully solved but meaningful improvement

**In MLIR terms:** a second `scf.forall` (or `scf.for`) inside the block tile
distributing the 32×32 block work across threads, with each thread iterating
over its 4×4 output submatrix in the K loop.

## Step 2 — Shared Memory Loading (on top of thread tiling)

Cooperative load: all threads in the block collectively load a K_TILE-wide strip
of A (shape [32, K_TILE]) and B (shape [K_TILE, 32]) into shared memory, then
each thread computes its 4×4 partial sum reading from shared memory.

```
for k_base in range(0, K, K_TILE):                     # K-strip loop
    cooperative_load(A_tile, A[block_row, k_base:k_base+K_TILE])  # global → shmem
    cooperative_load(B_tile, B[k_base:k_base+K_TILE, block_col])  # global → shmem
    gpu.barrier()                                        # all threads see full tile
    for ti, tj in thread_tile:                           # each thread's 4×4
        for k in range(K_TILE):
            C[ti, tj] += A_tile[ti, k] * B_tile[k, tj]  # shmem reads
    gpu.barrier()                                        # before next load
```

**What it adds over thread tiling alone:**
- A_tile and B_tile are read from global memory once per K-strip, shared across
  all 32×32/4×4 = 64 threads in the block
- Shared memory bandwidth >> global memory bandwidth
- Makes the kernel compute-bound rather than memory-bound at 512×512+

## MLIR Implementation Notes

### Address space

Shared memory uses `#gpu.address_space<workgroup>`:
```mlir
%A_tile = memref.alloca() : memref<32x8xf32, #gpu.address_space<workgroup>>
%B_tile = memref.alloca() : memref<8x32xf32, #gpu.address_space<workgroup>>
```

The `alloca` must be at block scope (shared across threads), not inside the
per-thread compute body.

### Pipeline placement

All of this happens **pre-outlining**, inside `addGPUPreOutliningPasses`.
The shared memory allocas are placed inside the `gpu.launch` body before
`createGpuKernelOutliningPass` runs; outlining correctly moves them into the
`gpu.func`.

### Cooperative loading — the hard part

Distributing the global→shared load across threads requires computing each
thread's slice of the tile:
```mlir
%tid_x = gpu.thread_id x
%tid_y = gpu.thread_id y
// thread (tid_y, tid_x) loads A_tile[tid_y, tid_x] from A[block_row+tid_y, k_base+tid_x]
```

This is not naturally expressed by a `linalg` op — linalg maps one output
element to one logical iteration, but shared memory loading is a collective
operation. Options:
1. Lower the load directly to `memref.load` / `memref.store` with explicit
   thread index arithmetic (straightforward but verbose)
2. Use `vector.transfer_read` into a vector register, then `vector.store` to
   shared memory (cleaner, aligns with the vector pipeline)
3. Use a `linalg.generic` with a thread-mapped indexing map (requires the
   thread forall to be set up first)

### Barrier insertion

Two `gpu.barrier` ops per K-strip iteration:
1. After the cooperative load (before compute) — all threads must see the full
   tile before any thread starts computing
2. After compute (before next load) — all threads must finish reading the tile
   before any thread overwrites it with the next strip

### Boundary handling

K_TILE must evenly divide K, or boundary iterations need masking. Easiest
approach: pad K to the next multiple of K_TILE before the kernel.

## Implementation Order

1. Add thread-level output tiling (`scf.forall` inside block tile, distributing
   32×32 work to 8×8 groups of threads each computing 4×4)
2. Add the K-strip outer loop inside the thread tile
3. Add shared memory allocas at block scope
4. Add cooperative load (memref.load/store with thread index arithmetic)
5. Add `gpu.barrier` ops in the right places
6. Verify correctness on small sizes (16×16, 32×32) before benchmarking
7. Benchmark 256–1024 against current pipeline and cuBLAS reference

## Tile Size Choices

| Parameter | Value | Rationale |
|---|---|---|
| Block tile | 32×32 | current, maps to one thread block |
| Thread tile | 4×4 | 32/4 = 8 → 8×8 = 64 threads per block, good occupancy |
| K_TILE | 8 | A_tile = 32×8×4B = 1KB, B_tile = 8×32×4B = 1KB → 2KB shmem per block, leaves room for multiple blocks |

Shared memory per SM is typically 48–96KB. With 2KB per block, up to 24–48
blocks can reside simultaneously (limited in practice by register pressure too).
