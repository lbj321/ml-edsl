# Benchmark data-gathering TODO

The unit of independence is one worker **process**, not one timed call.
Samples inside a process share memory layout, JIT code placement, thread
placement and frequency state, so their spread understates run-to-run
variance.

## Core

### Pilot

- [x] **1** `--dump-samples PATH` on all three workers: every raw sample in
      timed order, plus warmup samples.
- [ ] **2** `pilot` preset: ~4 points (64³ 1T, 1024³ 1T, 1024³ 8T,
      2048³ 8T) × MKL + EDSL × ~10 rounds, each a fresh process.
- [ ] **3** Analyse it:
  - between-process vs. within-process variance → more rounds or more
    `min_time`?
  - bimodal process medians (thread placement)?
  - clock ramp within a process: how long until steady state?
- [ ] **4** Act on the ramp. `WARMUP` is 5 calls (~0.1 s at large N), so the
      burst phase lands inside the timed window.
  - Well under 0.5 s: keep `repeats_for_budget`; fix its docstring
    (`MAX_REPEATS` wins for µs calls, so "at least `min_time`" is false).
  - Longer: warm up by time (~1–2 s), size samples for a stable median
    only, spend the saved time on rounds.

### Harness

- [ ] **5** Consistent pinning for every backend: `taskset` from the
      orchestrator, plus `OMP_PROC_BIND=close` `OMP_PLACES=cores` for the
      OpenMP backends. Drop the JAX-only `sched_setaffinity`.
- [ ] **6** Canary (MKL 1024³ 1T) at the start and end of each session;
      record its drift.
- [ ] **7** Two EDSL builds as separate backends (e.g. `edsl-base`,
      `edsl-new`, each pointing at its own checkout/`.so`) for
      before/after comparisons of a change.

### Reporting

- [ ] **8** `report()`: show between-round min–max next to each median.
- [ ] **9** Ratios computed per round (paired), then summarised — not a
      ratio of medians. Applies to EDSL vs. MKL and build vs. build.
- [ ] **10** Set `rounds` (≥ 3) and `min_time` per preset from the pilot.

## Presentation

Error bands in every plot come from between-round spread, not
within-process p10–p90.

- [ ] **P1** GFLOP/s vs. size, log2 x-axis, one line per backend, dashed
      peak line; 1T and all-cores versions.
- [ ] **P2** Speedup vs. MKL, log y-axis, reference line at 1.0.
- [ ] **P3** Small-size latency (µs/call, log-log) with an empty-call
      overhead floor.
- [ ] **P4** Thread scaling at one or two large sizes, with ideal-scaling
      line.
- [ ] **P5** Shape coverage: non-square / tall-skinny / GEMV-like (M=1);
      coarse (~5×5) M×N heatmap of EDSL vs. best competitor, diverging
      colormap centred at 1.0.
- [ ] **P6** Summary table: backends × representative sizes, GFLOP/s and
      % of peak, hardware/settings in the caption.
- [ ] **P7** Same color per backend across all figures; equal line weights
      for analysis plots.
- [ ] **P8** Peak GFLOP/s formula and assumed clock in meta (9700KF AVX2:
      2 FMA × 8 f32 × 2 = 32 flop/cycle/core), needed by P1 and P6.

Cap square sizes at 4096 (8192 only at 8T if at all): one 8192³ call is
~1.1 TFLOP, ~10 s single-threaded, and the EDSL compiles per static shape.

## Rules

- Compare across sessions only via same-session ratios to MKL.
- Discard a session whose canary drifted.
- ≥ 5 rounds for any claim; ~10 for differences under 5%.

## Later

- Shuffle the full grid per round.
- Extra env metadata (load average, kernel, power mode).
- Normwise float64 error check (only if reordering float math).
- Automatic spread/drift flags; bootstrap CIs.
- Report p99 only where sample counts support it (small sizes).
