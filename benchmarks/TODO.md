# Benchmark data-gathering TODO

The unit of independence is one worker **process**, not one timed call.
Samples inside a process share memory layout, JIT code placement, thread
placement and frequency state, so their spread understates run-to-run
variance.

## Core

### Pilot

- [x] **1** `--dump-samples PATH` on all three workers: every raw sample in
      timed order, plus warmup samples.
- [x] **2** `pilot` preset: {64³, 1024³, 2048³} × {1T, 8T} × MKL + EDSL
      × 10 rounds, each a fresh process, samples dumped by default.
- [x] **3** Analyse it. Findings from
      `results/20261005-120055-pilot.jsonl` (EDSL @ 7ad3bbe):
  - **Between-process spread dominates.** sd of round medians 1–3%
    (7% for EDSL 2048³ 8T); one process's median is good to 0.01–0.5%,
    3–535× tighter. More samples per process buy nothing; rounds do.
    ~2% sd → ±0.9% at 5 rounds, ±0.6% at 10. 64³ needs far fewer than
    `MAX_REPEATS` samples.
  - **Order effect at 8T.** MKL 1024³ 8T alternates exactly with
    position: 2–4% fast when it runs first, 2–4% slow when it runs right
    after EDSL's 8T run; EDSL mirrors it. Likely thermal carry-over.
    Rotation cancels it on average, but every `rounds: 1` preset is biased
    3–5% against the second backend at 8T.
  - **MKL 2048³ 8T shows the order effect too**, smaller: every second-place
    round ≥ 1.003 of the median, every first-place round ≤ 0.997.
  - **EDSL 2048³ 8T has a slow mode.** 4 of 10 rounds 7–18% slow. Three
    ran second, but the worst (1.176) ran first, so it is more than the
    order effect. `cpu_per_wall` normal (7.6); MKL doesn't show it. Working
    set (3 × 16 MB) far exceeds L3; page placement / THP is the first
    suspect.
  - **Ramp is long at 8T, absent at 1T.** 8T: first 0.25 s ~10% faster
    than the last window, still ~1% faster at 1.5–2 s, so the 2 s window's
    median sits mid-ramp and steady state isn't confirmed. 1T: flat within
    1% (MKL 2048³ 1T is 5% *slower* in the first 0.25 s: cold start).
- [x] **3a** 15 s runs at 1024³ 8T, one per backend. The ramp settles
      after ~5–6 s (MKL) and ~8 s (EDSL). Against steady state, the median
      of the first 2 s is 12% fast for MKL and 7% for EDSL, so 8T EDSL/MKL
      ratios likely understate EDSL by ~5% (one run each: tentative).
      MKL's first ~10 calls were ~2× slow, so its budget estimate covered
      only 7.9 of 15 s.
- [ ] **3b** *(deferred: EDSL perf work, not harness)* Investigate the
      EDSL 2048³ 8T slow mode (compare fast vs. slow rounds' samples; try
      THP on/off).
- [x] **4** Decision: **keep the short warmup and accept the ramp bias.**
      The order effect is variance, handled by rounds + reported spread
      (8, 10). The ramp is bias that rounds can't remove, but it mostly
      cancels in build-vs-build comparisons and is small next to the
      EDSL/MKL gap. So 8T numbers mean "over the first ~2 s", not steady
      state; never compare runs with different `min_time`. Revisit for
      headline 8T numbers (measure those few points with `--min-time 15`)
      or for a change that alters how hard the kernel loads the cores.
- [x] **4a** Fix the `repeats_for_budget` docstring: `MAX_REPEATS` wins for
      µs calls, so "at least `min_time`" is false there.

### Harness

- [ ] **5** Consistent pinning for every backend: `taskset` from the
      orchestrator, plus `OMP_PROC_BIND=close` `OMP_PLACES=cores` for the
      OpenMP backends. Drop the JAX-only `sched_setaffinity`.
- [ ] **6** Canary (MKL 1024³ 1T) at the start and end of each session;
      record its drift.

### Reporting

- [x] **8** `report()`: show between-round min–max next to each median.
- [x] **9** Ratios to MKL computed per round (paired), then summarised —
      not a ratio of medians.
- [x] **10** Rounds a multiple of the backend count, ≥ 4 (all presets have
      8T runs; order effect + ~2% sd). With unbalanced positions, e.g.
      5 rounds over 2 backends, the median lands in the majority position
      and keeps the order effect (pilot: ~3% at MKL 1024³ 8T).
      `run_matrix.py` warns on unbalanced rounds. Each preset keeps its
      `min_time` (see 4); `MAX_REPEATS` 100k → 10k. Presets take ~5–14
      min.

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

- Compare across sessions only via same-session ratios to MKL. This is
  also how to judge an EDSL change: run the preset, change and rebuild,
  run it again, compare the paired EDSL/MKL ratios (one EDSL build only).
- Discard a session whose canary drifted.
- Rounds a multiple of the backend count: ≥ 4 for any claim; ~8–10 for
  differences under 5%.

## Later

- Shuffle the full grid per round.
- Extra env metadata (load average, kernel, power mode).
- Normwise float64 error check (only if reordering float math).
- Automatic spread/drift flags; bootstrap CIs.
- Report p99 only where sample counts support it (small sizes).
