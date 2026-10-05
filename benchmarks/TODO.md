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
- [ ] **3a** One long run (MKL 1024³ 8T, `--min-time 15 --dump-samples`)
      to find where the 8T ramp flattens; sets the warmup length for 4.
- [ ] **3b** Investigate the EDSL 2048³ 8T slow mode (compare fast vs. slow
      rounds' samples; try THP on/off).
- [ ] **4** Ramp is long (from 3), so: warm up by time (length from 3a)
      before timing, size samples for a stable median only, spend the saved
      time on rounds. Should also remove the order effect, since every
      process then starts timing from the same thermal state. Re-run the
      pilot to confirm. Fix the `repeats_for_budget` docstring either way
      (`MAX_REPEATS` wins for µs calls).

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
- [ ] **10** Set `rounds` and `min_time` per preset from the pilot: ≥ 5
      rounds for any preset with 8T runs (order effect + ~2% sd); lower
      `MAX_REPEATS` for small sizes.

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
