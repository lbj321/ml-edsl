// Stage 1 kernel micro-benchmark: calls the isolated 6x16xKC microkernel
// in a tight loop (recomputing the same panel repeatedly, so it stays hot
// in L1) and measures sustained GFLOPS, to check against the llvm-mca
// FMA-port-bound estimate and the ">=90% of single-core peak" done-when
// bar for Stage 1.
//
// Usage: ./bench_kernel <path-to-.so> <symbol> KC reps
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef void (*KernelFn)(
    void *a_alloc, void *a_align, long a_off, long a_s0, long a_s1, long a_st0, long a_st1,
    void *b_alloc, void *b_align, long b_off, long b_s0, long b_s1, long b_st0, long b_st1,
    void *c_alloc, void *c_align, long c_off, long c_s0, long c_s1, long c_st0, long c_st1);

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <path-to-.so> <symbol> KC reps\n", argv[0]);
    return 1;
  }
  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  KernelFn fn = (KernelFn)dlsym(handle, argv[2]);
  if (!fn) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
  long KC = atol(argv[3]);
  long reps = atol(argv[4]);

  float A[256 * 6], B[256 * 16], C[6 * 16];
  for (int i = 0; i < 256 * 6; i++) A[i] = (float)(i % 7) * 0.1f;
  for (int i = 0; i < 256 * 16; i++) B[i] = (float)(i % 5) * 0.1f;

  double t0 = now_seconds();
  for (long r = 0; r < reps; r++) {
    fn(A, A, 0, KC, 6, 6, 1, B, B, 0, KC, 16, 16, 1, C, C, 0, 6, 16, 16, 1);
  }
  double t1 = now_seconds();

  double flops_per_call = 2.0 * 6.0 * 16.0 * (double)KC;
  double total_flops = flops_per_call * (double)reps;
  double gflops = total_flops / (t1 - t0) / 1e9;
  printf("KC=%ld reps=%ld time=%.6fs GFLOPS=%.2f C[0][0]=%.4f\n",
         KC, reps, t1 - t0, gflops, C[0]);
  return 0;
}
