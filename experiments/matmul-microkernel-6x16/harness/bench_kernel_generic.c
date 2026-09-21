// Generic microkernel micro-benchmark: same idea as bench_kernel.c, but MR,
// NR and KC come from the command line so one binary can bench the whole
// MRxNR sweep. Panels are 64B-aligned and stay hot in L1, so this measures
// the kernel loop itself, not the memory system.
//
// Usage: ./bench_kernel_generic <path-to-.so> <symbol> MR NR KC reps
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

static float *alloc_panel(long n) {
  // round the byte size up to a multiple of the alignment for aligned_alloc
  long bytes = ((n * (long)sizeof(float) + 63) / 64) * 64;
  float *p = (float *)aligned_alloc(64, (size_t)bytes);
  if (!p) { fprintf(stderr, "aligned_alloc failed\n"); exit(1); }
  return p;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr, "usage: %s <path-to-.so> <symbol> MR NR KC reps\n", argv[0]);
    return 1;
  }
  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  KernelFn fn = (KernelFn)dlsym(handle, argv[2]);
  if (!fn) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
  long MR = atol(argv[3]), NR = atol(argv[4]), KC = atol(argv[5]);
  long reps = atol(argv[6]);

  float *A = alloc_panel(KC * MR), *B = alloc_panel(KC * NR), *C = alloc_panel(MR * NR);
  for (long i = 0; i < KC * MR; i++) A[i] = (float)(i % 7) * 0.1f;
  for (long i = 0; i < KC * NR; i++) B[i] = (float)(i % 5) * 0.1f;
  for (long i = 0; i < MR * NR; i++) C[i] = 0.0f;

  // warmup
  for (long r = 0; r < 100; r++)
    fn(A, A, 0, KC, MR, MR, 1, B, B, 0, KC, NR, NR, 1, C, C, 0, MR, NR, NR, 1);

  double t0 = now_seconds();
  for (long r = 0; r < reps; r++) {
    fn(A, A, 0, KC, MR, MR, 1, B, B, 0, KC, NR, NR, 1, C, C, 0, MR, NR, NR, 1);
  }
  double t1 = now_seconds();

  double total_flops = 2.0 * (double)MR * (double)NR * (double)KC * (double)reps;
  double gflops = total_flops / (t1 - t0) / 1e9;
  printf("%.2f %.4f\n", gflops, C[0]);
  free(A); free(B); free(C);
  return 0;
}
