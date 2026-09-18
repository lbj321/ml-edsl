// Stage 2 macro-kernel benchmark: same idea as bench_matmul.c, but for the
// K-major packed-panel layout Stage 1/2 use throughout (A is KxM, not MxK)
// instead of bench_matmul.c's plain MxK/KxN/MxN convention -- kept as a
// separate small driver rather than generalizing bench_matmul.c, since the
// two ABIs/layouts are genuinely different things being measured.
//
// Usage: ./bench_macrokernel <path-to-.so> <symbol> M N K [repeats]
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*MacroKernelFn)(
    void *a_alloc, void *a_align, long a_off, long a_s0, long a_s1, long a_st0, long a_st1,
    void *b_alloc, void *b_align, long b_off, long b_s0, long b_s1, long b_st0, long b_st1,
    void *c_alloc, void *c_align, long c_off, long c_s0, long c_s1, long c_st0, long c_st1);

// A is KxM (A[k,m]), B is KxN (B[k,n]), C is MxN (C[m,n]).
static void naive_matmul_ta(const float *A, const float *B, float *C, long M, long N, long K) {
  for (long m = 0; m < M; m++) {
    for (long n = 0; n < N; n++) {
      float acc = 0.0f;
      for (long k = 0; k < K; k++) acc += A[k * M + m] * B[k * N + n];
      C[m * N + n] = acc;
    }
  }
}

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "usage: %s <path-to-.so> <symbol> M N K [repeats]\n", argv[0]);
    return 1;
  }
  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 1; }
  MacroKernelFn fn = (MacroKernelFn)dlsym(handle, argv[2]);
  if (!fn) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 1; }
  long M = atol(argv[3]), N = atol(argv[4]), K = atol(argv[5]);
  int repeats = argc > 6 ? atoi(argv[6]) : 50;

  float *A = malloc(sizeof(float) * K * M);
  float *B = malloc(sizeof(float) * K * N);
  float *C = malloc(sizeof(float) * M * N);
  float *C_ref = malloc(sizeof(float) * M * N);

  srand(1);
  for (long i = 0; i < K * M; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
  for (long i = 0; i < K * N; i++) B[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

  naive_matmul_ta(A, B, C_ref, M, N, K);

  memset(C, 0, sizeof(float) * M * N);
  fn(A, A, 0, K, M, M, 1, B, B, 0, K, N, N, 1, C, C, 0, M, N, N, 1);

  double max_abs_err = 0.0;
  for (long i = 0; i < M * N; i++) {
    double e = C[i] - C_ref[i];
    if (e < 0) e = -e;
    if (e > max_abs_err) max_abs_err = e;
  }
  double tol = 1e-3 * (double)K;
  if (max_abs_err > tol) {
    fprintf(stderr, "CORRECTNESS FAILED: max_abs_err=%g tol=%g\n", max_abs_err, tol);
    return 1;
  }
  printf("correctness OK (max_abs_err=%g, tol=%g)\n", max_abs_err, tol);

  double best_seconds = 1e300;
  for (int r = 0; r < repeats; r++) {
    memset(C, 0, sizeof(float) * M * N);
    double t0 = now_seconds();
    fn(A, A, 0, K, M, M, 1, B, B, 0, K, N, N, 1, C, C, 0, M, N, N, 1);
    double t1 = now_seconds();
    if (t1 - t0 < best_seconds) best_seconds = t1 - t0;
  }

  double gflops = 2.0 * (double)M * (double)N * (double)K / best_seconds / 1e9;
  printf("M=%ld N=%ld K=%ld best_of=%d best_time=%.6fs GFLOPS=%.2f\n",
         M, N, K, repeats, best_seconds, gflops);
  return 0;
}
