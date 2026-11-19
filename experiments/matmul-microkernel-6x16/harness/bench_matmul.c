// Stage 0 harness: loads a shared library exporting a matmul function
// lowered by `mlir-edsl-opt -cpu-pipeline` (the production
// buildCPUPipeline in cpp/src/MLIRLowering.cpp) and calls it using the
// *flattened scalar* memref descriptor calling convention -- the same
// convention the production JIT path uses from Python
// (mlir_edsl/backend.py: _memref_descriptor_c_types + ctypes.CFUNCTYPE),
// confirmed by inspecting the lowered llvm.func signature:
//
//   llvm.func @matmul_baseline(ptr, ptr, i64, i64, i64, i64, i64,   // A
//                               ptr, ptr, i64, i64, i64, i64, i64,   // B
//                               ptr, ptr, i64, i64, i64, i64, i64)   // C
//
// i.e. per memref arg: allocated_ptr, aligned_ptr, offset, size0, size1,
// stride0, stride1 (rank 2, row-major identity layout).
//
// Usage: ./bench_matmul <path-to-.so> <symbol> M N K [repeats]
// Run under `taskset -c <core>` to pin to one core for a stable number.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*MatmulFn)(
    void *a_alloc, void *a_align, long a_off, long a_s0, long a_s1, long a_st0, long a_st1,
    void *b_alloc, void *b_align, long b_off, long b_s0, long b_s1, long b_st0, long b_st1,
    void *c_alloc, void *c_align, long c_off, long c_s0, long c_s1, long c_st0, long c_st1);

static void naive_matmul(const float *A, const float *B, float *C, long M, long N, long K) {
  for (long i = 0; i < M; i++) {
    for (long j = 0; j < N; j++) {
      float acc = 0.0f;
      for (long k = 0; k < K; k++) {
        acc += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = acc;
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
  const char *so_path = argv[1];
  const char *symbol = argv[2];
  long M = atol(argv[3]);
  long N = atol(argv[4]);
  long K = atol(argv[5]);
  int repeats = argc > 6 ? atoi(argv[6]) : 20;

  void *handle = dlopen(so_path, RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    fprintf(stderr, "dlopen failed: %s\n", dlerror());
    return 1;
  }
  dlerror();
  MatmulFn fn = (MatmulFn)dlsym(handle, symbol);
  const char *err = dlerror();
  if (err) {
    fprintf(stderr, "dlsym failed: %s\n", err);
    return 1;
  }

  float *A = malloc(sizeof(float) * M * K);
  float *B = malloc(sizeof(float) * K * N);
  float *C = malloc(sizeof(float) * M * N);
  float *C_ref = malloc(sizeof(float) * M * N);
  if (!A || !B || !C || !C_ref) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  srand(0);
  for (long i = 0; i < M * K; i++) A[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
  for (long i = 0; i < K * N; i++) B[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;

  naive_matmul(A, B, C_ref, M, N, K);

  // Warm up + correctness check on the first call.
  memset(C, 0, sizeof(float) * M * N);
  fn(A, A, 0, M, K, K, 1, B, B, 0, K, N, N, 1, C, C, 0, M, N, N, 1);

  double max_abs_err = 0.0;
  for (long i = 0; i < M * N; i++) {
    double err_i = C[i] - C_ref[i];
    if (err_i < 0) err_i = -err_i;
    if (err_i > max_abs_err) max_abs_err = err_i;
  }
  // f32 accumulation over K terms; scale tolerance with K and value range.
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
    fn(A, A, 0, M, K, K, 1, B, B, 0, K, N, N, 1, C, C, 0, M, N, N, 1);
    double t1 = now_seconds();
    double dt = t1 - t0;
    if (dt < best_seconds) best_seconds = dt;
  }

  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = flops / best_seconds / 1e9;
  printf("M=%ld N=%ld K=%ld best_of=%d best_time=%.6fs GFLOPS=%.2f\n",
         M, N, K, repeats, best_seconds, gflops);

  free(A);
  free(B);
  free(C);
  free(C_ref);
  dlclose(handle);
  return 0;
}
