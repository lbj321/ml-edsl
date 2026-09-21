#!/usr/bin/env python3
"""Sweep MRxNR register-tile shapes for the packed-panel microkernel.

Generalizes Stage 1 (`repro/run_stage1.sh`, fixed at 6x16) so the same
pipeline can be run for any MR/NR: generate the isolated
`linalg.matmul_transpose_a` on packed KCxMR / KCxNR panels, run the Stage 1
transform + lowering recipe verbatim, then report for each shape:

  - measured GFLOPS (panels hot in L1, so this is the kernel loop alone),
  - llvm-mca cycles per k-iteration vs. the FMA-port floor (MR*NR/8/2),
  - loop-body instruction counts (FMA / broadcast / B loads / spills).

Usage: python3 harness/sweep_kernels.py [--kc 256] [--reps 400000] [MRxNR ...]
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "out" / "sweep"
TRANSFORM = ROOT / "repro" / "stage1_transform.mlir"

MLIR_OPT = os.environ.get("MLIR_OPT", "mlir-opt")
MLIR_TRANSLATE = os.environ.get("MLIR_TRANSLATE", "mlir-translate")
LLVM_OPT = os.environ.get("LLVM_OPT", "opt")
LLVM_LLC = os.environ.get("LLVM_LLC", "llc")
LLVM_MCA = os.environ.get("LLVM_MCA", "llvm-mca")
MCPU = os.environ.get("MCPU", "skylake")

KERNEL_TEMPLATE = """\
func.func @microkernel(%A: memref<{kc}x{mr}xf32>, %B: memref<{kc}x{nr}xf32>, %C: memref<{mr}x{nr}xf32>) {{
  %a = bufferization.to_tensor %A restrict : memref<{kc}x{mr}xf32> to tensor<{kc}x{mr}xf32>
  %b = bufferization.to_tensor %B restrict : memref<{kc}x{nr}xf32> to tensor<{kc}x{nr}xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %init = tensor.empty() : tensor<{mr}x{nr}xf32>
  %filled = linalg.fill ins(%cst : f32) outs(%init : tensor<{mr}x{nr}xf32>) -> tensor<{mr}x{nr}xf32>
  %res = linalg.matmul_transpose_a ins(%a, %b : tensor<{kc}x{mr}xf32>, tensor<{kc}x{nr}xf32>) outs(%filled : tensor<{mr}x{nr}xf32>) -> tensor<{mr}x{nr}xf32>
  bufferization.materialize_in_destination %res in restrict writable %C : (tensor<{mr}x{nr}xf32>, memref<{mr}x{nr}xf32>) -> ()
  return
}}
"""


def run(cmd: list, **kw) -> subprocess.CompletedProcess:
    """Run a command, raising with its stderr attached on failure."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def transform_for(ktile: int) -> Path:
    """Stage 1's transform script with the k tile size swapped in."""
    if ktile == 1:
        return TRANSFORM
    path = OUT / f"transform_k{ktile}.mlir"
    path.write_text(TRANSFORM.read_text().replace("tile_sizes [0, 0, 1]",
                                                  f"tile_sizes [0, 0, {ktile}]"))
    return path


def build_kernel(mr: int, nr: int, kc: int, ktile: int = 1) -> Path:
    """Lower one MRxNR kernel to a shared object; returns the .so path."""
    tag = f"mk_{mr}x{nr}" + (f"_k{ktile}" if ktile != 1 else "")
    src = OUT / f"{tag}.mlir"
    src.write_text(KERNEL_TEMPLATE.format(mr=mr, nr=nr, kc=kc))

    vec = OUT / f"{tag}_vectorized.mlir"
    run([MLIR_OPT, str(src),
         f"-transform-preload-library=transform-library-paths={transform_for(ktile)}",
         "-transform-interpreter", "-canonicalize", "-cse", "-o", str(vec)])

    buf = OUT / f"{tag}_bufferized.mlir"
    run([MLIR_OPT, str(vec), "-eliminate-empty-tensors",
         "-one-shot-bufferize=bufferize-function-boundaries "
         "function-boundary-type-conversion=identity-layout-map",
         "-canonicalize", "-cse", "-o", str(buf)])

    scal = OUT / f"{tag}_scalarized.mlir"
    run([MLIR_OPT, str(buf),
         "-test-scalar-vector-transfer-lowering=allow-multiple-uses",
         "-canonicalize", "-cse", "-o", str(scal)])

    llvm = OUT / f"{tag}_llvm.mlir"
    run([MLIR_OPT, str(scal), "-convert-vector-to-scf",
         "-buffer-hoisting", "-buffer-loop-hoisting", "-canonicalize",
         "-convert-vector-to-llvm=enable-x86vector", "-convert-ub-to-llvm",
         "-convert-scf-to-cf", "-expand-strided-metadata", "-lower-affine",
         "-convert-arith-to-llvm", "-finalize-memref-to-llvm",
         "-convert-cf-to-llvm", "-convert-func-to-llvm",
         "-reconcile-unrealized-casts", "-o", str(llvm)])

    ll, opt_ll = OUT / f"{tag}.ll", OUT / f"{tag}.opt.ll"
    run([MLIR_TRANSLATE, "--mlir-to-llvmir", str(llvm), "-o", str(ll)])
    run([LLVM_OPT, "-passes=default<O3>", f"-mcpu={MCPU}", str(ll), "-S", "-o", str(opt_ll)])
    run([LLVM_LLC, "-O3", f"-mcpu={MCPU}", "-relocation-model=pic", str(opt_ll),
         "-o", str(OUT / f"{tag}.opt.s")])
    obj = OUT / f"{tag}.o"
    run([LLVM_LLC, "-O3", f"-mcpu={MCPU}", "-relocation-model=pic", "-filetype=obj",
         str(opt_ll), "-o", str(obj)])
    so = OUT / f"lib{tag}.so"
    run(["clang", "-shared", "-fPIC", str(obj), "-o", str(so)])
    return so


def loop_body(tag: str) -> str:
    """Return the asm basic block holding the k-loop (the one with most FMAs)."""
    text = (OUT / f"{tag}.opt.s").read_text()
    blocks, current = [], []
    for line in text.splitlines():
        if re.match(r"^(\.LBB\d+_\d+:|# %bb\.\d+:)", line):
            blocks.append(current)
            current = []
            continue
        current.append(line)
    blocks.append(current)
    best = max(blocks, key=lambda b: sum(l.count("vfmadd") for l in b))
    return "\n".join(best) + "\n"


def mca_cycles(body_path: Path) -> float:
    """Cycles per k-iteration from llvm-mca, or nan if it can't analyze."""
    try:
        res = run([LLVM_MCA, f"-mcpu={MCPU}", "-iterations=100", str(body_path)])
    except subprocess.CalledProcessError:
        return float("nan")
    m = re.search(r"Total Cycles:\s+(\d+)", res.stdout)
    return int(m.group(1)) / 100.0 if m else float("nan")


def bench(so: Path, mr: int, nr: int, kc: int, reps: int, bin_path: Path) -> float:
    """Best-of-3 GFLOPS for one kernel, pinned to core 0."""
    best = 0.0
    for _ in range(3):
        res = run(["taskset", "-c", "0", str(bin_path), str(so), "microkernel",
                   str(mr), str(nr), str(kc), str(reps)])
        best = max(best, float(res.stdout.split()[0]))
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("shapes", nargs="*", default=[])
    p.add_argument("--kc", type=int, default=256)
    p.add_argument("--reps", type=int, default=200000)
    args = p.parse_args()

    default_shapes = ["8x8", "6x8", "12x8", "4x16", "6x16", "8x16", "2x16",
                      "2x32", "3x32", "4x24"]
    # "MRxNR" or "MRxNRxKTILE" (k-loop unroll factor; default 1)
    shapes = []
    for s in (args.shapes or default_shapes):
        parts = [int(v) for v in s.split("x")]
        shapes.append((parts[0], parts[1], parts[2] if len(parts) > 2 else 1))

    OUT.mkdir(parents=True, exist_ok=True)
    bin_path = OUT / "bench_kernel_generic"
    run(["clang", "-O2", str(HERE / "bench_kernel_generic.c"), "-o", str(bin_path), "-ldl"])

    print(f"{'kernel':>8} {'acc':>4} {'GFLOPS':>8} {'cyc/it':>7} {'floor':>6} "
          f"{'%peak':>6} {'fma':>4} {'bcst':>5} {'ld':>4} {'spill':>6}")
    rows = []
    for mr, nr, ktile in shapes:
        name = f"{mr}x{nr}" + (f"/k{ktile}" if ktile != 1 else "")
        tag = f"mk_{mr}x{nr}" + (f"_k{ktile}" if ktile != 1 else "")
        try:
            so = build_kernel(mr, nr, args.kc, ktile)
        except subprocess.CalledProcessError as e:
            print(f"{name:<8} BUILD FAILED: {e.stderr.strip().splitlines()[-1][:80]}")
            continue
        body = loop_body(tag)
        body_path = OUT / f"{tag}_loop_body.s"
        body_path.write_text(body)
        fma = body.count("vfmadd")
        bcst = body.count("vbroadcastss")
        loads = len(re.findall(r"vmovu?p[sd]\s+.*\(", body))
        spills = body.count("Spill") + body.count("Reload")
        cyc = mca_cycles(body_path)
        floor = ktile * mr * nr / 8 / 2  # one mca iteration covers ktile k-steps
        gf = bench(so, mr, nr, args.kc, args.reps, bin_path)
        pct = 100.0 * floor / cyc if cyc == cyc and cyc else float("nan")
        rows.append((name, gf))
        print(f"{name:<8} {mr*nr//8:>4} {gf:>8.1f} {cyc:>7.2f} {floor:>6.1f} "
              f"{pct:>5.1f}% {fma:>4} {bcst:>5} {loads:>4} {spills:>6}")

    if rows:
        best = max(rows, key=lambda r: r[1])
        print(f"\nbest: {best[0]} at {best[1]:.1f} GFLOPS")
    return 0


if __name__ == "__main__":
    if shutil.which("taskset") is None:
        print("taskset not found; results will be noisy", file=sys.stderr)
    sys.exit(main())
