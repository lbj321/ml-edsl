#!/usr/bin/env python3
"""Run every worker over a grid of shapes, thread counts and ops.

Usage:
    python3 benchmarks/run_matrix.py [--backends mkl edsl ...]
        [--shapes 1024x1024x1024 64x4096x1024] [--threads 1 8]
        [--ops matmul dense] [--rounds 2] [--min-time 3]
    python3 benchmarks/run_matrix.py --report benchmarks/results/<file>.jsonl

Each point runs in its own worker process, in that backend's env. Rounds
repeat the whole grid with the backend order rotated, so no backend always
runs first. Results are appended to a JSONL file whose first line records
the run settings and the EDSL build; --report prints the tables from one.
Stdlib only, so any python3 can run it.
"""

import argparse
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKERS = REPO_ROOT / "benchmarks" / "workers"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
CONDA_ENVS = Path.home() / "anaconda3" / "envs"
BASELINE = "mkl"

# name -> (interpreter, worker, field of the result line, expected value)
BACKENDS = {
    "openblas": (CONDA_ENVS / "np-openblas/bin/python", "numpy_worker.py",
                 "blas", "openblas"),
    "mkl": (CONDA_ENVS / "np-mkl/bin/python", "numpy_worker.py",
            "blas", "mkl"),
    "blis": (CONDA_ENVS / "np-blis/bin/python", "numpy_worker.py",
             "blas", "blis"),
    "jax": (CONDA_ENVS / "jax/bin/python", "jax_worker.py",
            "framework", "jax"),
    "edsl": (REPO_ROOT / "venv/bin/python", "edsl_worker.py",
             "framework", "edsl"),
}


def parse_args() -> argparse.Namespace:
    """Parse the grid, or the results file to report on."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS),
                        default=list(BACKENDS))
    parser.add_argument("--shapes", nargs="+",
                        default=["1024x1024x1024", "64x4096x1024"],
                        help="MxNxK: X is MxK, W is KxN")
    parser.add_argument("--threads", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--ops", nargs="+", choices=("matmul", "dense"),
                        default=["matmul", "dense"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--min-time", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=None,
                        help="results file (default: results/<timestamp>)")
    parser.add_argument("--report", type=Path, default=None,
                        help="print the tables for an existing results file")
    return parser.parse_args()


def parse_shape(shape: str) -> tuple:
    """'64x4096x1024' -> (64, 4096, 1024)."""
    parts = shape.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"shape {shape!r} is not MxNxK")
    return tuple(int(p) for p in parts)


def git(*args: str) -> str:
    """Run a git command in the repo and return its stripped stdout."""
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def edsl_build_info() -> dict:
    """The checkout and _mlir_backend.so the EDSL worker will use.

    The .so is untracked, so it survives branch switches; warn when it is
    older than the last commit that touched the C++ sources.
    """
    so = REPO_ROOT / "mlir_edsl" / "_mlir_backend.so"
    if not so.is_file():
        raise RuntimeError(f"{so} not found; run ./build.sh")
    so_mtime = so.stat().st_mtime
    cpp_commit_time = int(git("log", "-1", "--format=%ct", "--", "cpp"))
    if so_mtime < cpp_commit_time:
        print(f"warning: {so.name} is older than the last commit touching "
              "cpp/; rebuild with ./build.sh", file=sys.stderr)
    return {
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "so_built": datetime.datetime.fromtimestamp(so_mtime)
                    .isoformat(timespec="seconds"),
        "so_older_than_cpp": so_mtime < cpp_commit_time,
    }


def run_point(backend: str, shape: tuple, threads: int, op: str,
              min_time: float) -> dict:
    """Run one worker process and return its result line, checked."""
    python, worker, field, expected = BACKENDS[backend]
    cmd = [str(python), str(WORKERS / worker), *map(str, shape),
           "--threads", str(threads), "--op", op,
           "--min-time", str(min_time)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"backend": backend, "op": op, "m": shape[0], "n": shape[1],
                "k": shape[2], "threads": threads,
                "error": proc.stderr.strip().splitlines()[-1:]}
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    if result.get(field) != expected:
        raise RuntimeError(f"{backend}: expected {field}={expected!r}, "
                           f"worker reported {result.get(field)!r}")
    return {"backend": backend, **result}


def run(args: argparse.Namespace) -> Path:
    """Run the grid, appending each result to the results file."""
    shapes = [parse_shape(s) for s in args.shapes]
    for backend in args.backends:
        python = BACKENDS[backend][0]
        if not python.is_file():
            raise RuntimeError(f"{backend}: {python} not found; "
                               "run benchmarks/setup_envs.sh")

    out = args.out or RESULTS_DIR / (
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "meta": {
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "backends": args.backends, "shapes": args.shapes,
            "threads": args.threads, "ops": args.ops,
            "rounds": args.rounds, "min_time": args.min_time,
            "edsl": edsl_build_info() if "edsl" in args.backends else None,
        }
    }
    total = (args.rounds * len(shapes) * len(args.threads) * len(args.ops)
             * len(args.backends))
    done = 0
    with out.open("w") as f:
        f.write(json.dumps(meta) + "\n")
        for rnd in range(args.rounds):
            shift = rnd % len(args.backends)
            order = args.backends[shift:] + args.backends[:shift]
            for shape in shapes:
                for threads in args.threads:
                    for op in args.ops:
                        for backend in order:
                            res = run_point(backend, shape, threads, op,
                                            args.min_time)
                            res["round"] = rnd
                            f.write(json.dumps(res) + "\n")
                            f.flush()
                            done += 1
                            print(f"[{done}/{total}] {progress(res)}",
                                  file=sys.stderr)
    return out


def progress(res: dict) -> str:
    """One line describing a finished point."""
    where = (f"{res['backend']:9} {res['op']:6} "
             f"{res['m']}x{res['n']}x{res['k']} t={res['threads']}")
    if "error" in res:
        return f"{where} FAILED: {res['error']}"
    return f"{where} {res['gflops']:7.1f} GF"


def report(path: Path) -> None:
    """Print one table per (op, shape) plus geomean ratios to the baseline."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    meta = lines[0]["meta"]
    results = [r for r in lines[1:] if "error" not in r]
    for r in lines[1:]:
        if "error" in r:
            print(f"FAILED: {progress(r)}")

    # Median over rounds of each round's median time.
    gf = {}
    for r in results:
        key = (r["op"], (r["m"], r["n"], r["k"]), r["backend"], r["threads"])
        gf.setdefault(key, []).append(r["gflops"])
    gf = {key: statistics.median(v) for key, v in gf.items()}

    edsl = meta.get("edsl")
    if edsl:
        print(f"edsl: {edsl['branch']}@{edsl['commit']}"
              f"{' (dirty)' if edsl['dirty'] else ''}, "
              f".so built {edsl['so_built']}"
              f"{' (OLDER THAN cpp/ HEAD)' if edsl['so_older_than_cpp'] else ''}")
    threads = meta["threads"]
    ratios = {}
    for op in meta["ops"]:
        for shape in map(parse_shape, meta["shapes"]):
            m, n, k = shape
            print(f"\n=== {op} {m}x{n}x{k}  (GFLOPS, median of "
                  f"{meta['rounds']} rounds) ===")
            header = f"{'backend':9}" + "".join(
                f"{f'{t}T':>9}{f'/{BASELINE}':>8}" for t in threads)
            if len(threads) > 1:
                header += f"{f'{threads[-1]}T/{threads[0]}T':>9}"
            print(header)
            for backend in meta["backends"]:
                row = f"{backend:9}"
                for t in threads:
                    val = gf.get((op, shape, backend, t))
                    base = gf.get((op, shape, BASELINE, t))
                    if val is None:
                        row += f"{'-':>9}{'-':>8}"
                        continue
                    row += f"{val:9.1f}"
                    if base:
                        ratio = val / base
                        ratios.setdefault((backend, t), []).append(ratio)
                        row += f"{ratio:7.2f}x"
                    else:
                        row += f"{'-':>8}"
                lo = gf.get((op, shape, backend, threads[0]))
                hi = gf.get((op, shape, backend, threads[-1]))
                if len(threads) > 1 and lo and hi:
                    row += f"{hi / lo:8.1f}x"
                print(row)

    if ratios:
        print(f"\n=== geometric mean of GFLOPS / {BASELINE} "
              "over all ops and shapes ===")
        for backend in meta["backends"]:
            cells = []
            for t in threads:
                r = ratios.get((backend, t))
                cells.append(f"{t}T {statistics.geometric_mean(r):.2f}x"
                             if r else f"{t}T -")
            print(f"{backend:9} " + "   ".join(cells))


def main() -> int:
    """Run the grid and report it, or just report an existing file."""
    args = parse_args()
    if args.report:
        report(args.report)
        return 0
    out = run(args)
    print(f"\nresults: {out}", file=sys.stderr)
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
