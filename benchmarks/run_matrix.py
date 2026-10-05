#!/usr/bin/env python3
"""Run every worker over a grid of shapes, thread counts and ops.

Usage:
    python3 benchmarks/run_matrix.py --preset overhead|scaling|padding|epilogue|ml|pilot
    python3 benchmarks/run_matrix.py [--preset ...] [--backends mkl edsl ...]
        [--shapes 1024x1024x1024 64x4096x1024] [--threads 1 2 4 8]
        [--ops matmul dense] [--rounds 2] [--min-time 2]
        [--dump-samples [DIR]]
    python3 benchmarks/run_matrix.py --report benchmarks/results/<file>.jsonl

Each preset answers one question in 5-14 minutes (see PRESETS).
Options given on the command line override the preset's; without a preset,
--backends, --shapes, --threads and --ops are required.

Each point runs in its own worker process, in that backend's env. Rounds
repeat the whole grid with the backend and op orders rotated, so no
backend or op always runs first. Results are appended to a JSONL file
whose first line records the run settings and the EDSL build; --report
prints the tables from one.
Stdlib only, so any python3 can run it.
"""

import argparse
import datetime
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Optional

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


def _square(*sizes: int) -> list:
    return [f"{n}x{n}x{n}" for n in sizes]


# Every preset includes MKL so its ratios have a baseline from the same
# session. Durations on the i7-9700KF dev machine, scaled from 1- or
# 2-round timings. Rounds, not samples, set the noise: round medians vary
# 1-3%, and at 8T a process running right after another 8T run is 3-5%
# slower. Rotation cancels that only when every backend holds every
# position equally often, so rounds are a multiple of the backend count
# (at least 4); otherwise the median lands in the majority position.
PRESETS = {
    # ~5 min. Fixed per-call cost, and which libraries thread small sizes.
    "overhead": {
        "backends": list(BACKENDS), "shapes": _square(8, 16, 32, 64, 128),
        "threads": [1, 8], "ops": ["matmul"], "rounds": 5, "min_time": 1.0,
    },
    # ~14 min. Thread scaling of large square matmuls.
    "scaling": {
        "backends": ["mkl", "openblas", "jax", "edsl"],
        "shapes": _square(256, 512, 1024, 2048),
        "threads": [1, 2, 4, 8], "ops": ["matmul"],
        "rounds": 4, "min_time": 2.0,
    },
    # ~7 min. Cost of extents no cache block divides, against 1024.
    "padding": {
        "backends": ["mkl", "edsl"], "shapes": _square(1000, 1023, 1024),
        "threads": [1, 2, 4, 8], "ops": ["matmul"],
        "rounds": 6, "min_time": 2.0,
    },
    # ~12 min. Dense layer against bare matmul, around the 256 -> 512 break.
    "epilogue": {
        "backends": ["mkl", "jax", "edsl"], "shapes": _square(256, 512, 1024),
        "threads": [1, 8], "ops": ["matmul", "dense"],
        "rounds": 6, "min_time": 2.0,
    },
    # ~6 min. Dense layers at batch 1..256 on a 1024 -> 4096 layer.
    "ml": {
        "backends": ["mkl", "openblas", "jax", "edsl"],
        "shapes": [f"{b}x4096x1024" for b in (1, 16, 64, 256)],
        "threads": [1, 8], "ops": ["dense"], "rounds": 4, "min_time": 2.0,
    },
    # ~7 min. Variance pilot: many rounds of a few points, with raw samples, to
    # compare between-process and within-process spread and to see the
    # clock ramp. Not for comparing backends.
    "pilot": {
        "backends": ["mkl", "edsl"], "shapes": _square(64, 1024, 2048),
        "threads": [1, 8], "ops": ["matmul"], "rounds": 10, "min_time": 2.0,
        "dump_samples": True,
    },
}
GRID_DEFAULTS = {"rounds": 1, "min_time": 2.0}


def parse_args() -> argparse.Namespace:
    """Parse the grid, or the results file to report on."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=list(PRESETS))
    parser.add_argument("--backends", nargs="+", choices=list(BACKENDS))
    parser.add_argument("--shapes", nargs="+",
                        help="MxNxK: X is MxK, W is KxN")
    parser.add_argument("--threads", nargs="+", type=int)
    parser.add_argument("--ops", nargs="+", choices=("matmul", "dense"))
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--min-time", type=float)
    parser.add_argument("--out", type=Path, default=None,
                        help="results file (default: results/<timestamp>)")
    parser.add_argument("--dump-samples", type=Path, default=None,
                        nargs="?", const=True, metavar="DIR",
                        help="write each point's raw durations to a JSON "
                             "file in DIR (default: <results file>-samples/)")
    parser.add_argument("--report", type=Path, default=None,
                        help="print the tables for an existing results file")
    args = parser.parse_args()
    if args.report:
        return args

    defaults = {**GRID_DEFAULTS, **PRESETS.get(args.preset, {})}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    missing = [k for k in ("backends", "shapes", "threads", "ops")
               if getattr(args, k) is None]
    if missing:
        parser.error("give --preset, or all of: "
                     + ", ".join(f"--{k}" for k in missing))
    return args


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
              min_time: float, samples_file: Optional[Path] = None) -> dict:
    """Run one worker process and return its result line, checked."""
    python, worker, field, expected = BACKENDS[backend]
    cmd = [str(python), str(WORKERS / worker), *map(str, shape),
           "--threads", str(threads), "--op", op,
           "--min-time", str(min_time)]
    if samples_file:
        cmd += ["--dump-samples", str(samples_file)]
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
    if args.rounds % len(args.backends):
        print(f"warning: {args.rounds} round(s) over {len(args.backends)} "
              "backends leaves run positions unbalanced; at 8T that biases "
              "medians by up to the 3-5% order effect. Use a multiple of "
              f"{len(args.backends)}.", file=sys.stderr)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = args.out or RESULTS_DIR / f"{stamp}-{args.preset or 'custom'}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dump_samples is True:
        args.dump_samples = out.with_name(f"{out.stem}-samples")
    meta = {
        "meta": {
            "started": datetime.datetime.now().isoformat(timespec="seconds"),
            "preset": args.preset,
            "backends": args.backends, "shapes": args.shapes,
            "threads": args.threads, "ops": args.ops,
            "rounds": args.rounds, "min_time": args.min_time,
            "dump_samples": (str(args.dump_samples) if args.dump_samples
                             else None),
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
            op_shift = rnd % len(args.ops)
            ops = args.ops[op_shift:] + args.ops[:op_shift]
            for shape in shapes:
                for threads in args.threads:
                    for op in ops:
                        for backend in order:
                            samples_file = None
                            if args.dump_samples:
                                samples_file = args.dump_samples / (
                                    f"{backend}-{op}-"
                                    f"{'x'.join(map(str, shape))}"
                                    f"-t{threads}-r{rnd}.json")
                            res = run_point(backend, shape, threads, op,
                                            args.min_time, samples_file)
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


def fmt_time(seconds: float) -> str:
    """Microseconds below 1 ms, milliseconds below 1 s, else seconds."""
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f}ms"
    return f"{seconds:.2f}s"


def ratio_cell(per_round: list, width: int) -> str:
    """Median of per-round ratios, with half their min-max range relative
    to the median when there is more than one round."""
    if not per_round:
        return f"{'-':>{width}}"
    med = statistics.median(per_round)
    if len(per_round) == 1:
        return f"{med:{width - 1}.2f}x"
    half = (max(per_round) - min(per_round)) / 2 / med
    return f"{med:{width - 8}.2f}x ±{half:5.1%}"


def report(path: Path) -> None:
    """Print GFLOPS and time tables per (op, shape), dense/matmul time
    ratios when both ops ran, and geomean ratios to the baseline, paired
    per round."""
    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    meta = lines[0]["meta"]
    for r in lines[1:]:
        if "error" in r:
            print(f"FAILED: {progress(r)}")

    # Median over rounds of each round's median. Rounds are separate
    # processes, so their spread is the run-to-run noise; the spread of
    # samples within one process is far smaller and not reported here.
    gf_rounds, secs, gf_by_round = {}, {}, {}
    for r in lines[1:]:
        if "error" in r:
            continue
        key = (r["op"], (r["m"], r["n"], r["k"]), r["backend"], r["threads"])
        gf_rounds.setdefault(key, []).append(r["gflops"])
        secs.setdefault(key, []).append(r["median_s"])
        gf_by_round[(*key, r["round"])] = r["gflops"]
    gf = {key: statistics.median(v) for key, v in gf_rounds.items()}
    secs = {key: statistics.median(v) for key, v in secs.items()}
    # Half the min-max range over rounds, relative to the median.
    half_range = {key: (max(v) - min(v)) / 2 / gf[key]
                  for key, v in gf_rounds.items() if len(v) > 1}

    print(f"preset: {meta.get('preset') or 'custom'}, {meta['rounds']} "
          f"round(s), min_time {meta['min_time']} s")
    if half_range:
        print("±: half the min-max range over rounds")
    edsl = meta.get("edsl")
    if edsl:
        print(f"edsl: {edsl['branch']}@{edsl['commit']}"
              f"{' (dirty)' if edsl['dirty'] else ''}, "
              f".so built {edsl['so_built']}"
              f"{' (OLDER THAN cpp/ HEAD)' if edsl['so_older_than_cpp'] else ''}")

    threads, backends = meta["threads"], meta["backends"]
    shapes = [parse_shape(s) for s in meta["shapes"]]
    thread_cols = "".join(f"{f'{t}T':>10}" for t in threads)
    speedup = f"{f'{threads[-1]}T/{threads[0]}T':>9}" if len(threads) > 1 else ""

    # Wider GFLOPS cells when there is a spread to show.
    width = 16 if half_range else 10
    gf_cols = "".join(f"{f'{t}T':>{width}}" for t in threads)

    def gf_cell(key: tuple) -> str:
        """Median GFLOPS, with its spread over rounds when there is one."""
        if key not in gf:
            return f"{'-':>{width}}"
        if key not in half_range:
            return f"{gf[key]:{width}.1f}"
        return f"{gf[key]:{width - 7}.1f} ±{half_range[key]:5.1%}"

    for op in meta["ops"]:
        for shape in shapes:
            print(f"\n=== {op} {'x'.join(map(str, shape))} ===")
            print(f"{'GFLOPS':9}{gf_cols}{speedup}")
            for b in backends:
                vals = [gf.get((op, shape, b, t)) for t in threads]
                row = "".join(gf_cell((op, shape, b, t)) for t in threads)
                if speedup and vals[0] and vals[-1]:
                    row += f"{vals[-1] / vals[0]:8.1f}x"
                print(f"{b:9}{row}")
            print(f"{'time':9}{thread_cols}")
            for b in backends:
                vals = [secs.get((op, shape, b, t)) for t in threads]
                print(f"{b:9}" + "".join(
                    f"{fmt_time(v):>10}" if v else f"{'-':>10}"
                    for v in vals))

    if {"matmul", "dense"} <= set(meta["ops"]):
        print("\n=== dense / matmul time (1.00 = epilogue is free) ===")
        print(f"{'shape':16}{'backend':9}{thread_cols}")
        for shape in shapes:
            for b in backends:
                cells = []
                for t in threads:
                    d = secs.get(("dense", shape, b, t))
                    m = secs.get(("matmul", shape, b, t))
                    cells.append(f"{d / m:10.2f}" if d and m else f"{'-':>10}")
                print(f"{'x'.join(map(str, shape)):16}{b:9}" + "".join(cells))

    if BASELINE in backends:
        # Ratios are taken within each round, where both backends ran
        # minutes apart, then summarised over rounds. This cancels drift
        # across the session that dividing the two medians would keep.
        rounds = sorted({key[-1] for key in gf_by_round})
        print(f"\n=== geometric mean of GFLOPS / {BASELINE} over all ops "
              "and shapes, paired per round ===")
        print(f"{'':9}{gf_cols}")
        for b in backends:
            cells = []
            for t in threads:
                per_round = []
                for rnd in rounds:
                    ratios = [gf_by_round[(op, s, b, t, rnd)]
                              / gf_by_round[(op, s, BASELINE, t, rnd)]
                              for op in meta["ops"] for s in shapes
                              if (op, s, b, t, rnd) in gf_by_round
                              and (op, s, BASELINE, t, rnd) in gf_by_round]
                    if ratios:
                        per_round.append(statistics.geometric_mean(ratios))
                cells.append(ratio_cell(per_round, width))
            print(f"{b:9}" + "".join(cells))


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
