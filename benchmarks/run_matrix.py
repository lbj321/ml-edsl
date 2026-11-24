#!/usr/bin/env python3
"""Run every worker over a grid of shapes, thread counts and ops.

Usage:
    python3 benchmarks/run_matrix.py --preset overhead|scaling|large|ml|pilot
    python3 benchmarks/run_matrix.py [--preset ...] [--backends mkl edsl ...]
        [--shapes 1024x1024x1024 64x4096x1024] [--threads 1 2 4 8]
        [--ops matmul dense] [--rounds 2] [--min-time 2]
        [--dump-samples [DIR]]
    python3 benchmarks/run_matrix.py --session report [--out-dir DIR]
        [--backends ...] [--rounds 1] [--min-time 0.1] [--force]
    python3 benchmarks/run_matrix.py --report benchmarks/results/<file>.jsonl

Each preset answers one question in 10-120 minutes (see PRESETS). A
session runs several presets into one folder, one file each, plus a
session.json; --backends, --rounds, --min-time and --dump-samples apply to
all of them. It refuses a dirty tree or a stale .so unless forced.
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
import subprocess
import sys
from pathlib import Path
from typing import Optional

from aggregate import (clock_label, epilogue_ratios, gflops_spread, load,
                       parse_shape, peak_gflops, ratio_spread, secs_spread,
                       trim_count)

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKERS = REPO_ROOT / "benchmarks" / "workers"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
CONDA_ENVS = Path.home() / "anaconda3" / "envs"

# Peak f32 throughput of the i7-9700KF dev machine, recorded in each results
# file so plots can draw it. AVX2: 2 FMA ports x 8 lanes x 2 flop. `ghz` is
# the clock per active core count (index 0 = 1 core). With turbo off in the
# BIOS every core runs at the 3.6 GHz base clock; WSL can't read the clock,
# so this is a stated setting, not a measurement. Files from before have
# `turbo_ghz`, Intel's stock turbo table, instead.
PEAK = {
    "cpu": "i7-9700KF",
    "flop_per_cycle_per_core": 32,
    "clock": "base 3.6 GHz, turbo off in BIOS",
    "ghz": [3.6] * 8,
}

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


# Settings every report preset shares, so all figures show the same
# backends at the same precision. Rounds, not samples, set the noise: round
# medians vary 1-3%, and at 8T a process running right after another 8T run
# is 3-5% slower. Rotation cancels that only when every backend holds every
# position equally often, so rounds are a multiple of the backend count;
# 10 is enough for differences under 5%. Raw samples are kept so any
# statistic can be computed afterwards without re-running.
REPORT = {
    "backends": list(BACKENDS), "rounds": 10, "min_time": 2.0,
    "dump_samples": True,
}

# The report presets split the shapes without overlap: cubes up to 128,
# 256-2048 and from 4096, plus the non-square ML shapes. All extents are
# powers of two. Durations on the i7-9700KF dev machine, scaled from
# earlier runs.
PRESETS = {
    # ~10 min. Fixed per-call cost, and which libraries thread small sizes.
    # MAX_REPEATS, not min_time, sets the sample count at these sizes.
    "overhead": {
        **REPORT, "shapes": _square(8, 16, 32, 64, 128),
        "threads": [1, 8], "ops": ["matmul"],
    },
    # ~80 min. Size and thread scaling, and the cost of the dense epilogue
    # (bias + ReLU), which needs both ops in one file to pair per round.
    "scaling": {
        **REPORT, "shapes": _square(256, 512, 1024, 2048),
        "threads": [1, 2, 4, 8], "ops": ["matmul", "dense"],
    },
    # ~120 min, most of it 8192. Whether EDSL keeps up once the working set
    # is far past L3; calls run long enough to reach the sustained 8T clock.
    # Dense too, since EDSL's epilogue cost is not monotonic in size (~1.0x
    # at 256 and 2048, ~2.5x at 512), so 2048 does not predict 4096+.
    # 1T is left out, since one 8192 call there is ~10 s.
    "large": {
        **REPORT, "shapes": _square(4096, 8192),
        "threads": [8], "ops": ["matmul", "dense"],
    },
    # ~38 min. Batch 1..256 on a 1024 -> 4096 layer: 1T isolates the
    # kernel's code generation, 8T how the work is split across threads.
    "ml": {
        **REPORT, "shapes": [f"{b}x4096x1024" for b in (1, 16, 64, 256)],
        "threads": [1, 8], "ops": ["matmul", "dense"],
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

# name -> presets run one after another into one folder, one file each.
SESSIONS = {
    "report": ["overhead", "scaling", "large", "ml"],
}
# Options a session passes on to every preset; the rest are per preset.
SESSION_OVERRIDES = ("backends", "rounds", "min_time", "dump_samples")


def parse_args() -> argparse.Namespace:
    """Parse the grid, the session, or the results file to report on."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=list(PRESETS))
    parser.add_argument("--session", choices=list(SESSIONS),
                        help="run a list of presets into one folder")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="session folder (default: "
                             "results/<timestamp>-<session>)")
    parser.add_argument("--force", action="store_true",
                        help="run a session on a dirty tree or stale .so")
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

    if args.session:
        if args.preset:
            parser.error("give --session or --preset, not both")
        per_preset = [k for k in ("shapes", "threads", "ops", "out")
                      if getattr(args, k) is not None]
        if per_preset:
            parser.error("a session takes its grids from its presets; drop "
                         + ", ".join(f"--{k}" for k in per_preset))
        return args
    if args.out_dir or args.force:
        parser.error("--out-dir and --force only apply to --session")

    args = resolve(args, args.preset)
    missing = [k for k in ("backends", "shapes", "threads", "ops")
               if getattr(args, k) is None]
    if missing:
        parser.error("give --preset, or all of: "
                     + ", ".join(f"--{k}" for k in missing))
    return args


def resolve(args: argparse.Namespace,
            preset: Optional[str]) -> argparse.Namespace:
    """A copy of `args` with the preset's values filled in wherever the
    command line left them unset."""
    resolved = argparse.Namespace(**vars(args))
    resolved.preset = preset
    defaults = {**GRID_DEFAULTS, **PRESETS.get(preset, {})}
    for key, value in defaults.items():
        if getattr(resolved, key) is None:
            setattr(resolved, key, value)
    return resolved


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


def run(args: argparse.Namespace, edsl: Optional[dict] = None) -> Path:
    """Run the grid, appending each result to the results file. `edsl` is
    build info already taken by a session; otherwise it is read here."""
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
            "preset": args.preset, "session": args.session,
            "backends": args.backends, "shapes": args.shapes,
            "threads": args.threads, "ops": args.ops,
            "rounds": args.rounds, "min_time": args.min_time,
            "dump_samples": (str(args.dump_samples) if args.dump_samples
                             else None),
            "edsl": ((edsl or edsl_build_info())
                     if "edsl" in args.backends else None),
            "peak": PEAK,
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


def checked_build(force: bool) -> dict:
    """EDSL build info for a session, refusing a dirty tree or a stale .so
    unless forced: an hours-long run should not end in an unusable file."""
    info = edsl_build_info()
    problems = []
    if info["dirty"]:
        problems.append("the working tree has uncommitted changes")
    if info["so_older_than_cpp"]:
        problems.append("_mlir_backend.so is older than the last cpp/ commit")
    if problems and not force:
        raise RuntimeError("; ".join(problems)
                           + ". Commit and ./build.sh, or pass --force.")
    for problem in problems:
        print(f"warning: {problem} (--force)", file=sys.stderr)
    return info


def run_session(args: argparse.Namespace) -> list:
    """Run each preset of the session into its own file in one folder,
    keeping session.json current so an interrupted run shows where it
    stopped. Returns the results files."""
    presets = SESSIONS[args.session]
    grids = [resolve(args, preset) for preset in presets]
    edsl = (checked_build(args.force)
            if any("edsl" in g.backends for g in grids) else None)

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.out_dir or RESULTS_DIR / f"{stamp}-{args.session}"
    # Refuses an existing folder rather than mixing two sessions in one.
    out_dir.mkdir(parents=True)
    session = {
        "session": args.session, "presets": presets,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished": None,
        "overrides": {k: (str(v) if isinstance(v, Path) else v)
                      for k in SESSION_OVERRIDES
                      if (v := getattr(args, k)) is not None},
        "edsl": edsl, "completed": [],
    }
    session_file = out_dir / "session.json"
    session_file.write_text(json.dumps(session, indent=2) + "\n")

    outs = []
    for grid in grids:
        grid.out = out_dir / f"{grid.preset}.jsonl"
        if isinstance(grid.dump_samples, Path):
            grid.dump_samples = grid.dump_samples / grid.preset
        print(f"\n=== session {args.session}: preset {grid.preset} ===",
              file=sys.stderr)
        outs.append(run(grid, edsl))
        session["completed"].append(grid.preset)
        session_file.write_text(json.dumps(session, indent=2) + "\n")
    session["finished"] = datetime.datetime.now().isoformat(
        timespec="seconds")
    session_file.write_text(json.dumps(session, indent=2) + "\n")
    return outs


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


def report(path: Path) -> None:
    """Print GFLOPS and time tables per (op, shape), and dense/matmul time
    ratios, paired per round, when both ops ran."""
    res = load(path)
    meta = res.meta
    for r in res.failed:
        print(f"FAILED: {progress(r)}")

    gf_spread = {key: gflops_spread(p.gflops)
                 for key, p in res.points.items()}
    gf = {key: s.mean for key, s in gf_spread.items()}
    secs = {key: secs_spread(p.secs).mean for key, p in res.points.items()}
    half_ci = {key: s.half_ci for key, s in gf_spread.items()
               if len(res.points[key].gflops) > 1}

    print(f"preset: {meta.get('preset') or 'custom'}, {meta['rounds']} "
          f"round(s), min_time {meta['min_time']} s")
    g = trim_count(meta["rounds"])
    print(f"trimmed mean over rounds ({g} lowest and {g} highest dropped) "
          "of each round's median call time")
    if half_ci:
        print("±: half the 95% confidence interval of the trimmed mean")
    edsl = meta.get("edsl")
    if edsl:
        print(f"edsl: {edsl['branch']}@{edsl['commit']}"
              f"{' (dirty)' if edsl['dirty'] else ''}, "
              f".so built {edsl['so_built']}"
              f"{' (OLDER THAN cpp/ HEAD)' if edsl['so_older_than_cpp'] else ''}")

    threads, backends, shapes = meta["threads"], meta["backends"], res.shapes
    if meta.get("peak"):
        print(f"peak ({meta['peak']['cpu']}, "
              f"{clock_label(meta['peak'])}): " + ", ".join(
            f"{t}T {peak_gflops(meta, t):.0f} GF" for t in threads))
    thread_cols = "".join(f"{f'{t}T':>10}" for t in threads)
    speedup = f"{f'{threads[-1]}T/{threads[0]}T':>9}" if len(threads) > 1 else ""

    # Wider GFLOPS cells when there is a spread to show.
    width = 16 if half_ci else 10
    gf_cols = "".join(f"{f'{t}T':>{width}}" for t in threads)

    def gf_cell(key: tuple) -> str:
        """Mean GFLOPS, with its CI over rounds when there is one."""
        if key not in gf:
            return f"{'-':>{width}}"
        if key not in half_ci:
            return f"{gf[key]:{width}.1f}"
        return f"{gf[key]:{width - 7}.1f} ±{half_ci[key]:5.1%}"

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
        print("\n=== dense / matmul time, paired per round "
              "(1.00 = same time as matmul) ===")
        print(f"{'shape':16}{'backend':9}{thread_cols}")
        for shape in shapes:
            for b in backends:
                cells = []
                for t in threads:
                    ratios = epilogue_ratios(res, shape, b, t)
                    cells.append(f"{ratio_spread(ratios).mean:10.2f}"
                                 if ratios else f"{'-':>10}")
                print(f"{'x'.join(map(str, shape)):16}{b:9}" + "".join(cells))


def main() -> int:
    """Run the grid or session and report it, or just report an existing
    file."""
    args = parse_args()
    if args.report:
        report(args.report)
        return 0
    if args.session:
        outs = run_session(args)
        print(f"\nresults: {outs[0].parent}", file=sys.stderr)
        for out in outs:
            print(f"\n##### {out.stem} #####")
            report(out)
        return 0
    out = run(args)
    print(f"\nresults: {out}", file=sys.stderr)
    report(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
