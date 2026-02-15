#!/usr/bin/env python3
"""Migration script: swap Array[shape, dtype] -> Array[dtype, shape] and Tensor[shape, dtype] -> Tensor[dtype, shape].

Handles patterns like:
  Array[10, i32]       -> Array[i32, 10]
  Array[2, 3, f32]     -> Array[f32, 2, 3]
  Array[2, 3, 4, i32]  -> Array[i32, 2, 3, 4]
  Tensor[4, f32]       -> Tensor[f32, 4]
  Tensor[2, 3, i32]    -> Tensor[i32, 2, 3]

Also handles ArrayType(shape, dtype) -> ArrayType(dtype, shape) and
TensorType(shape, dtype) -> TensorType(dtype, shape) constructor calls.

Run with --dry-run to preview changes without writing.
"""

import re
import sys
from pathlib import Path

# Known scalar type names used in subscript syntax
DTYPE_NAMES = {'i32', 'f32', 'i1', 'f64', 'i64'}

# Match Array[...] or Tensor[...] subscript expressions
# This regex captures the full bracket content
SUBSCRIPT_RE = re.compile(
    r'\b(Array|Tensor)\[([^\[\]]+)\]'
)

def swap_subscript_params(match: re.Match) -> str:
    """Swap last param (dtype) to first position."""
    name = match.group(1)
    inner = match.group(2)
    params = [p.strip() for p in inner.split(',')]

    if len(params) < 2:
        return match.group(0)  # Leave as-is

    # Check if last param is a dtype name
    last = params[-1]
    if last not in DTYPE_NAMES:
        return match.group(0)  # Not a pattern we recognize

    # Move dtype from last to first
    new_params = [last] + params[:-1]
    return f"{name}[{', '.join(new_params)}]"


def migrate_file(filepath: Path, dry_run: bool = False) -> bool:
    """Migrate a single file. Returns True if changes were made."""
    content = filepath.read_text()
    new_content = SUBSCRIPT_RE.sub(swap_subscript_params, content)

    if new_content == content:
        return False

    if dry_run:
        print(f"  WOULD modify: {filepath}")
        # Show diffs
        for i, (old_line, new_line) in enumerate(
            zip(content.splitlines(), new_content.splitlines()), 1
        ):
            if old_line != new_line:
                print(f"    L{i}: {old_line.strip()}")
                print(f"      -> {new_line.strip()}")
    else:
        filepath.write_text(new_content)
        print(f"  Modified: {filepath}")

    return True


def main():
    dry_run = '--dry-run' in sys.argv
    root = Path(__file__).parent

    # Files to migrate (skip types.py - handled manually, skip this script)
    py_files = sorted(root.rglob('*.py'))
    skip = {
        root / 'migrate_type_order.py',
        root / 'mlir_edsl' / 'types.py',
    }

    changed = 0
    for f in py_files:
        if f in skip:
            continue
        if 'build' in f.parts or '__pycache__' in f.parts or '.eggs' in f.parts:
            continue
        if migrate_file(f, dry_run):
            changed += 1

    mode = "Would modify" if dry_run else "Modified"
    print(f"\n{mode} {changed} file(s).")
    if dry_run:
        print("Run without --dry-run to apply changes.")


if __name__ == '__main__':
    main()
