#!/usr/bin/env bash
# Create or update the conda environments the benchmark workers run in.
#
# Usage:
#   benchmarks/setup_envs.sh              # every envs/*.yml
#   benchmarks/setup_envs.sh np-openblas  # just the named ones
#
# Safe to rerun: an existing env is updated with --prune, so it matches its
# yml exactly. Set CONDA to override the conda binary.

set -euo pipefail

CONDA="${CONDA:-$HOME/anaconda3/condabin/conda}"
ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/envs" && pwd)"

if [[ ! -x "$CONDA" ]]; then
    echo "conda not found at $CONDA (set CONDA=...)" >&2
    exit 1
fi

if [[ $# -gt 0 ]]; then
    names=("$@")
else
    names=()
    for yml in "$ENV_DIR"/*.yml; do
        names+=("$(basename "$yml" .yml)")
    done
fi

existing="$("$CONDA" env list --json)"

for name in "${names[@]}"; do
    yml="$ENV_DIR/$name.yml"
    if [[ ! -f "$yml" ]]; then
        echo "no such env file: $yml" >&2
        exit 1
    fi
    if grep -q "/envs/$name\"" <<<"$existing"; then
        echo "==> updating $name"
        "$CONDA" env update -n "$name" -f "$yml" --prune
    else
        echo "==> creating $name"
        "$CONDA" env create -f "$yml"
    fi
done
