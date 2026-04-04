#!/bin/zsh
# run.sh — One-click launcher for manifold index data generation.
#
# Usage:
#   ./run.sh              # run all assigned tasks for this machine
#   ./run.sh --dry-run    # preview tasks without computing
#   ./run.sh --no-push    # compute but don't push to GitHub
#   ./run.sh --task 0     # run only the first assigned task
#
# What it does:
#   1. Activates the correct Python (system python3 with snappy)
#   2. Sets MANIFOLD_INDEX_CACHE_DIR → v0.4/cache/  (tracked by git LFS)
#   3. Pulls latest from GitHub (gets other machine's finished kernels)
#   4. Runs this machine's assigned tasks from work_manifest.json
#   5. Commits & pushes completed cache files to GitHub

set -e
cd "$(dirname "$0")"   # always run from v0.4/

echo "============================================================"
echo "  Manifold Index — Data Generation"
echo "  Host    : $(hostname)"
echo "  Dir     : $(pwd)"
echo "  Python  : $(python3 --version 2>&1)"
echo "  Date    : $(date)"
echo "============================================================"

# Point all cache reads/writes to the in-repo cache/ directory
export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"

# Run the coordinator (pass through any flags: --dry-run, --no-push, --task N)
python3 scripts/run_assigned.py "$@"
