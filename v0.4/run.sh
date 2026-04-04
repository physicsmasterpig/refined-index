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
#   1. Auto-detects the best available Python (3.10+ with snappy/numpy)
#   2. Sets MANIFOLD_INDEX_CACHE_DIR → v0.4/cache/  (tracked by git LFS)
#   3. Pulls latest from GitHub (gets other machine's finished kernels)
#   4. Runs this machine's assigned tasks from work_manifest.json
#   5. Commits & pushes completed cache files to GitHub

set -e
cd "$(dirname "$0")"   # always run from v0.4/

# ── Auto-detect a working Python (3.10+, has numpy) ──────────────────────────
PYTHON=""
for candidate in ./.venv/bin/python python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
    # Check version >= 3.10 and numpy importable
    if "$candidate" -c "import sys,numpy; assert sys.version_info>=(3,10)" 2>/dev/null; then
      PYTHON="$candidate"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  echo "ERROR: No suitable Python found (need 3.10+ with numpy installed)."
  echo ""
  echo "Fix with:"
  echo "  brew install python@3.12"
  echo "  python3.12 -m venv .venv"
  echo "  .venv/bin/python -m pip install -e ."
  exit 1
fi

echo "============================================================"
echo "  Manifold Index — Data Generation"
echo "  Host    : $(hostname)"
echo "  Dir     : $(pwd)"
echo "  Python  : $($PYTHON --version 2>&1)  ($PYTHON)"
echo "  Date    : $(date)"
echo "============================================================"

# Point all cache reads/writes to the in-repo cache/ directory
export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"

# Run the coordinator (pass through any flags: --dry-run, --no-push, --task N)
"$PYTHON" scripts/run_assigned.py "$@"
