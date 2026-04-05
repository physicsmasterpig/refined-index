#!/bin/zsh
# bin/kernel_build_start.sh — Start (or restart) the kernel pre-computation run.
#
# Pulls the latest cached kernels from GitHub, runs this machine's assigned
# tasks from work_manifest.json, then pushes the results back.
#
# Usage:
#   ./bin/kernel_build_start.sh                # run all assigned tasks
#   ./bin/kernel_build_start.sh --dry-run      # preview without computing
#   ./bin/kernel_build_start.sh --no-push      # compute but skip git push
#   ./bin/kernel_build_start.sh --task 0       # run only the first task
#
# Companion scripts:
#   bin/kernel_build_pause.sh   — signal a clean stop after current slope
#   bin/kernel_build_resume.sh  — remove pause signal and restart

set -e
cd "$(dirname "$0")/.."   # always run from v0.4/

# ── Auto-detect a working Python (3.10+, has numpy) ──────────────────────────
PYTHON=""
for candidate in ./.venv/bin/python python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" &>/dev/null; then
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
echo "  Manifold Index — Kernel Pre-computation"
echo "  Host    : $(hostname)"
echo "  Dir     : $(pwd)"
echo "  Python  : $($PYTHON --version 2>&1)  ($PYTHON)"
echo "  Date    : $(date)"
echo "============================================================"

# Point all cache reads/writes to the in-repo cache/ directory
export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"

# Run the coordinator (pass through any flags: --dry-run, --no-push, --task N)
"$PYTHON" scripts/run_assigned.py "$@"
