#!/bin/zsh
# bin/iref_cache_build.sh — Pre-compute and cache I^ref(m,e) for census manifolds.
#
# Run AFTER kernel_build_start.sh has finished.  Populates the on-disk I^ref
# cache so Dehn-filling calculations in the GUI become near-instant lookups.
#
# No separate pause/resume scripts needed here: each manifold is saved
# atomically (a few seconds each).  Just Ctrl+C to stop at any time, then
# re-run with --skip-existing to continue from where you left off.
#
# Usage:
#   ./bin/iref_cache_build.sh                         # all m003–m412, qq=25
#   ./bin/iref_cache_build.sh --qq 20                 # different q-order
#   ./bin/iref_cache_build.sh --census m003-m050      # smaller range
#   ./bin/iref_cache_build.sh --manifolds m003 m004   # specific manifolds
#   ./bin/iref_cache_build.sh --skip-existing         # skip already-cached
#   ./bin/iref_cache_build.sh --dry-run               # preview only

set -e
cd "$(dirname "$0")/.."   # always run from v0.4/

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
  exit 1
fi

echo "============================================================"
echo "  Manifold Index — I^ref Cache Builder"
echo "  Host    : $(hostname)"
echo "  Dir     : $(pwd)"
echo "  Python  : $($PYTHON --version 2>&1)  ($PYTHON)"
echo "  Date    : $(date)"
echo "============================================================"

export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"
"$PYTHON" scripts/build_iref_cache.py "$@"
