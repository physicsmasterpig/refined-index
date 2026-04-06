#!/bin/zsh
# bin/iref_cache_build.sh — Pre-compute and cache I^ref(m,e) for census manifolds.
#
# Machine profiles (auto-detected from RAM + chip):
#   M4 Max  36 GB  →  workers=10  mem-per-worker=3.5 GB
#   M1 Max  64 GB  →  workers=8   mem-per-worker=6.0 GB
#   other          →  workers=cpu_count//2   mem-per-worker=3.0 GB
#
# Defaults applied automatically; override any flag on the command line:
#   ./bin/iref_cache_build.sh                         # all m003–m412, resume
#   ./bin/iref_cache_build.sh --fresh                 # force full recompute
#   ./bin/iref_cache_build.sh --workers 6             # override worker count
#   ./bin/iref_cache_build.sh --qq 25                 # higher q-order
#   ./bin/iref_cache_build.sh --census m003-m050      # smaller range
#   ./bin/iref_cache_build.sh --manifolds m003 m004   # specific manifolds
#   ./bin/iref_cache_build.sh --dry-run               # preview only

set -e
cd "$(dirname "$0")/.."   # always run from v0.4/

# ── Python discovery ─────────────────────────────────────────────────────────
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

# ── Machine profile detection ─────────────────────────────────────────────────
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 8589934592)
RAM_GB=$(( RAM_BYTES / 1073741824 ))
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || \
             sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

if [[ "$CHIP" == *"M4 Max"* ]] || [[ $RAM_GB -ge 32 && $RAM_GB -lt 50 ]]; then
  PROFILE="M4 Max  ${RAM_GB} GB  (${PERF_CORES}P cores)"
  DEFAULT_WORKERS=10         # all 10 performance cores
  DEFAULT_MEM=3.5            # 10 × 3.5 GB = 35 GB ceiling
elif [[ "$CHIP" == *"M1 Max"* ]] || [[ $RAM_GB -ge 50 ]]; then
  PROFILE="M1 Max  ${RAM_GB} GB  (${PERF_CORES}P cores)"
  DEFAULT_WORKERS=8          # all 8 performance cores
  DEFAULT_MEM=6.0            # 8 × 6 GB = 48 GB ceiling
else
  PROFILE="generic  ${RAM_GB} GB  (${PERF_CORES} cores)"
  DEFAULT_WORKERS=$(( PERF_CORES > 2 ? PERF_CORES / 2 : 2 ))
  DEFAULT_MEM=3.0
fi

# ── Intercept --fresh (suppress default --skip-existing) ─────────────────────
FRESH=0
PASSTHROUGH=()
for arg in "$@"; do
  if [[ "$arg" == "--fresh" ]]; then
    FRESH=1
  else
    PASSTHROUGH+=("$arg")
  fi
done

# ── Build default argument list ───────────────────────────────────────────────
# Defaults come BEFORE user flags — argparse uses the LAST occurrence, so
# any flag the user passes on the command line wins automatically.
DEFAULTS=(
  --workers        $DEFAULT_WORKERS
  --mem-per-worker $DEFAULT_MEM
  --qq             20
)
[[ $FRESH -eq 0 ]] && DEFAULTS+=(--skip-existing)

# ── Header ───────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Manifold Index — I^ref Cache Builder"
echo "  Host    : $(hostname)"
echo "  Profile : $PROFILE"
echo "  Dir     : $(pwd)"
echo "  Python  : $($PYTHON --version 2>&1)  ($PYTHON)"
echo "  Date    : $(date)"
[[ $FRESH -eq 1 ]] && echo "  Mode    : FRESH (full recompute)" \
                    || echo "  Mode    : resume (--skip-existing)"
echo "============================================================"

export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"
exec "$PYTHON" scripts/build_iref_cache.py "${DEFAULTS[@]}" "${PASSTHROUGH[@]}"
