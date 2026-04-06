#!/bin/zsh
# bin/iref_cache_build.sh — Pre-compute and cache I^ref(m,e) for census manifolds.
#
# ── Machine profiles (auto-detected) ────────────────────────────────────────
#   M4 Max  36 GB  →  workers=10  mem-per-worker=3.5 GB
#   M1 Max  64 GB  →  workers=8   mem-per-worker=6.0 GB
#
# ── ETA reference table ─────────────────────────────────────────────────────
#   Census          Count   Tet   Both Macs   M4 alone   M1 alone
#   m-series        410     2–5   ~37 min      ~60 min    ~80 min   (almost done)
#   s-series        962     6     ~3–4 h       ~5–6 h     ~8–10 h
#   v-series        3552    7     ~20–30 h     ~35–50 h   days
#   t-series        12846   8     days         days       —
#   knot file       varies  —     depends on manifolds
#
# ── Census flags ────────────────────────────────────────────────────────────
#   (default)       m003-m412
#   --s-series      s000-s961   (6-tet, adds ~3-4 h both Macs)
#   --v-series      v0000-v3551 (7-tet, adds ~20-30 h — plan carefully)
#   --census RANGE  any explicit range, e.g. s000-s200,m390-m412
#   --manifolds ... individual names: m003 4_1 5_1^2 L5a1
#   --knots FILE    text file, one manifold per line (# comments ok)
#
# ── Common usage ────────────────────────────────────────────────────────────
#   ./bin/iref_cache_build.sh                         # m-series, resume
#   ./bin/iref_cache_build.sh --s-series              # s-series, resume
#   ./bin/iref_cache_build.sh --s-series --fresh      # s-series, full recompute
#   ./bin/iref_cache_build.sh --census s000-s480      # M4 half of s-series
#   ./bin/iref_cache_build.sh --census s481-s961      # M1 half of s-series
#   ./bin/iref_cache_build.sh --knots knots.txt       # arbitrary manifold list
#   ./bin/iref_cache_build.sh --manifolds 4_1 5_1 5_2 # individual knots
#   ./bin/iref_cache_build.sh --fresh                 # force full recompute
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
  echo "ERROR: No suitable Python found (need 3.10+ with numpy)."
  exit 1
fi

# ── Machine profile ───────────────────────────────────────────────────────────
RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo 8589934592)
RAM_GB=$(( RAM_BYTES / 1073741824 ))
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "unknown")
PERF_CORES=$(sysctl -n hw.perflevel0.logicalcpu 2>/dev/null || \
             sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

if [[ "$CHIP" == *"M4 Max"* ]] || [[ $RAM_GB -ge 32 && $RAM_GB -lt 50 ]]; then
  PROFILE="M4 Max  ${RAM_GB} GB  (${PERF_CORES}P cores)"
  DEFAULT_WORKERS=10
  DEFAULT_MEM=3.5
elif [[ "$CHIP" == *"M1 Max"* ]] || [[ $RAM_GB -ge 50 ]]; then
  PROFILE="M1 Max  ${RAM_GB} GB  (${PERF_CORES}P cores)"
  DEFAULT_WORKERS=8
  DEFAULT_MEM=6.0
else
  PROFILE="generic  ${RAM_GB} GB  (${PERF_CORES} cores)"
  DEFAULT_WORKERS=$(( PERF_CORES > 2 ? PERF_CORES / 2 : 2 ))
  DEFAULT_MEM=3.0
fi

# ── Parse flags ───────────────────────────────────────────────────────────────
FRESH=0
S_SERIES=0
V_SERIES=0
PASSTHROUGH=()
KNOTS_FILE=""
CENSUS_OVERRIDE=""

i=1
while [[ $i -le $# ]]; do
  arg="${@[$i]}"
  case "$arg" in
    --fresh)      FRESH=1 ;;
    --s-series)   S_SERIES=1 ;;
    --v-series)   V_SERIES=1 ;;
    --knots)
      i=$(( i+1 )); KNOTS_FILE="${@[$i]}"
      PASSTHROUGH+=(--knot-file "$KNOTS_FILE") ;;
    --census)
      i=$(( i+1 )); CENSUS_OVERRIDE="${@[$i]}"
      PASSTHROUGH+=(--census "$CENSUS_OVERRIDE") ;;
    *)
      PASSTHROUGH+=("$arg") ;;
  esac
  i=$(( i+1 ))
done

# ── Expand census shorthand flags ─────────────────────────────────────────────
if [[ $S_SERIES -eq 1 ]]; then
  PASSTHROUGH+=(--census s000-s961)
fi
if [[ $V_SERIES -eq 1 ]]; then
  PASSTHROUGH+=(--census v0000-v3551)
fi

# ── Build default argument list ───────────────────────────────────────────────
DEFAULTS=(
  --workers        $DEFAULT_WORKERS
  --mem-per-worker $DEFAULT_MEM
  --qq             20
)
[[ $FRESH -eq 0 ]] && DEFAULTS+=(--skip-existing)

# ── Header ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Manifold Index — I^ref Cache Builder"
echo "  Host    : $(hostname)"
echo "  Profile : $PROFILE"
echo "  Dir     : $(pwd)"
echo "  Python  : $($PYTHON --version 2>&1)  ($PYTHON)"
echo "  Date    : $(date)"
if [[ $S_SERIES -eq 1 ]]; then
  echo "  Census  : s-series (s000-s961, 962 manifolds, 6-tet)"
  echo "  ETA     : ~3-4 h both Macs  /  ~5-6 h M4 alone"
elif [[ $V_SERIES -eq 1 ]]; then
  echo "  Census  : v-series (v0000-v3551, 3552 manifolds, 7-tet)"
  echo "  ETA     : ~20-30 h both Macs — plan accordingly"
elif [[ -n "$KNOTS_FILE" ]]; then
  N_KNOTS=$(grep -v '^\s*#' "$KNOTS_FILE" 2>/dev/null | grep -c '\S' || echo '?')
  echo "  Census  : knot file ($KNOTS_FILE, $N_KNOTS entries)"
else
  echo "  Census  : m-series (default: m003-m412)"
fi
[[ $FRESH -eq 1 ]] && echo "  Mode    : FRESH (full recompute)" \
                    || echo "  Mode    : resume (--skip-existing)"
echo "============================================================"

export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"
exec "$PYTHON" scripts/build_iref_cache.py "${DEFAULTS[@]}" "${PASSTHROUGH[@]}"
