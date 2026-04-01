#!/bin/bash
# chain_generate.sh — Wait for kernel generation to finish, then:
#   1. Package kernels-qq50 tarball
#   2. Start iref-census-qq20 generation (parallelized)
#
# Usage:
#   nohup bash scripts/chain_generate.sh > logs/chain.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
PYTHON=".venv/bin/python"
LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=== chain_generate.sh started at $(date) ==="

# ── Step 0: Wait for kernel generation to finish ──────────────────
KERNEL_PID=$(ps aux | grep 'generate_data_packs.py.*kernels' | grep -v grep | awk '{print $2}' | head -1)

if [ -n "$KERNEL_PID" ]; then
    echo "[chain] Waiting for kernel generation (PID $KERNEL_PID) to finish..."
    while kill -0 "$KERNEL_PID" 2>/dev/null; do
        DONE=$(ls ~/Library/Caches/manifold-index/kernel_cache/kernel_*_qq50.pkl.gz 2>/dev/null | wc -l | tr -d ' ')
        echo "[chain]   $(date +%H:%M:%S) — $DONE/106 kernels saved, still running..."
        sleep 120
    done
    echo "[chain] Kernel generation finished at $(date)"
else
    echo "[chain] No kernel generation process found — skipping wait."
fi

FINAL_COUNT=$(ls ~/Library/Caches/manifold-index/kernel_cache/kernel_*_qq50.pkl.gz 2>/dev/null | wc -l | tr -d ' ')
echo "[chain] Final kernel count: $FINAL_COUNT"

# ── Step 1: Package kernels-qq50 ─────────────────────────────────
echo ""
echo "=== Step 1: Packaging kernels-qq50 ==="
$PYTHON scripts/generate_data_packs.py --pack kernels-qq50 --package-only 2>&1 | tee "$LOGDIR/kernels_package.log"

# ── Step 2: Generate iref-census-qq20 (parallelized) ─────────────
echo ""
echo "=== Step 2: Generating iref-census-qq20 ==="
$PYTHON scripts/generate_data_packs.py --pack iref-census-qq20 --resume 2>&1 | tee "$LOGDIR/iref_qq20.log"

echo ""
echo "=== chain_generate.sh completed at $(date) ==="
