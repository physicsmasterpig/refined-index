#!/usr/bin/env bash
# build_app.sh — Build Manifold Index as a macOS .app bundle
#
# Usage:
#   ./build_app.sh          # full build
#   ./build_app.sh --clean  # clean + rebuild
#
# Output:
#   dist/ManifoldIndex.app   — the standalone application
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Clean previous build artifacts ────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "🧹  Cleaning previous build…"
    rm -rf build/ dist/
fi

# ── Activate venv if present ──────────────────────────────────────
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# ── Check PyInstaller ─────────────────────────────────────────────
if ! command -v pyinstaller &>/dev/null; then
    echo "❌  PyInstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

# ── Build ─────────────────────────────────────────────────────────
echo "🔨  Building ManifoldIndex.app …"
echo "    (this may take a few minutes)"
echo ""

pyinstaller manifold_index.spec --noconfirm 2>&1 | tail -20

echo ""

# ── Verify ────────────────────────────────────────────────────────
APP="dist/ManifoldIndex.app"
if [[ -d "$APP" ]]; then
    SIZE=$(du -sh "$APP" | cut -f1)
    echo "✅  Built successfully: $APP  ($SIZE)"
    echo ""
    echo "To run:"
    echo "    open dist/ManifoldIndex.app"
    echo ""
    echo "To create a DMG for distribution:"
    echo "    hdiutil create -volname ManifoldIndex \\"
    echo "        -srcfolder dist/ManifoldIndex.app \\"
    echo "        -ov -format UDZO \\"
    echo "        dist/ManifoldIndex.dmg"
else
    echo "❌  Build failed — dist/ManifoldIndex.app not found."
    echo "    Check the output above for errors."
    exit 1
fi
