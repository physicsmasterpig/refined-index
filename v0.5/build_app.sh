#!/usr/bin/env bash
# build_app.sh — Build Refined Index Calculator v0.5 as a macOS .app
#
# Usage:
#   ./build_app.sh              # full build
#   ./build_app.sh --clean      # clean previous artifacts, then rebuild
#
# Output:
#   dist/ManifoldIndex.app         — standalone application bundle
#   dist/ManifoldIndex.zip         — distributable zip
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SPEC="ManifoldIndex.spec"
APP="dist/ManifoldIndex.app"
ZIP="dist/ManifoldIndex.zip"
APP_VERSION="0.5.12"

# ── Clean ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "🧹  Cleaning previous build artifacts…"
    rm -rf build/ dist/ManifoldIndex.app dist/ManifoldIndex.zip
fi

# ── Prefer project venv (one level up from v0.5/) ─────────────────
VENV_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)/.venv"
if [[ -d "$VENV_ROOT" ]]; then
    source "$VENV_ROOT/bin/activate"
elif [[ -d "$SCRIPT_DIR/.venv" ]]; then
    source "$SCRIPT_DIR/.venv/bin/activate"
fi

# ── Locate PyInstaller ────────────────────────────────────────────
if [[ -f "$VENV_ROOT/bin/pyinstaller" ]]; then
    PYINSTALLER="$VENV_ROOT/bin/pyinstaller"
elif [[ -f "$SCRIPT_DIR/.venv/bin/pyinstaller" ]]; then
    PYINSTALLER="$SCRIPT_DIR/.venv/bin/pyinstaller"
elif command -v pyinstaller &>/dev/null; then
    PYINSTALLER="pyinstaller"
else
    echo "❌  PyInstaller not found.  Install with: pip install pyinstaller"
    exit 1
fi

if [[ ! -f "$SPEC" ]]; then
    echo "❌  Spec file not found: $SPEC"
    exit 1
fi

# ── Preflight check — confirm entry point importable ──────────────
echo "🔍  Checking entry point…"
PYTHON_BIN="$(dirname "$PYINSTALLER")/python"
"$PYTHON_BIN" -c "from manifold_index.app import launch_gui; print('  ✓ launch_gui importable')"

# ── Build ─────────────────────────────────────────────────────────
echo ""
echo "🔨  Building Refined Index Calculator v${APP_VERSION}…"
echo "    Spec:  $SPEC"
echo "    Entry: launcher.py → manifold_index.app:launch_gui"
echo ""

"$PYINSTALLER" "$SPEC" --noconfirm 2>&1 | tail -40

echo ""

# ── Verify bundle was created ─────────────────────────────────────
if [[ ! -d "$APP" ]]; then
    echo "❌  Build failed — $APP not found."
    exit 1
fi

SIZE=$(du -sh "$APP" | cut -f1)
BUNDLE_VER=$(defaults read "$SCRIPT_DIR/$APP/Contents/Info.plist" \
             CFBundleShortVersionString 2>/dev/null || echo "?")
echo "✅  Built: $APP  ($SIZE, bundle version $BUNDLE_VER)"

# ── Ad-hoc code sign (local use only; re-sign with Developer ID for distribution) ──
echo "🔏  Ad-hoc signing…"
codesign --force --deep --sign - "$APP" 2>/dev/null \
    && echo "✅  Signed (ad-hoc)" \
    || echo "⚠️   Sign step skipped (codesign not available)"

# ── Create distributable zip ──────────────────────────────────────
echo "📦  Creating $ZIP…"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
ZIP_SIZE=$(ls -lh "$ZIP" | awk '{print $5}')
echo "✅  Zip: $ZIP  ($ZIP_SIZE)"

echo ""
echo "To test locally:"
echo "    open $APP"
echo ""
echo "To notarise for distribution, run:"
echo "    xcrun notarytool submit $ZIP --keychain-profile <PROFILE> --wait"
