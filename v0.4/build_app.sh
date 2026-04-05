#!/usr/bin/env bash
# build_app.sh — Build Manifold Index v0.4 as a macOS .app
#
# Usage:
#   ./build_app.sh              # full build
#   ./build_app.sh --clean      # clean + rebuild
#
# Output:
#   dist/ManifoldIndex.app         — standalone application
#   dist/ManifoldIndex.zip         — distributable zip
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SPEC="ManifoldIndex.spec"
APP="dist/ManifoldIndex.app"
ZIP="dist/ManifoldIndex.zip"

# ── Clean ─────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "🧹  Cleaning previous build artifacts…"
    rm -rf build/ dist/ManifoldIndex.app dist/ManifoldIndex.zip
fi

# ── Activate venv ─────────────────────────────────────────────────
if [[ -d .venv ]]; then
    source .venv/bin/activate
fi

# ── Prefer venv pyinstaller to avoid picking up wrong Python ─────
if [[ -f .venv/bin/pyinstaller ]]; then
    PYINSTALLER=".venv/bin/pyinstaller"
elif command -v pyinstaller &>/dev/null; then
    PYINSTALLER="pyinstaller"
else
    echo "❌  PyInstaller not found. Install with: pip install pyinstaller"
    exit 1
fi

if [[ ! -f "$SPEC" ]]; then
    echo "❌  Spec file not found: $SPEC"
    exit 1
fi

# ── Build ─────────────────────────────────────────────────────────
echo "🔨  Building ManifoldIndex.app v0.4.0 …"
echo "    Spec: $SPEC"
echo "    Entry: launcher.py → manifold_index.app.main()"
echo ""

"$PYINSTALLER" "$SPEC" --noconfirm 2>&1 | tail -30

echo ""

# ── Verify ────────────────────────────────────────────────────────
if [[ ! -d "$APP" ]]; then
    echo "❌  Build failed — $APP not found."
    exit 1
fi

SIZE=$(du -sh "$APP" | cut -f1)
BUNDLE_VER=$(defaults read "$SCRIPT_DIR/$APP/Contents/Info.plist" CFBundleShortVersionString 2>/dev/null || echo "?")
echo "✅  Built: $APP  ($SIZE, bundle version $BUNDLE_VER)"

# ── Ad-hoc code sign ──────────────────────────────────────────────
echo "🔏  Ad-hoc signing …"
codesign --force --deep --sign - "$APP" 2>/dev/null && echo "✅  Signed" || echo "⚠️  Sign skipped"

# ── Create distributable zip ──────────────────────────────────────
echo "📦  Creating $ZIP …"
rm -f "$ZIP"
ditto -c -k --sequesterRsrc --keepParent "$APP" "$ZIP"
ZIP_SIZE=$(ls -lh "$ZIP" | awk '{print $5}')
echo "✅  Zip: $ZIP  ($ZIP_SIZE)"

echo ""
echo "To test locally:"
echo "    open $APP"
echo ""
echo "To upload to GitHub:"
echo "    gh release create v0.4.0 dist/ManifoldIndex.zip --title 'v0.4.0' --notes 'v0.4.0 release'"
echo "  or upload to existing release:"
echo "    gh release upload v0.4.0 dist/ManifoldIndex.zip --clobber"
