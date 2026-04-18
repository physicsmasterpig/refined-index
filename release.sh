#!/usr/bin/env bash
# release.sh — Build macOS + Windows and publish a GitHub Release.
#
# Runs on Mac. Builds the macOS app locally, then pushes a version tag to
# trigger GitHub Actions for the Windows exe. Both artifacts are uploaded
# to the same GitHub Release, and docs/index.html is updated automatically.
#
# Usage:
#   ./release.sh v1.0.0            # full release
#   ./release.sh v1.0.0 --dry-run  # preview all changes, build nothing
#   ./release.sh v1.0.0 --skip-mac # skip local macOS build (Windows only)
#
# Requirements:
#   - macOS (for the local .app build)
#   - PyInstaller in your venv or PATH
#   - gh CLI authenticated:  brew install gh && gh auth login

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'; BLD='\033[1m'; RST='\033[0m'
info()  { echo -e "  $*"; }
ok()    { echo -e "  ${GRN}✓${RST}  $*"; }
warn()  { echo -e "  ${YEL}⚠${RST}   $*"; }
die()   { echo -e "  ${RED}✗${RST}  $*" >&2; exit 1; }
header(){ echo -e "\n${BLD}$*${RST}"; }

# ── Argument parsing ──────────────────────────────────────────────────────────
VERSION_ARG="${1:-}"
DRY_RUN=false
SKIP_MAC=false

for arg in "${@:2}"; do
  case "$arg" in
    --dry-run)   DRY_RUN=true ;;
    --skip-mac)  SKIP_MAC=true ;;
    *) die "Unknown option: $arg" ;;
  esac
done

# ── Version validation ────────────────────────────────────────────────────────
if [[ -z "$VERSION_ARG" ]]; then
  echo "Usage: ./release.sh v<MAJOR>.<MINOR>.<PATCH> [--dry-run] [--skip-mac]"
  echo "  e.g.  ./release.sh v1.0.0"
  exit 1
fi

VERSION_NUM="${VERSION_ARG#v}"    # 0.5.4
VERSION_TAG="v${VERSION_NUM}"    # v1.0.0

if ! [[ "$VERSION_NUM" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  die "Bad version format — use vMAJOR.MINOR.PATCH (e.g. v1.0.0)"
fi

# ── Detect current version ────────────────────────────────────────────────────
CURRENT_VERSION=$(grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

header "━━  Refined Index Calculator — Release  ━━"
info "Current : v${CURRENT_VERSION}"
info "New     : ${VERSION_TAG}"
$DRY_RUN  && info "Mode    : ${YEL}DRY-RUN${RST} (no changes written)"
$SKIP_MAC && info "macOS   : ${YEL}skipped${RST}"
echo ""

# ── Pre-flight: git ───────────────────────────────────────────────────────────
header "[ 1/6 ]  Pre-flight"

BRANCH=$(git rev-parse --abbrev-ref HEAD)
ok "Branch: $BRANCH"

# Block on uncommitted changes to *non-version* tracked files only.
# It's fine if version files are already bumped — the commit step handles that.
DIRTY=$(git status --porcelain | grep -v '^ ' \
  | grep -vE '(pyproject\.toml|ManifoldIndex.*\.spec|build_app\.(sh|bat)|docs/index\.html)' \
  | grep -v '^??' || true)
if [[ -n "$DIRTY" ]]; then
  warn "Uncommitted staged changes outside version files:"
  echo "$DIRTY"
  $DRY_RUN || die "Commit or stash these before releasing."
fi

if git rev-parse "$VERSION_TAG" >/dev/null 2>&1; then
  die "Tag '$VERSION_TAG' already exists. Delete it first or choose a new version."
fi

command -v gh >/dev/null 2>&1 || die "gh CLI not found. Install: brew install gh && gh auth login"

# ── Pre-flight: PyInstaller ───────────────────────────────────────────────────
if ! $SKIP_MAC; then
  VENV_ROOT="$SCRIPT_DIR/.venv"
  if [[ -d "$VENV_ROOT/bin" ]]; then
    PYINSTALLER="$VENV_ROOT/bin/pyinstaller"
    PYTHON_BIN="$VENV_ROOT/bin/python"
  elif [[ -d "$SCRIPT_DIR/.venv/bin" ]]; then
    PYINSTALLER="$SCRIPT_DIR/.venv/bin/pyinstaller"
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
  elif command -v pyinstaller >/dev/null 2>&1; then
    PYINSTALLER="pyinstaller"
    PYTHON_BIN="python"
  else
    die "PyInstaller not found. Install: pip install pyinstaller  (or use --skip-mac)"
  fi
  ok "PyInstaller: $PYINSTALLER"

  "$PYTHON_BIN" -c "
import sys; sys.path.insert(0, 'src')
from manifold_index.app import launch_gui
" 2>/dev/null && ok "Entry point importable" \
              || die "launch_gui not importable — check your venv"
fi

# ── sed helper ────────────────────────────────────────────────────────────────
do_sed() {
  local file="$1" pattern="$2" replacement="$3"
  if $DRY_RUN; then
    info "  [dry] $file  ::  s|$pattern|$replacement|"
  else
    sed -i '' "s|$pattern|$replacement|g" "$file"
  fi
}

# ── Version bump ──────────────────────────────────────────────────────────────
header "[ 2/6 ]  Version bump  (${CURRENT_VERSION} → ${VERSION_NUM})"

do_sed "pyproject.toml"        "version = \"${CURRENT_VERSION}\""   "version = \"${VERSION_NUM}\""
do_sed "ManifoldIndex.spec"    "APP_VERSION = \".*\""               "APP_VERSION = \"${VERSION_NUM}\""
do_sed "ManifoldIndex_win.spec" "APP_VERSION = \".*\""              "APP_VERSION = \"${VERSION_NUM}\""
do_sed "build_app.sh"          "APP_VERSION=\".*\""                 "APP_VERSION=\"${VERSION_NUM}\""
do_sed "build_app.bat"         "set APP_VERSION=.*"                 "set APP_VERSION=${VERSION_NUM}"
do_sed "src/manifold_index/__init__.py" \
  "__version__ = \"${CURRENT_VERSION}\"" \
  "__version__ = \"${VERSION_NUM}\""

# docs/index.html — URLs and version text
do_sed "docs/index.html" \
  "releases/download/v${CURRENT_VERSION}/ManifoldIndex\.zip" \
  "releases/download/${VERSION_TAG}/ManifoldIndex.zip"
do_sed "docs/index.html" \
  "releases/download/v${CURRENT_VERSION}/ManifoldIndex\.exe" \
  "releases/download/${VERSION_TAG}/ManifoldIndex.exe"
do_sed "docs/index.html" \
  "v${CURRENT_VERSION}" \
  "v${VERSION_NUM}"

ok "pyproject.toml, both specs, build scripts, docs/index.html"

# ── Verify docs/index.html actually points at the new tag ────────────────────
# The sed above substitutes CURRENT→NEW; if docs drifted (a prior release
# skipped this step) the patterns silently no-op.  Catch that here.
if ! $DRY_RUN; then
  for asset in ManifoldIndex.zip ManifoldIndex.exe; do
    expected="releases/download/${VERSION_TAG}/${asset}"
    if ! grep -q "$expected" docs/index.html; then
      die "docs/index.html does not reference ${expected} after version bump.
    Likely cause: docs still point at an older version than pyproject.toml.
    Fix docs/index.html manually so its download links match v${CURRENT_VERSION}
    before re-running, or edit this release to the new tag directly."
    fi
  done
  ok "docs/index.html references v${VERSION_NUM} download URLs"
fi

# ── Dry-run exit ──────────────────────────────────────────────────────────────
if $DRY_RUN; then
  echo ""
  echo "━━  DRY-RUN complete — nothing written  ━━"
  echo "    Run without --dry-run to execute."
  exit 0
fi

# ── macOS build ───────────────────────────────────────────────────────────────
MAC_ZIP=""
if ! $SKIP_MAC; then
  header "[ 3/6 ]  macOS build"

  info "Running PyInstaller..."
  "$PYINSTALLER" ManifoldIndex.spec --noconfirm --clean 2>&1 | tail -5

  [[ -d "dist/ManifoldIndex.app" ]] || die "Build failed — dist/ManifoldIndex.app not found"
  ok "dist/ManifoldIndex.app  ($(du -sh dist/ManifoldIndex.app | cut -f1))"

  info "Ad-hoc signing..."
  codesign --force --deep --sign - dist/ManifoldIndex.app 2>/dev/null \
    && ok "Signed (ad-hoc)" || warn "codesign skipped"

  info "Creating zip..."
  rm -f dist/ManifoldIndex.zip
  ditto -c -k --sequesterRsrc --keepParent dist/ManifoldIndex.app dist/ManifoldIndex.zip
  ok "dist/ManifoldIndex.zip  ($(du -sh dist/ManifoldIndex.zip | cut -f1))"

  MAC_ZIP="$SCRIPT_DIR/dist/ManifoldIndex.zip"
else
  header "[ 3/6 ]  macOS build — skipped"
fi

# ── Commit + push branch ──────────────────────────────────────────────────────
header "[ 4/6 ]  Commit & push branch"

git add \
  pyproject.toml \
  ManifoldIndex.spec \
  ManifoldIndex_win.spec \
  build_app.sh \
  build_app.bat \
  src/manifold_index/__init__.py \
  docs/index.html

if git diff --cached --quiet; then
  ok "Nothing to commit (version files already at ${VERSION_NUM})"
else
  git commit -m "chore: bump version to ${VERSION_NUM}"
  ok "Committed version bump"
fi

git push origin "$BRANCH"
ok "Pushed branch '$BRANCH'"

# ── Create GitHub Release + upload macOS zip ──────────────────────────────────
header "[ 5/6 ]  GitHub Release"

REPO_SLUG=$(git remote get-url origin \
  | sed 's|git@github\.com:||; s|https://github\.com/||; s|\.git$||')

gh release create "$VERSION_TAG" \
  --repo "$REPO_SLUG" \
  --title "${VERSION_TAG}" \
  --notes "## ${VERSION_TAG}

### macOS
1. Download \`ManifoldIndex.zip\`
2. Extract and move \`ManifoldIndex.app\` to Applications
3. On first launch: \`xattr -cr ~/Downloads/ManifoldIndex.app\`

### Windows
1. Download \`ManifoldIndex.exe\`
2. Run it — no installation needed
3. If Windows Defender warns, click **More info → Run anyway**"

ok "Release created: https://github.com/${REPO_SLUG}/releases/tag/${VERSION_TAG}"

if [[ -n "$MAC_ZIP" ]]; then
  info "Uploading macOS zip..."
  gh release upload "$VERSION_TAG" "$MAC_ZIP" --repo "$REPO_SLUG"
  ok "ManifoldIndex.zip uploaded"
fi

# ── Push tag → trigger Windows CI ─────────────────────────────────────────────
header "[ 6/6 ]  Windows build (GitHub Actions)"

git tag -f "$VERSION_TAG"
git push origin "$VERSION_TAG" --force
ok "Tag '${VERSION_TAG}' pushed — Windows CI triggered"

info "Waiting for build to finish (Ctrl+C to detach)..."
echo ""

# Give Actions a moment to register the run
for i in 1 2 3 4 5; do
  RUN_ID=$(gh run list --repo "$REPO_SLUG" \
    --workflow build-windows.yml --limit 1 \
    --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
  [[ -n "$RUN_ID" && "$RUN_ID" != "null" ]] && break
  sleep 3
done

if [[ -n "${RUN_ID:-}" && "$RUN_ID" != "null" ]]; then
  gh run watch "$RUN_ID" --repo "$REPO_SLUG" || true
  echo ""
  STATUS=$(gh run view "$RUN_ID" --repo "$REPO_SLUG" --json conclusion \
    --jq '.conclusion' 2>/dev/null || echo "unknown")
  if [[ "$STATUS" == "success" ]]; then
    ok "Windows build succeeded"
  else
    warn "Windows build status: $STATUS — check https://github.com/${REPO_SLUG}/actions"
  fi
else
  warn "Could not find CI run — check: https://github.com/${REPO_SLUG}/actions"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${BLD}━━  Release complete: ${VERSION_TAG}  ━━${RST}"
echo ""
gh release view "$VERSION_TAG" --repo "$REPO_SLUG" \
  --json assets --jq '.assets[] | "  " + .name + "  (" + (.size/1024/1024 | floor | tostring) + " MB)"' \
  2>/dev/null || true
echo ""
echo "  https://github.com/${REPO_SLUG}/releases/tag/${VERSION_TAG}"
echo ""
