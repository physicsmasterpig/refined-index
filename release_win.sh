#!/usr/bin/env bash
# release_win.sh — Mac에서 실행하는 Windows 빌드 자동화 스크립트
#
# ⚠️  Windows 빌드 전용
#     (Mac 빌드는 향후 release_mac.sh 로 통합될 예정)
#
# 하는 일:
#   1. 버전 문자열을 모든 관련 파일에 업데이트
#   2. 변경사항 커밋 & 푸시
#   3. git tag 생성 & 푸시 → GitHub Actions 자동 트리거
#   4. (gh CLI 설치 시) 빌드 진행 상황 실시간 추적
#
# Usage:
#   ./release_win.sh v1.0.0            # 버전 업데이트 + 태그 푸시
#   ./release_win.sh v1.0.0 --dry-run  # 실제 변경 없이 미리 보기
#
# 필요 조건 (Mac):
#   - git
#   - gh CLI (선택, 빌드 상태 추적용): brew install gh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── 인자 파싱 ─────────────────────────────────────────────────────
VERSION_ARG="${1:-}"
DRY_RUN=false
if [[ "${2:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# ── 버전 검증 ─────────────────────────────────────────────────────
if [[ -z "$VERSION_ARG" ]]; then
  echo "Usage: ./release_win.sh v<MAJOR>.<MINOR>.<PATCH> [--dry-run]"
  echo "  예: ./release_win.sh v1.0.0"
  exit 1
fi

VERSION_NUM="${VERSION_ARG#v}"   # 0.5.4
VERSION_TAG="v${VERSION_NUM}"   # v1.0.0

if ! [[ "$VERSION_NUM" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "오류: 버전 형식이 잘못됨 — 'v1.0.0' 형식으로 입력하세요."
  exit 1
fi

# ── 현재 버전 감지 및 일관성 확인 ────────────────────────────────
CURRENT_VERSION=$(grep '^version' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
VER_SPEC=$(grep 'APP_VERSION = ' ManifoldIndex_win.spec | sed 's/.*APP_VERSION = "\(.*\)"/\1/')
VER_DOCS=$(grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' docs/index.html | head -1 | sed 's/v//')

# 파일 간 버전 불일치 감지
if [[ "$CURRENT_VERSION" != "$VER_SPEC" || "$CURRENT_VERSION" != "$VER_DOCS" ]]; then
  echo "경고: 파일 간 버전이 일치하지 않습니다:"
  echo "  pyproject.toml       : $CURRENT_VERSION"
  echo "  ManifoldIndex_win.spec: $VER_SPEC"
  echo "  docs/index.html       : $VER_DOCS"
  echo ""
  echo "  가장 높은 버전을 기준으로 진행합니다: 수동으로 확인 후 실행하세요."
  # 가장 높은 버전을 현재 버전으로 사용
  CURRENT_VERSION=$(printf '%s\n' "$CURRENT_VERSION" "$VER_SPEC" "$VER_DOCS" \
    | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
  echo "  사용할 현재 버전: $CURRENT_VERSION"
  echo ""
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Refined Index Calculator — Release"
echo "  현재 버전: $CURRENT_VERSION"
echo "  새 버전  : $VERSION_NUM"
echo "  태그     : $VERSION_TAG"
if $DRY_RUN; then
echo "  모드     : DRY-RUN (실제 변경 없음)"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "$CURRENT_VERSION" == "$VERSION_NUM" ]]; then
  echo "경고: 새 버전이 현재 버전과 동일합니다 ($CURRENT_VERSION)."
  read -rp "계속 진행하시겠습니까? (y/N) " confirm
  [[ "$confirm" == "y" || "$confirm" == "Y" ]] || exit 0
fi

# ── git 상태 확인 ─────────────────────────────────────────────────
echo "[ 1/5 ] git 상태 확인..."

BRANCH=$(git rev-parse --abbrev-ref HEAD)
echo "  브랜치: $BRANCH"

if git rev-parse "$VERSION_TAG" >/dev/null 2>&1; then
  echo "오류: 태그 '$VERSION_TAG' 가 이미 존재합니다."
  exit 1
fi

# ── 버전 파일 업데이트 함수 ───────────────────────────────────────
do_sed() {
  local file="$1" pattern="$2" replacement="$3"
  if $DRY_RUN; then
    echo "    [dry-run] $file"
    echo "              '$pattern' → '$replacement'"
  else
    # macOS sed는 -i '' 필요
    sed -i '' "s|$pattern|$replacement|g" "$file"
  fi
}

# ── 버전 업데이트 ─────────────────────────────────────────────────
echo "[ 2/5 ] 버전 업데이트..."

# pyproject.toml
do_sed "pyproject.toml" \
  "version = \"$CURRENT_VERSION\"" \
  "version = \"$VERSION_NUM\""
echo "  pyproject.toml"

# ManifoldIndex_win.spec (Windows 빌드)
do_sed "ManifoldIndex_win.spec" \
  "APP_VERSION = \".*\"" \
  "APP_VERSION = \"$VERSION_NUM\""
echo "  ManifoldIndex_win.spec"

# ManifoldIndex.spec (macOS 빌드)
do_sed "ManifoldIndex.spec" \
  "APP_VERSION = \".*\"" \
  "APP_VERSION = \"$VERSION_NUM\""
echo "  ManifoldIndex.spec"

# build_app.sh (macOS 빌드 스크립트)
do_sed "build_app.sh" \
  "APP_VERSION=\".*\"" \
  "APP_VERSION=\"$VERSION_NUM\""
echo "  build_app.sh"

# build_app.bat (Windows 로컬 빌드 스크립트)
do_sed "build_app.bat" \
  "set APP_VERSION=.*" \
  "set APP_VERSION=$VERSION_NUM"
echo "  build_app.bat"

# docs/index.html — 다운로드 링크 및 버전 표시 업데이트
# macOS 다운로드 링크
do_sed "docs/index.html" \
  "releases/download/v${CURRENT_VERSION}/ManifoldIndex\.zip" \
  "releases/download/${VERSION_TAG}/ManifoldIndex.zip"
# Windows 다운로드 링크 (버전 태그만 업데이트, 파일명은 ManifoldIndex.exe 고정)
do_sed "docs/index.html" \
  "releases/download/v${CURRENT_VERSION}/ManifoldIndex\.exe" \
  "releases/download/${VERSION_TAG}/ManifoldIndex.exe"
# 버튼/heading의 버전 텍스트
do_sed "docs/index.html" \
  "v${CURRENT_VERSION}" \
  "v${VERSION_NUM}"
echo "  docs/index.html"

echo ""

# ── dry-run이면 여기서 종료 ───────────────────────────────────────
if $DRY_RUN; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  DRY-RUN 완료 — 실제 변경사항 없음"
  echo "  실제 실행: ./release_win.sh $VERSION_TAG"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  exit 0
fi

# ── 커밋 ──────────────────────────────────────────────────────────
echo "[ 3/5 ] 커밋..."
git diff --stat
git add \
  pyproject.toml \
  ManifoldIndex_win.spec \
  ManifoldIndex.spec \
  build_app.sh \
  build_app.bat \
  docs/index.html
git commit -m "chore: bump version to $VERSION_NUM"
echo "  완료"

# ── 푸시 & 태그 ───────────────────────────────────────────────────
echo "[ 4/5 ] 푸시 및 태그 생성..."
git push origin "$BRANCH"
echo "  브랜치 '$BRANCH' 푸시 완료"

git tag -f "$VERSION_TAG"
git push origin "$VERSION_TAG" --force
echo "  태그 '$VERSION_TAG' 푸시 완료"
echo "  → GitHub Actions 'Build Windows' 자동 트리거됨"

# ── 빌드 상태 추적 ────────────────────────────────────────────────
echo ""
echo "[ 5/5 ] 빌드 상태..."

REPO_SLUG=$(git remote get-url origin \
  | sed 's/git@github\.com://; s/https:\/\/github\.com\///; s/\.git$//')

if command -v gh >/dev/null 2>&1; then
  echo "  gh CLI 감지 — 빌드 완료까지 대기 (Ctrl+C로 중단 가능)"
  echo ""
  sleep 6  # Actions 트리거 대기
  gh run watch --repo "$REPO_SLUG" || true
  echo ""
  echo "  Release 다운로드 URL:"
  gh release view "$VERSION_TAG" \
    --repo "$REPO_SLUG" \
    --json assets \
    --jq '.assets[].browserDownloadUrl' 2>/dev/null || echo "  (Release 업로드 대기 중)"
else
  REPO_URL="https://github.com/$REPO_SLUG"
  echo "  빌드 상태 : $REPO_URL/actions"
  echo "  Release   : $REPO_URL/releases/tag/$VERSION_TAG"
  echo ""
  echo "  gh CLI 설치 시 빌드 상태를 터미널에서 직접 확인할 수 있습니다:"
  echo "    brew install gh && gh auth login"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  완료: $VERSION_TAG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
