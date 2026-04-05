#!/bin/zsh
# bin/datapack_publish.sh — Package cache files into a release tarball and
# update data_packs.json, ready for upload to a GitHub Release.
#
# Usage:
#   # Pack all qq=50 kernels (|P|≤5, Q=1–5):
#   ./bin/datapack_publish.sh --type kernels --qq 50 --tag data-v1
#
#   # Pack iref cache:
#   ./bin/datapack_publish.sh --type iref --qq 50 --tag data-v1
#
#   # Pack NC cycle cache:
#   ./bin/datapack_publish.sh --type nc --qq 20 --tag data-v1
#
#   # Dry run (list files, write nothing):
#   ./bin/datapack_publish.sh --type kernels --qq 50 --dry-run
#
# Output tarball is written to v0.4/dist/.
# data_packs.json is updated automatically — commit it after uploading.

set -e
cd "$(dirname "$0")/.."

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
  echo "ERROR: No suitable Python found."
  exit 1
fi

export MANIFOLD_INDEX_CACHE_DIR="$(pwd)/cache"
"$PYTHON" scripts/publish_datapack.py "$@"
