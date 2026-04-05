#!/bin/zsh
# pause.sh — Thin wrapper kept for backwards compatibility.
# Prefer:  ./bin/kernel_build_pause.sh
exec "$(dirname "$0")/bin/kernel_build_pause.sh" "$@"
