#!/bin/zsh
# resume.sh — Thin wrapper kept for backwards compatibility.
# Prefer:  ./bin/kernel_build_resume.sh
exec "$(dirname "$0")/bin/kernel_build_resume.sh" "$@"
