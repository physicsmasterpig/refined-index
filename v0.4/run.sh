#!/bin/zsh
# run.sh — Thin wrapper kept for backwards compatibility.
# Prefer:  ./bin/kernel_build_start.sh
#
# Usage:
#   ./run.sh              # run all assigned tasks for this machine
#   ./run.sh --dry-run    # preview tasks without computing
#   ./run.sh --no-push    # compute but don't push to GitHub
#   ./run.sh --task 0     # run only the first assigned task
exec "$(dirname "$0")/bin/kernel_build_start.sh" "$@"
