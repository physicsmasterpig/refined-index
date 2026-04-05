#!/bin/zsh
# bin/kernel_build_resume.sh — Remove the pause signal and restart computation.
#
# Clears the PAUSE file (if present) then delegates to kernel_build_start.sh.
# Accepts the same flags as kernel_build_start.sh.

cd "$(dirname "$0")/.."

PAUSE_FILE="$(pwd)/PAUSE"

if [ -f "$PAUSE_FILE" ]; then
  rm "$PAUSE_FILE"
  echo "✅ Pause signal cleared."
fi

echo "Restarting kernel pre-computation..."
exec "$(dirname "$0")/kernel_build_start.sh" "$@"
