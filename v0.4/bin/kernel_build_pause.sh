#!/bin/zsh
# bin/kernel_build_pause.sh — Signal a clean stop after the current slope finishes.
#
# The running computation will:
#   1. Finish the slope it is currently on (no partial data lost)
#   2. Push completed results to GitHub
#   3. Exit
#
# Run ./bin/kernel_build_resume.sh (or ./bin/kernel_build_start.sh) to continue.

cd "$(dirname "$0")/.."

PAUSE_FILE="$(pwd)/PAUSE"

if [ -f "$PAUSE_FILE" ]; then
  echo "Already paused (PAUSE file exists). Nothing to do."
  echo "Run ./bin/kernel_build_resume.sh (or ./bin/kernel_build_start.sh) to restart."
else
  touch "$PAUSE_FILE"
  echo "✅ Pause requested."
  echo "   The running computation will finish its current slope, push results, then stop."
  echo "   Run ./bin/kernel_build_resume.sh when ready to continue."
fi
