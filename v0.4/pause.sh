#!/bin/zsh
# pause.sh — Ask run.sh to stop cleanly after the current task finishes.
#
# The running computation will:
#   1. Finish the slope/manifold it's currently on (atomic — no data lost)
#   2. Push completed results to GitHub
#   3. Exit
#
# Then run ./run.sh (or ./resume.sh) to continue later.

cd "$(dirname "$0")"

PAUSE_FILE="$(pwd)/PAUSE"

if [ -f "$PAUSE_FILE" ]; then
  echo "Already paused (PAUSE file exists). Nothing to do."
  echo "Run ./resume.sh (or ./run.sh) to start again."
else
  touch "$PAUSE_FILE"
  echo "✅ PAUSE requested."
  echo "   The running computation will finish its current slope, push results, then stop."
  echo "   Run ./resume.sh (or ./run.sh) when ready to continue."
fi
