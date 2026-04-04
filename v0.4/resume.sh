#!/bin/zsh
# resume.sh — Remove the PAUSE file and restart computation.
# Equivalent to just running ./run.sh, but removes PAUSE first.

cd "$(dirname "$0")"

PAUSE_FILE="$(pwd)/PAUSE"

if [ -f "$PAUSE_FILE" ]; then
  rm "$PAUSE_FILE"
  echo "✅ PAUSE file removed."
fi

echo "Starting computation..."
exec ./run.sh "$@"
