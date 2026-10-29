#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$PROJECT_DIR/.server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file found. Server may not be running."
  exit 0
fi

PID=$(cat "$PID_FILE")

if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  rm -f "$PID_FILE"
  echo "Server stopped (PID $PID)."
else
  echo "Process $PID not found. Cleaning up stale PID file."
  rm -f "$PID_FILE"
fi
