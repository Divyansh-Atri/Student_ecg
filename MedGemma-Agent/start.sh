#!/usr/bin/env bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/venv"
PID_FILE="$PROJECT_DIR/.server.pid"
LOG_FILE="$PROJECT_DIR/server.log"
PORT="${PORT:-8000}"

# ── Checks ────────────────────────────────────────────────────────────────────
if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server is already running (PID $(cat "$PID_FILE")). Run ./stop.sh first."
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "Virtual environment not found. Run setup first:"
  echo "  python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [ ! -f "$PROJECT_DIR/.env" ]; then
  echo "No .env file found. Copy .env.example and fill in your values:"
  echo "  cp .env.example .env"
  exit 1
fi

# Check ENCRYPTION_KEY is set
source "$PROJECT_DIR/.env" 2>/dev/null || true
if [ -z "$ENCRYPTION_KEY" ]; then
  echo "ENCRYPTION_KEY is not set in .env."
  echo "Generate one with:"
  echo "  python3 -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
  exit 1
fi

# Check Ollama is reachable
OLLAMA_URL="${OLLAMA_BASE_URL:-http://localhost:11434}"
if ! curl -sf "$OLLAMA_URL" > /dev/null 2>&1; then
  echo "Warning: Ollama is not reachable at $OLLAMA_URL."
  echo "The server will start but fall back to rule-based scoring only."
fi

# ── Start ─────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
if [ -f "$VENV/Scripts/activate" ]; then
  source "$VENV/Scripts/activate"
else
  source "$VENV/bin/activate"
fi

echo "Starting MedGemma Monitoring Agent on port $PORT..."
nohup python -m uvicorn api.main:app \
  --host 0.0.0.0 \
  --port "$PORT" \
  --log-level info \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
sleep 2

if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server started (PID $(cat "$PID_FILE"))"
  echo ""
  echo "  UI:       http://localhost:$PORT/ui"
  echo "  API docs: http://localhost:$PORT/docs"
  echo "  Health:   http://localhost:$PORT/api/v1/health"
  echo "  Logs:     tail -f $LOG_FILE"
else
  echo "Server failed to start. Check logs:"
  cat "$LOG_FILE"
  rm -f "$PID_FILE"
  exit 1
fi
