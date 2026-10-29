#!/usr/bin/env bash
set -e

echo "======================================================"
echo "    Starting VitalModel Full System Components        "
echo "======================================================"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Start MedGemma-Agent (API and UI)
echo ""
echo "[1/2] Starting MedGemma-Agent Host and UI..."
if [ -x "$PROJECT_ROOT/MedGemma-Agent/start.sh" ]; then
    cd "$PROJECT_ROOT/MedGemma-Agent"
    ./start.sh
else
    echo "Warning: MedGemma-Agent/start.sh not found or not executable."
    echo "Attempting to run anyway via sh..."
    if [ -f "$PROJECT_ROOT/MedGemma-Agent/start.sh" ]; then
        cd "$PROJECT_ROOT/MedGemma-Agent"
        sh ./start.sh
    else
        echo "Error: MedGemma-Agent start script is completely missing."
    fi
fi

# 2. Run ECG Preprocessing Pipeline
echo ""
echo "[2/2] Running ECG Preprocessing Pipeline..."
if [ -x "$PROJECT_ROOT/ecg-preprocessing-and-quality-analysis/start.sh" ]; then
    cd "$PROJECT_ROOT/ecg-preprocessing-and-quality-analysis"
    ./start.sh "$@"
else
    echo "Warning: ecg-preprocessing-and-quality-analysis/start.sh not found or not executable."
    echo "Attempting to run anyway via sh..."
    if [ -f "$PROJECT_ROOT/ecg-preprocessing-and-quality-analysis/start.sh" ]; then
        cd "$PROJECT_ROOT/ecg-preprocessing-and-quality-analysis"
        sh ./start.sh "$@"
    else
        echo "Error: ECG Preprocessing start script is completely missing."
    fi
fi

echo ""
echo "======================================================"
echo "                   Startup Complete                   "
echo "======================================================"
echo "MedGemma Agent is running in the background."
echo "ECG Pipeline execution has finished."
echo "To stop the MedGemma Agent later, run: ./MedGemma-Agent/stop.sh"
echo "To view live logs, run: tail -f MedGemma-Agent/server.log"
