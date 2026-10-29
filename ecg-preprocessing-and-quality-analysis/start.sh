#!/usr/bin/env bash
# Run the full ECG preprocessing and ML pipeline end-to-end.
#
# Usage:
#   ./run_all.sh                          # auto-detects DATA/RAW/*_RAW (see below)
#   ./run_all.sh /path/to/raw/csv/folder  # custom input folder
#   ECG_INPUT=DATA/RAW/my_folder ./run_all.sh
#   SKIP_TESTS=1 ./run_all.sh             # skip pytest
#   SKIP_TRAIN=1 ./run_all.sh             # skip ML training (manifest still built)
#
# Steps:
#   1. Install Python dependencies
#   2. Run unit tests (27 tests)
#   3. Batch preprocess daily ECG CSVs (segmentation + beat windows)
#   4. Build segment labeling manifest (ml/segment_manifest.csv)
#   5. Train quality model (if manual labels exist in manifest)
#   6. Train arrhythmia model (if arrhythmia labels exist in manifest)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

OUTPUT_DIR="${ECG_OUTPUT:-Processed_ECG}"
VENV_DIR="${VENV_DIR:-.venv}"

# Resolve raw CSV input folder (this repo uses DATA/RAW/, not ECG_DATA/).
resolve_input_dir() {
  local candidate

  if [[ -n "${1:-}" ]]; then
    echo "$1"
    return 0
  fi
  if [[ -n "${ECG_INPUT:-}" ]]; then
    echo "$ECG_INPUT"
    return 0
  fi

  for candidate in \
    "DATA/RAW/18_05_2026_183594_RAW" \
    "ECG_DATA/DATA/RAW/18_05_2026_183594_RAW" \
    "ECG_DATA/data/18_05_2026_183594_RAW" \
    "ECG_DATA"; do
    if [[ -d "$candidate" ]] && compgen -G "$candidate"/*_ecg*.csv >/dev/null 2>&1; then
      echo "$candidate"
      return 0
    fi
  done

  if [[ -d DATA/RAW ]]; then
    candidate="$(find DATA/RAW -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | while read -r d; do
      if compgen -G "$d"/*_ecg*.csv >/dev/null 2>&1; then echo "$d"; break; fi
    done)"
    if [[ -n "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  fi

  return 1
}

INPUT_DIR="$(resolve_input_dir "${1:-}" || true)"

echo "==> Project root: $ROOT"
echo "==> Input folder:  ${INPUT_DIR:-<not found>}"
echo "==> Output folder: $OUTPUT_DIR"

# --- Dependencies ---
if [[ -d "$VENV_DIR" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
elif [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "==> Creating virtualenv at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck source=/dev/null
  source "$VENV_DIR/bin/activate"
fi

echo "==> Installing requirements"
pip install -q -r requirements.txt

# --- Tests ---
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
  echo "==> Running pytest (27 tests)"
  python -m pytest tests/ -v
else
  echo "==> Skipping tests (SKIP_TESTS=1)"
fi

# --- Daily preprocessing ---
if [[ -z "${INPUT_DIR:-}" ]] || [[ ! -d "$INPUT_DIR" ]]; then
  echo "ERROR: No raw ECG input folder found." >&2
  echo "Put *_ecg*.csv files in DATA/RAW/<your_day>_RAW/ (see README.md), then run:" >&2
  echo "  ./run_all.sh DATA/RAW/18_05_2026_183594_RAW" >&2
  exit 1
fi
if ! compgen -G "$INPUT_DIR"/*_ecg*.csv >/dev/null 2>&1; then
  echo "ERROR: No *_ecg*.csv files in: $INPUT_DIR" >&2
  exit 1
fi

echo "==> Running daily ECG preprocessing pipeline"
python -m scripts.process_daily_ecg \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR" \
  --target-fs 125 \
  --powerline-hz 50 \
  --segment-seconds 5

# --- ML manifest ---
echo "==> Building segment manifest"
python -m ml.label_segments

MANIFEST="$ROOT/ml/segment_manifest.csv"
if [[ ! -f "$MANIFEST" ]]; then
  echo "WARNING: Manifest not created at $MANIFEST" >&2
  exit 1
fi

# --- Optional training (requires human labels in manifest) ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  echo "==> Training quality model"
  python -m ml.train_quality_model

  echo "==> Training arrhythmia model"
  python -m ml.train_arrhythmia
else
  echo "==> Skipping ML training (SKIP_TRAIN=1)"
fi

echo "==> Done."
echo "    Processed outputs: $OUTPUT_DIR/"
echo "    Segment manifest:  $MANIFEST"
echo "    ML models:         ml/models/ (after labeling + training)"
