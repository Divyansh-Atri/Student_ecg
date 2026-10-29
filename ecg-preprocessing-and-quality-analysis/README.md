# ECG Preprocessing and Quality Analysis

This repository provides a reproducible pipeline to clean, preprocess, and score the quality of ECG (electrocardiogram) recordings. It is intended for researchers and engineers who want high-quality ECG signals for downstream analysis or ML.

Key features:
- Signal cleaning (filters, artifact handling, timestamp fixes)
- Beat and segment-level extraction
- Heuristic quality scoring for segments and full recordings
- Utilities for batch processing and producing reports/plots
- Simple ML scripts for training and predicting quality / arrhythmia labels

---

**Contents and quick reference**

The project root contains the main scripts, ML helpers, and data folders used by the pipeline. The important top-level items are:

- `DATA/` — Input data. Put raw ECG CSV files under `DATA/RAW/`.
- `Processed_ECG/` — Output directory for cleaned files, plots, and reports.
- `ALL_CLEANED_ECG_CSV/` — Consolidated cleaned CSV exports for quick access.
- `scripts/` — Main processing scripts and helpers (e.g. `clean_ecg.py`, `process_daily_ecg.py`, `score_first_csv.py`).
- `ml/` — Feature extraction, model training and prediction scripts (e.g. `extract_features.py`, `predict_quality.py`, `train_quality_model.py`). Trained model files and metadata are in `ml/models/`.
- `docs/` — Notes, design documents, and example outputs.
- `tests/` — Unit tests (use `pytest`).
- `requirements.txt` — Python package dependencies.
- `start.sh` — Convenience script to activate the virtual environment and run quick checks.

---

## Installation

1. Create and activate a virtual environment (Unix/macOS):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. (Optional) Use `start.sh` to activate the venv and run recommended quick checks.

---

## Quick Usage

1) Process a single file (quick test):

```bash
python -m scripts.score_first_csv
```

2) Batch preprocess a folder of raw ECG CSVs:

```bash
python -m scripts.process_daily_ecg --input DATA/RAW/<your_folder> --output Processed_ECG --target-fs 125 --powerline-hz 50 --segment-seconds 5
```

3) Run the cleaning helper directly on a file (script interface may vary):

```bash
python -m scripts.clean_ecg --input DATA/RAW/example.csv --output Processed_ECG/example/ --target-fs 125
```

4) Score/predict quality using the ML helper:

```bash
python -m ml.predict_quality --input Processed_ECG/<recording>/cleaned.csv --model ml/models/quality_model.joblib
```

5) Train a quality model (example):

```bash
python -m ml.train_quality_model --features ml/segment_manifest.csv --output-model ml/models/quality_model.joblib
```

Note: check the top of each script for exact CLI arguments (some modules expose `argparse` options).

---

## Outputs and Where to Look

- `Processed_ECG/` — per-recording directories with:
	- cleaned CSVs and per-segment CSVs
	- plots: raw_vs_cleaned.png, frequency_spectrum.png, rpeaks/diagnostics
	- `daily_report/` — aggregated quality trends and summary reports
- `ALL_CLEANED_ECG_CSV/` — convenient combined cleaned CSVs for analysis
- `ml/models/` — saved model artifacts and JSON metadata (e.g. `arrhythmia_model_metadata.json`, `quality_feature_names.json`)

---

## Testing

Run unit tests with `pytest`:

```bash
pytest -q
```

Add tests under `tests/` when you change code.

---

## Notes on Data and Licensing

- This project does not provide clinical diagnoses — it prepares signals for analysis only.
- Ensure you have permission to process patient or proprietary ECG data before using this pipeline.

---

## Developer Notes

- Scripts are implemented in `scripts/` and ML routines live in `ml/`.
- If you add new models, update `ml/models/` and include a corresponding metadata JSON.
- If you reorganize output folders, update `scripts/process_daily_ecg.py` and any CI/test config that depends on those paths.

---

## Getting Help / Contributing

If you find bugs or want to add features, open an issue or submit a pull request. For quick questions, see the `docs/` folder and inline code comments.

---

Last updated: 2026-05-22