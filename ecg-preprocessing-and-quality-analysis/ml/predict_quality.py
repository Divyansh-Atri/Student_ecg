"""Inference helpers for the trained ECG segment quality model."""

from __future__ import annotations

from functools import lru_cache
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.extract_features import extract_features_from_file, extract_segment_features

_MODEL_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def load_quality_model() -> Tuple[object, List[str]]:
    """Load the pickled quality model and its feature name list.

    Returns:
        Tuple of (sklearn model, feature_names).

    Raises:
        FileNotFoundError: If model or feature names file is missing.
    """
    model_path = _MODEL_DIR / "quality_model.pkl"
    names_path = _MODEL_DIR / "quality_feature_names.json"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Quality model not found at {model_path}. "
            "Run ml/train_quality_model.py after labeling the manifest."
        )
    if not names_path.is_file():
        raise FileNotFoundError(f"Feature names not found at {names_path}.")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    feature_names = json.loads(names_path.read_text(encoding="utf-8"))
    return model, feature_names


def predict_segment_quality(
    segment: np.ndarray,
    fs: float = 125.0,
) -> Dict[str, object]:
    """Predict good/bad quality for a single ECG segment.

    Args:
        segment: 1D cleaned ECG samples.
        fs: Sampling rate in Hz.

    Returns:
        Dict with prediction (0/1), probability_good, and features_used.
    """
    model, feature_names = load_quality_model()
    feats = extract_segment_features(segment, fs=fs)
    row = np.asarray([[feats[n] for n in feature_names]], dtype=np.float64)
    pred = int(model.predict(row)[0])
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(row)[0]
        prob_good = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        prob_good = float(pred)
    return {
        "prediction": pred,
        "probability_good": prob_good,
        "features_used": feats,
    }


def _file_label_from_score(file_score: float) -> str:
    if file_score >= 90.0:
        return "EXCELLENT"
    if file_score >= 75.0:
        return "GOOD"
    if file_score >= 60.0:
        return "FAIR"
    return "POOR"


def predict_file_quality(
    cleaned_csv_path: str,
    fs: float = 125.0,
    segment_seconds: float = 5.0,
) -> Dict[str, object]:
    """Predict quality for all fixed windows in a cleaned ECG file.

    Args:
        cleaned_csv_path: Path to cleaned_ecg.csv.
        fs: Sampling rate in Hz.
        segment_seconds: Window length in seconds.

    Returns:
        Dict with segment_predictions, file_score (0-100), and file_label.
    """
    feat_df = extract_features_from_file(cleaned_csv_path, fs=fs, segment_seconds=segment_seconds)
    path = Path(cleaned_csv_path)
    df = pd.read_csv(path)
    col = "clean_ecg" if "clean_ecg" in df.columns else "ecg_value"
    signal = df[col].to_numpy(dtype=np.float64)
    n_per = int(round(segment_seconds * fs))

    preds: List[Dict[str, object]] = []
    good = 0
    total = 0
    for seg_idx in range(len(feat_df)):
        start = seg_idx * n_per
        end = start + n_per
        if end > signal.size:
            break
        result = predict_segment_quality(signal[start:end], fs=fs)
        result["segment_index"] = seg_idx
        preds.append(result)
        total += 1
        if int(result["prediction"]) == 1:
            good += 1

    file_score = 100.0 * good / total if total else 0.0
    return {
        "segment_predictions": preds,
        "file_score": file_score,
        "file_label": _file_label_from_score(file_score),
    }
