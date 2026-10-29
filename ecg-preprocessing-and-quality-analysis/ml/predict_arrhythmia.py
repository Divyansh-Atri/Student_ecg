"""Inference helpers for arrhythmia detection on ECG segments."""

from __future__ import annotations

from functools import lru_cache
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml.train_arrhythmia import SEGMENT_LEN, _pad_truncate

_MODEL_DIR = Path(__file__).resolve().parent / "models"


@lru_cache(maxsize=1)
def load_arrhythmia_model() -> Tuple[object, str]:
    """Load Keras or sklearn arrhythmia model from ml/models.

    Returns:
        Tuple of (model, model_type) where model_type is 'keras' or 'sklearn'.

    Raises:
        FileNotFoundError: If neither model file exists.
    """
    keras_path = _MODEL_DIR / "arrhythmia_model.keras"
    sklearn_path = _MODEL_DIR / "arrhythmia_model_sklearn.pkl"

    if keras_path.is_file():
        try:
            import tensorflow as tf

            model = tf.keras.models.load_model(keras_path)
            return model, "keras"
        except Exception as exc:
            raise FileNotFoundError(
                f"Found {keras_path} but failed to load TensorFlow model: {exc}"
            ) from exc

    if sklearn_path.is_file():
        with open(sklearn_path, "rb") as f:
            return pickle.load(f), "sklearn"

    raise FileNotFoundError(
        f"No arrhythmia model found in {_MODEL_DIR}. "
        "Run ml/train_arrhythmia.py after labeling the manifest."
    )


def predict_segment_arrhythmia(segment: np.ndarray) -> Dict[str, object]:
    """Predict arrhythmia for one normalized ECG segment.

    Args:
        segment: ecg_z samples (any length; padded/truncated to 625).

    Returns:
        Dict with prediction, probability_arrhythmia, and model_type.
    """
    model, model_type = load_arrhythmia_model()
    x = _pad_truncate(segment)

    if model_type == "keras":
        prob = float(model.predict(x[np.newaxis, :, np.newaxis], verbose=0)[0, 0])
    else:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(x.reshape(1, -1))[0]
            prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            prob = float(model.predict(x.reshape(1, -1))[0])

    pred = 1 if prob >= 0.5 else 0
    return {
        "prediction": pred,
        "probability_arrhythmia": prob,
        "model_type": model_type,
    }


def predict_file_arrhythmia(
    normalized_csv_path: str,
    segment_seconds: float = 5.0,
    fs: float = 125.0,
) -> Dict[str, object]:
    """Run arrhythmia inference on fixed windows from normalized_ecg.csv.

    Args:
        normalized_csv_path: Path to normalized_ecg.csv (ecg_z column).
        segment_seconds: Window length in seconds.
        fs: Sampling rate in Hz.

    Returns:
        Dict with segment_predictions, arrhythmia_burden, overall_flag, summary.
    """
    df = pd.read_csv(normalized_csv_path)
    if "ecg_z" not in df.columns:
        raise ValueError("normalized CSV must contain ecg_z column")
    signal = df["ecg_z"].to_numpy(dtype=np.float64)
    n_per = int(round(segment_seconds * fs))

    preds: List[Dict[str, object]] = []
    flagged = 0
    total = 0

    for seg_idx in range(0, max(0, signal.size - n_per + 1), n_per):
        chunk = signal[seg_idx : seg_idx + n_per]
        if chunk.size < n_per:
            break
        result = predict_segment_arrhythmia(chunk)
        result["segment_index"] = len(preds)
        preds.append(result)
        total += 1
        if int(result["prediction"]) == 1:
            flagged += 1

    burden = float(flagged / total) if total else 0.0
    overall = burden > 0.2
    summary = (
        f"Arrhythmia burden {burden:.1%} ({flagged}/{total} segments); "
        f"overall_flag={'YES' if overall else 'NO'}"
    )
    return {
        "segment_predictions": preds,
        "arrhythmia_burden": burden,
        "overall_flag": overall,
        "summary": summary,
    }
