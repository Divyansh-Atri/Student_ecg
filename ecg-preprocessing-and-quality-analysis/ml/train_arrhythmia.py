"""Train a 1D CNN (or MLP fallback) for segment-level arrhythmia detection."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENT_LEN = 625


def _parse_arrhythmia_label(val: object) -> int | None:
    """Normalize arrhythmia_label to 0 or 1, or None if invalid."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s in ("0", "1"):
        return int(s)
    try:
        v = int(float(s))
        return v if v in (0, 1) else None
    except Exception:
        return None


def _derive_proxy_arrhythmia_label(row: pd.Series) -> int | None:
    """Derive a weak arrhythmia proxy label from available metadata.

    Uses manual arrhythmia labels when present; otherwise falls back to a
    simple heart-rate/quality heuristic so the model can still be trained on
    this dataset until manual rhythm labels are added.
    """
    lab = _parse_arrhythmia_label(row.get("arrhythmia_label"))
    if lab is not None:
        return lab

    hr = pd.to_numeric(row.get("heart_rate_bpm"), errors="coerce")
    quality = str(row.get("quality_label", "")).strip().upper()
    if not np.isfinite(hr):
        if quality in {"POOR", "UNUSABLE"}:
            return 1
        return None

    if hr < 50.0 or hr > 110.0:
        return 1
    if quality in {"POOR", "UNUSABLE"}:
        return 1
    return 0


def _load_segment_ecg_z(path: Path) -> np.ndarray | None:
    """Load ecg_z column from a segment CSV."""
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "ecg_z" not in df.columns:
        return None
    return df["ecg_z"].to_numpy(dtype=np.float64)


def _pad_truncate(x: np.ndarray, length: int = SEGMENT_LEN) -> np.ndarray:
    """Pad with zeros or truncate to fixed length."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size >= length:
        return x[:length].copy()
    out = np.zeros(length, dtype=np.float64)
    out[: x.size] = x
    return out


def train_arrhythmia_model(manifest_path: Path, output_dir: Path) -> None:
    """Train arrhythmia detector from labeled manifest rows.

    Args:
        manifest_path: segment_manifest.csv with quality and arrhythmia labels.
        output_dir: Directory for saved Keras or sklearn model files.
    """
    try:
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("scikit-learn is required. Install with: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    used_proxy = False

    for _, row in manifest.iterrows():
        lab = _derive_proxy_arrhythmia_label(row)
        if lab is None:
            continue
        seg = _load_segment_ecg_z(Path(str(row["segment_path"])))
        if seg is None:
            continue
        X_list.append(_pad_truncate(seg))
        y_list.append(lab)
        if _parse_arrhythmia_label(row.get("arrhythmia_label")) is None:
            used_proxy = True

    if not X_list:
        print(
            "No arrhythmia labels or usable proxy labels found. Add arrhythmia_label values or improve the proxy heuristic.",
            file=sys.stderr,
        )
        sys.exit(1)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int64)

    class_counts = pd.Series(y).value_counts().sort_index()
    if class_counts.size < 2:
        print(
            "Arrhythmia training requires at least two classes after label derivation. "
            f"Found: {class_counts.to_dict()}",
            file=sys.stderr,
        )
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    use_keras = False
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        use_keras = True
    except ImportError:
        print(
            "Warning: tensorflow not available; falling back to sklearn MLPClassifier.",
            file=sys.stderr,
        )

    if use_keras:
        X_train_k = X_train[..., np.newaxis]
        X_test_k = X_test[..., np.newaxis]

        model = keras.Sequential(
            [
                layers.Input(shape=(SEGMENT_LEN, 1)),
                layers.Conv1D(32, kernel_size=7, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Conv1D(64, kernel_size=5, padding="same", activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling1D(2),
                layers.Conv1D(128, kernel_size=3, padding="same", activation="relu"),
                layers.GlobalAveragePooling1D(),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            )
        ]
        model.fit(
            X_train_k,
            y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1,
        )
        y_prob = model.predict(X_test_k, verbose=0).ravel()
        y_pred = (y_prob >= 0.5).astype(int)
        out_path = output_dir / "arrhythmia_model.keras"
        model.save(out_path)
        print(f"Keras model saved to {out_path}")
    else:
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        out_path = output_dir / "arrhythmia_model_sklearn.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(clf, f)
        print(f"Sklearn MLP saved to {out_path}")

    meta_path = output_dir / "arrhythmia_model_metadata.json"
    meta_path.write_text(
        json.dumps(
            {
                "label_source": "proxy" if used_proxy else "manual",
                "class_counts": {str(k): int(v) for k, v in class_counts.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))


def main() -> None:
    """CLI entry: train from ml/segment_manifest.csv."""
    root = Path(__file__).resolve().parents[1]
    train_arrhythmia_model(root / "ml" / "segment_manifest.csv", root / "ml" / "models")


if __name__ == "__main__":
    main()
