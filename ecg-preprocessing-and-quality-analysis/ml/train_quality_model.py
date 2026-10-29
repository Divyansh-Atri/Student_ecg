"""Train a gradient-boosting classifier for manual ECG segment quality labels."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ml.extract_features import _feature_names, extract_segment_features

FEATURE_COLS = list(_feature_names())


def _label_to_binary(label: str) -> int:
    """Map quality_label to binary good (1) vs bad (0)."""
    u = str(label).strip().upper()
    return 1 if u in ("EXCELLENT", "GOOD") else 0


def train_quality_model(manifest_path: Path, output_dir: Path) -> None:
    """Train and persist a segment-level quality classifier.

    Args:
        manifest_path: CSV from prepare_training_manifest.
        output_dir: Directory for quality_model.pkl and feature names JSON.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    except ImportError:
        print(
            "scikit-learn is required for training. Install with: pip install scikit-learn",
            file=sys.stderr,
        )
        sys.exit(1)

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        print(f"Error: Manifest {manifest_path} not found.", file=sys.stderr)
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)

    # Use manual_quality_label if available, else auto_quality_label (Task 5 requirement)
    if "manual_quality_label" in manifest.columns and manifest["manual_quality_label"].notna().any():
        label_col = "manual_quality_label"
        print("Using manual_quality_label for training.")
    elif "auto_quality_label" in manifest.columns:
        label_col = "auto_quality_label"
        print("Using auto_quality_label for training.")
    else:
        label_col = "quality_label"
        print("Using quality_label for training.")

    labeled = manifest[manifest[label_col].notna()]
    if labeled.empty:
        print(f"Error: No labeled rows found in column {label_col}.", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting features from {len(labeled)} segments...")
    X_rows: list[list[float]] = []
    y_rows: list[int] = []

    for _, row in labeled.iterrows():
        seg_path = Path(str(row["segment_path"]))
        if not seg_path.is_file():
            continue
        try:
            df = pd.read_csv(seg_path)
            col = "clean_ecg" if "clean_ecg" in df.columns else "ecg_value"
            feats = extract_segment_features(df[col].to_numpy(dtype=np.float64), fs=125.0)
            X_rows.append([feats[c] for c in FEATURE_COLS])
            y_rows.append(_label_to_binary(row[label_col]))
        except Exception:
            continue

    if not X_rows:
        print("No valid segment data could be loaded.", file=sys.stderr)
        sys.exit(1)

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)

    # Print class balance
    unique, counts = np.unique(y, return_counts=True)
    balance = dict(zip(unique, counts))
    print(f"Class balance: 0 (Bad) = {balance.get(0, 0)}, 1 (Good) = {balance.get(1, 0)}")

    if len(balance) < 2:
        print("Error: Training requires both Good and Bad samples.", file=sys.stderr)
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    
    # Print cross-validation F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
    print(f"Cross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {np.mean(cv_scores):.4f}")

    # Train and print test F1
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred)
    print(f"Test F1: {test_f1:.4f}")
    print(classification_report(y_test, y_pred))

    # Final fit on all data
    model.fit(X, y)
    
    # Save to ml/models/quality_model.pkl (Task 5 requirement)
    model_out = output_dir / "quality_model.pkl"
    with open(model_out, "wb") as f:
        pickle.dump(model, f)
    
    (output_dir / "quality_feature_names.json").write_text(
        json.dumps(FEATURE_COLS, indent=2), encoding="utf-8"
    )
    print(f"Model saved to {model_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="ml/segment_manifest.csv")
    parser.add_argument("--output-dir", default="ml/models")
    args = parser.parse_args()
    
    train_quality_model(Path(args.manifest), Path(args.output_dir))


if __name__ == "__main__":
    main()
