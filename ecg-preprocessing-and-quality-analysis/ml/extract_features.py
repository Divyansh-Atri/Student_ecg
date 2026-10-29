"""Per-segment ECG feature extraction for ML quality models.

Computes interpretable time- and frequency-domain statistics on cleaned ECG
windows for training and inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def _safe_float(value: float) -> float:
    """Return a finite float or 0.0 on failure."""
    if not np.isfinite(value):
        return 0.0
    return float(value)


def extract_segment_features(
    segment: np.ndarray,
    fs: float = 125.0,
    powerline_hz: float = 50.0,
) -> Dict[str, float]:
    """Extract a fixed feature vector from one ECG segment.

    Args:
        segment: 1D cleaned ECG samples.
        fs: Sampling rate in Hz.
        powerline_hz: Mains frequency for powerline_ratio (±2 Hz band).

    Returns:
        Dictionary of scalar features (0.0 when computation fails).

    Notes:
        All ratios are clamped to [0, 1] where specified in the spec.
    """
    x = np.asarray(segment, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return {k: 0.0 for k in _feature_names()}

    try:
        from scipy import stats
        from scipy.signal import butter, sosfiltfilt
    except Exception:
        stats = None
        butter = None
        sosfiltfilt = None

    mean = _safe_float(np.mean(x))
    std = _safe_float(np.std(x))
    std = std if std > 1e-12 else 1e-12

    kurtosis = 0.0
    skewness = 0.0
    if stats is not None:
        try:
            kurtosis = _safe_float(float(stats.kurtosis(x, fisher=True, nan_policy="omit")))
            skewness = _safe_float(float(stats.skew(x, nan_policy="omit")))
        except Exception:
            pass

    p1, p5, p95, p99 = np.percentile(x, [1, 5, 95, 99])
    range_p95 = _safe_float(p95 - p5)
    range_p99 = _safe_float(p99 - p1)
    range_p95 = range_p95 if range_p95 > 1e-12 else std

    rms = _safe_float(np.sqrt(np.mean(x ** 2)))
    zcr = _safe_float(np.mean(np.abs(np.diff(np.signbit(x - mean)))))

    baseline_pp = 0.0
    if butter is not None and sosfiltfilt is not None and np.isfinite(fs) and fs > 1.0:
        try:
            cutoff = min(0.5, 0.45 * fs)
            cutoff = max(cutoff, 0.05)
            sos = butter(N=2, Wn=cutoff / (fs / 2.0), btype="low", output="sos")
            baseline = sosfiltfilt(sos, x)
            baseline_pp = _safe_float(np.max(baseline) - np.min(baseline))
        except Exception:
            baseline_pp = 0.0

    drift_ratio = _safe_float(baseline_pp / range_p95)
    drift_ratio = float(np.clip(drift_ratio, 0.0, 1.0))

    powerline_ratio = 0.0
    hf_ratio = 0.0
    try:
        n = x.size
        x0 = x - float(np.median(x))
        nfft = 1 << int(np.ceil(np.log2(max(1, n))))
        yf = np.fft.rfft(x0, n=nfft)
        xf = np.fft.rfftfreq(nfft, d=1.0 / float(fs))
        power = (np.abs(yf) ** 2).astype(np.float64)

        def bp(f0: float, f1: float) -> float:
            mask = (xf >= f0) & (xf <= f1)
            return float(np.sum(power[mask])) if np.any(mask) else 0.0

        ecg_band = bp(0.5, min(40.0, 0.45 * fs))
        ecg_band = ecg_band if ecg_band > 1e-12 else 1e-12
        pl = bp(float(powerline_hz) - 2.0, float(powerline_hz) + 2.0)
        hf = bp(40.0, min(0.5 * fs, 80.0))
        powerline_ratio = float(np.clip(pl / ecg_band, 0.0, 1.0))
        hf_ratio = float(np.clip(hf / ecg_band, 0.0, 1.0))
    except Exception:
        pass

    win = max(1, int(round(0.5 * fs)))
    flat_count = 0
    n_wins = 0
    for start in range(0, x.size, win):
        chunk = x[start : start + win]
        if chunk.size < max(4, win // 2):
            continue
        n_wins += 1
        if float(np.std(chunk)) < 0.02 * std:
            flat_count += 1
    flatline_fraction = _safe_float(flat_count / n_wins) if n_wins else 0.0

    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = xmax - xmin
    if span < 1e-12:
        clipping_ratio = 1.0 if xmin != 0 else 0.0
    else:
        tol = 0.01 * span
        clip_mask = (x <= xmin + tol) | (x >= xmax - tol)
        clipping_ratio = _safe_float(np.mean(clip_mask))

    autocorr_lag1 = 0.0
    if x.size >= 3 and std > 1e-12:
        xc = x - mean
        denom = float(np.sum(xc ** 2))
        if denom > 1e-12:
            autocorr_lag1 = _safe_float(float(np.sum(xc[:-1] * xc[1:]) / denom))

    return {
        "mean": mean,
        "std": std,
        "kurtosis": kurtosis,
        "skewness": skewness,
        "range_p95": range_p95,
        "range_p99": range_p99,
        "rms": rms,
        "zero_crossing_rate": zcr,
        "baseline_pp": baseline_pp,
        "drift_ratio": drift_ratio,
        "powerline_ratio": powerline_ratio,
        "hf_ratio": hf_ratio,
        "flatline_fraction": flatline_fraction,
        "clipping_ratio": clipping_ratio,
        "autocorr_lag1": autocorr_lag1,
    }


def _feature_names() -> tuple[str, ...]:
    return (
        "mean",
        "std",
        "kurtosis",
        "skewness",
        "range_p95",
        "range_p99",
        "rms",
        "zero_crossing_rate",
        "baseline_pp",
        "drift_ratio",
        "powerline_ratio",
        "hf_ratio",
        "flatline_fraction",
        "clipping_ratio",
        "autocorr_lag1",
    )


def extract_features_from_file(
    cleaned_csv_path: str,
    fs: float = 125.0,
    segment_seconds: float = 5.0,
) -> pd.DataFrame:
    """Split a cleaned ECG CSV into windows and extract features per window.

    Args:
        cleaned_csv_path: Path to cleaned_ecg.csv (timestamp_ms, clean_ecg).
        fs: Sampling rate in Hz.
        segment_seconds: Window length in seconds.

    Returns:
        DataFrame with one row per segment and a segment_index column.
    """
    path = Path(cleaned_csv_path)
    df = pd.read_csv(path)
    col = "clean_ecg" if "clean_ecg" in df.columns else "ecg_value"
    signal = df[col].to_numpy(dtype=np.float64)

    n_per = int(round(float(segment_seconds) * float(fs)))
    if n_per <= 0 or signal.size < n_per:
        return pd.DataFrame(columns=[*_feature_names(), "segment_index"])

    rows: list[Dict[str, float]] = []
    seg_idx = 0
    for start in range(0, signal.size - n_per + 1, n_per):
        end = start + n_per
        feats = extract_segment_features(signal[start:end], fs=fs)
        feats["segment_index"] = float(seg_idx)
        rows.append(feats)
        seg_idx += 1

    return pd.DataFrame(rows)
