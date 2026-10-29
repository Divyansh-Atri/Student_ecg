"""Synthetic ECG generators for unit tests.

Provides reproducible timestamped signals with QRS-like beats, baseline drift,
and noise for testing loaders, filters, and quality scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_ecg(
    duration_s: float = 10.0,
    fs: float = 125.0,
    hr_bpm: float = 72.0,
    noise_level: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic ECG-like signal with QRS spikes, drift, and noise.

    Args:
        duration_s: Recording length in seconds.
        fs: Sampling rate in Hz.
        hr_bpm: Heart rate used for QRS spacing.
        noise_level: Scale factor for additive Gaussian noise.

    Returns:
        Tuple of (timestamps_ms, signal) with int64 timestamps and float64 samples.

    Notes:
        Timestamps start at 1_700_000_000_000 ms and advance by 1000/fs per sample.
    """
    n = max(2, int(round(duration_s * fs)))
    dt_ms = 1000.0 / fs
    timestamps_ms = (
        1_700_000_000_000 + np.arange(n, dtype=np.float64) * dt_ms
    ).astype(np.int64)

    t_s = np.arange(n, dtype=np.float64) / fs
    beat_period_s = 60.0 / max(hr_bpm, 1.0)
    beat_times = np.arange(0.0, duration_s + beat_period_s, beat_period_s)

    signal = np.zeros(n, dtype=np.float64)
    sigma = max(1.0 / fs, 0.008)
    for bt in beat_times:
        signal += 1.2 * np.exp(-0.5 * ((t_s - bt) / sigma) ** 2)

    signal += 0.1 * np.sin(2.0 * np.pi * 0.2 * t_s)
    if noise_level > 0:
        rng = np.random.default_rng(42)
        signal += noise_level * rng.standard_normal(n)

    return timestamps_ms, signal.astype(np.float64)


def make_synthetic_df(duration_s: float = 10.0, fs: float = 125.0) -> pd.DataFrame:
    """Return a DataFrame with timestamp_ms and ecg_value columns.

    Args:
        duration_s: Recording length in seconds.
        fs: Sampling rate in Hz.

    Returns:
        DataFrame suitable for loader/sanitizer tests.
    """
    ts, sig = make_synthetic_ecg(duration_s=duration_s, fs=fs)
    return pd.DataFrame({"timestamp_ms": ts, "ecg_value": sig})


def make_noisy_ecg(duration_s: float = 10.0, fs: float = 125.0) -> tuple[np.ndarray, np.ndarray]:
    """Build a heavily degraded synthetic ECG for quality-scorer tests.

    Args:
        duration_s: Recording length in seconds.
        fs: Sampling rate in Hz.

    Returns:
        Tuple of (timestamps_ms, signal) with strong noise and baseline wander.

    Notes:
        Uses noise_level=0.5 and baseline drift amplitude 0.8.
    """
    n = max(2, int(round(duration_s * fs)))
    dt_ms = 1000.0 / fs
    timestamps_ms = (
        1_700_000_000_000 + np.arange(n, dtype=np.float64) * dt_ms
    ).astype(np.int64)

    t_s = np.arange(n, dtype=np.float64) / fs
    hr_bpm = 72.0
    beat_period_s = 60.0 / hr_bpm
    beat_times = np.arange(0.0, duration_s + beat_period_s, beat_period_s)

    signal = np.zeros(n, dtype=np.float64)
    sigma = max(1.0 / fs, 0.008)
    for bt in beat_times:
        signal += 1.2 * np.exp(-0.5 * ((t_s - bt) / sigma) ** 2)

    signal += 0.8 * np.sin(2.0 * np.pi * 0.2 * t_s)
    rng = np.random.default_rng(99)
    signal += 0.5 * rng.standard_normal(n)

    return timestamps_ms, signal.astype(np.float64)
