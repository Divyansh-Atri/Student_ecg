"""Unit tests for scripts.clean_ecg core signal-processing helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.clean_ecg import (
    detect_noise,
    estimate_sampling_rate,
    filter_ecg,
    hampel_filter,
    load_ecg,
    resample_to_uniform,
    robust_clip,
    sanitize_ecg,
    score_ecg_quality,
)
from tests.fixtures.synthetic_ecg import make_noisy_ecg, make_synthetic_ecg


def _band_power(x: np.ndarray, fs: float, f_low: float, f_high: float) -> float:
    """Helper: integrated FFT power between f_low and f_high."""
    n = x.size
    if n < 8:
        return 0.0
    y = x.astype(np.float64) - float(np.mean(x))
    nfft = 1 << int(np.ceil(np.log2(max(1, n))))
    yf = np.fft.rfft(y, n=nfft)
    xf = np.fft.rfftfreq(nfft, d=1.0 / fs)
    power = np.abs(yf) ** 2
    mask = (xf >= f_low) & (xf <= f_high)
    return float(np.sum(power[mask])) if np.any(mask) else 0.0


def test_load_ecg_two_column_with_header(tmp_path):
    """load_ecg parses a standard header CSV with timestamp_ms and ecg_value."""
    path = tmp_path / "ecg.csv"
    ts0 = 1_700_000_000_000
    rows = [f"{ts0 + i * 8},{float(i)}" for i in range(20)]
    path.write_text("timestamp_ms,ecg_value\n" + "\n".join(rows), encoding="utf-8")

    df = load_ecg(str(path))
    assert list(df.columns) == ["timestamp_ms", "ecg_value"]
    assert len(df) == 20


def test_load_ecg_no_header(tmp_path):
    """load_ecg parses a two-column CSV without a header row."""
    path = tmp_path / "ecg.csv"
    ts0 = 1_700_000_000_000
    rows = [f"{ts0 + i * 8},{float(i)}" for i in range(20)]
    path.write_text("\n".join(rows), encoding="utf-8")

    df = load_ecg(str(path))
    assert list(df.columns) == ["timestamp_ms", "ecg_value"]
    assert len(df) == 20


def test_load_ecg_wide_pair_format(tmp_path):
    """load_ecg expands wide rows of timestamp,value pairs into long format."""
    path = tmp_path / "wide.csv"
    ts0 = 1_700_000_000_000
    lines = []
    for row in range(5):
        pairs = []
        for p in range(4):
            idx = row * 4 + p
            pairs.append(f"{ts0 + idx * 8}")
            pairs.append(f"{float(idx)}")
        lines.append(",".join(pairs))
    path.write_text("\n".join(lines), encoding="utf-8")

    df = load_ecg(str(path))
    assert len(df) == 20
    assert df["timestamp_ms"].is_monotonic_increasing


def test_sanitize_removes_duplicates():
    """sanitize_ecg drops duplicate timestamps, keeping the first occurrence."""
    df = pd.DataFrame(
        {
            "timestamp_ms": [100, 200, 200, 300],
            "ecg_value": [1.0, 2.0, 9.0, 3.0],
        }
    )
    out = sanitize_ecg(df)
    assert out["timestamp_ms"].duplicated().sum() == 0
    assert len(out) == 3


def test_sanitize_removes_non_increasing():
    """sanitize_ecg drops duplicate timestamps so time strictly increases."""
    df = pd.DataFrame(
        {
            "timestamp_ms": [100, 200, 200, 300],
            "ecg_value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = sanitize_ecg(df)
    assert list(out["timestamp_ms"].to_numpy()) == [100, 200, 300]


def test_sanitize_sorts():
    """sanitize_ecg sorts rows so timestamps are strictly increasing."""
    df = pd.DataFrame(
        {
            "timestamp_ms": [300, 100, 200],
            "ecg_value": [3.0, 1.0, 2.0],
        }
    )
    out = sanitize_ecg(df)
    ts = out["timestamp_ms"].to_numpy()
    assert np.all(np.diff(ts) > 0)


def test_estimate_sampling_rate_125hz():
    """estimate_sampling_rate detects ~125 Hz from 8 ms median spacing."""
    n = 200
    ts = (1_700_000_000_000 + np.arange(n) * 8).astype(np.int64)
    info = estimate_sampling_rate(ts)
    assert abs(info.sampling_rate_hz - 125.0) < 1.0


def test_estimate_sampling_rate_250hz():
    """estimate_sampling_rate detects ~250 Hz from 4 ms median spacing."""
    n = 200
    ts = (1_700_000_000_000 + np.arange(n) * 4).astype(np.int64)
    info = estimate_sampling_rate(ts)
    assert abs(info.sampling_rate_hz - 250.0) < 1.0


def test_robust_clip_removes_spike():
    """robust_clip limits a single extreme outlier to a robust range."""
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.5, size=200).astype(np.float64)
    x[100] = 99999.0
    y = robust_clip(x)
    assert y[100] < 100.0


def test_robust_clip_preserves_normal():
    """robust_clip leaves a Gaussian signal nearly unchanged."""
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, size=500)
    y = robust_clip(x)
    max_rel = float(np.max(np.abs(y - x)) / (np.max(np.abs(x)) + 1e-12))
    assert max_rel < 0.01


def test_hampel_filter_removes_spike():
    """hampel_filter replaces an isolated spike with a local median estimate."""
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t).astype(np.float64)
    x[50] = 9999.0
    y = hampel_filter(x, window_size=11)
    assert y[50] < 100.0


def test_hampel_filter_preserves_signal():
    """hampel_filter preserves a clean sinusoid within a small tolerance."""
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t).astype(np.float64)
    y = hampel_filter(x, window_size=11)
    assert float(np.max(np.abs(y - x))) < 0.5


def test_filter_ecg_reduces_baseline():
    """filter_ecg attenuates low-frequency baseline wander below 0.4 Hz."""
    fs = 125.0
    n = int(fs * 20)
    t = np.arange(n) / fs
    baseline = 2.0 * np.sin(2 * np.pi * 0.1 * t)
    _, sig = make_synthetic_ecg(duration_s=20.0, fs=fs, noise_level=0.0)
    x = sig[:n] + baseline[:n]
    p_in = _band_power(x, fs, 0.05, 0.4)
    y = filter_ecg(x, sampling_rate_hz=fs)
    p_out = _band_power(y, fs, 0.05, 0.4)
    assert p_out < 0.5 * p_in


def test_filter_ecg_reduces_highfreq():
    """filter_ecg attenuates 60 Hz interference when Nyquist allows."""
    fs = 250.0
    n = int(fs * 4)
    t = np.arange(n) / fs
    x = np.sin(2 * np.pi * 60.0 * t).astype(np.float64)
    p_in = _band_power(x, fs, 58.0, 62.0)
    y = filter_ecg(x, sampling_rate_hz=fs, powerline_hz=50.0)
    p_out = _band_power(y, fs, 58.0, 62.0)
    assert p_out < p_in


def test_resample_to_uniform_output_length():
    """resample_to_uniform yields uniform timestamp spacing near 1/fs."""
    fs = 125.0
    n = 500
    rng = np.random.default_rng(1)
    jitter = rng.integers(-1, 2, size=n)
    ts = (1_700_000_000_000 + np.cumsum(8 + jitter)).astype(np.int64)
    x = np.sin(np.linspace(0, 10, n)).astype(np.float64)
    ts_u, x_u = resample_to_uniform(ts, x, fs)
    assert ts_u.size == x_u.size
    if ts_u.size >= 3:
        d = np.diff(ts_u.astype(np.float64))
        assert float(np.std(d)) < 0.5


def test_score_ecg_quality_returns_valid_structure():
    """score_ecg_quality returns score, label, metrics, and recommendations."""
    _, sig = make_synthetic_ecg(duration_s=10.0, fs=125.0, noise_level=0.01)
    result = score_ecg_quality(
        cleaned=sig,
        raw=sig,
        sampling_rate_hz=125.0,
    )
    assert "score" in result
    assert "label" in result
    assert "metrics" in result
    assert "recommendations" in result
    assert 0.0 <= float(result["score"]) <= 100.0
    assert result["label"] in ("EXCELLENT", "GOOD", "FAIR", "POOR", "UNUSABLE")


def test_score_ecg_quality_clean_scores_higher_than_noisy():
    """A low-noise synthetic ECG scores at least as high as a noisy one."""
    _, clean = make_synthetic_ecg(duration_s=10.0, fs=125.0, noise_level=0.01)
    _, noisy = make_noisy_ecg(duration_s=10.0, fs=125.0)
    q_clean = score_ecg_quality(cleaned=clean, raw=clean, sampling_rate_hz=125.0)
    q_noisy = score_ecg_quality(cleaned=noisy, raw=noisy, sampling_rate_hz=125.0)
    assert float(q_clean["score"]) >= float(q_noisy["score"])


def test_detect_noise_finds_flatline():
    """detect_noise reports flatline segments on a constant signal region."""
    fs = 125.0
    n = 400
    x = np.ones(n, dtype=np.float64) * 3.7
    x[:200] = 0.0
    ts = (1_700_000_000_000 + np.arange(n) * 8).astype(np.int64)
    info = detect_noise(ts, x, fs)
    assert int(info["flatline_count"]) > 0


def test_detect_noise_finds_spikes():
    """detect_noise counts extreme spikes in an otherwise smooth signal."""
    _, sig = make_synthetic_ecg(duration_s=5.0, fs=125.0, noise_level=0.01)
    x = sig.copy()
    for idx in [10, 50, 90, 120, 200]:
        if idx < x.size:
            x[idx] = 9999.0
    ts = (1_700_000_000_000 + np.arange(x.size) * 8).astype(np.int64)
    info = detect_noise(ts, x, 125.0)
    assert int(info["spike_count"]) > 0
