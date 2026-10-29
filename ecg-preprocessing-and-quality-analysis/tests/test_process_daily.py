"""Unit tests for scripts.process_daily_ecg batch utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.process_daily_ecg import (
    compute_irregular_gap_fraction,
    parse_ecg_filename,
    segment_into_windows,
    zscore_normalize,
)


def test_parse_ecg_filename_valid():
    """parse_ecg_filename extracts timestamp and patch ID from a standard name."""
    ts, patch = parse_ecg_filename("1779043703843_VC2B008BF_183594_ecg.csv")
    assert ts == 1779043703843
    assert patch == "VC2B008BF"


def test_parse_ecg_filename_invalid():
    """parse_ecg_filename raises ValueError for non-conforming filenames."""
    with pytest.raises(ValueError):
        parse_ecg_filename("bad_name.csv")


def test_zscore_normalize_mean_zero():
    """zscore_normalize centers the signal to approximately zero mean."""
    rng = np.random.default_rng(7)
    x = rng.normal(5.0, 2.0, size=1000)
    norm, _, _ = zscore_normalize(x)
    assert abs(float(np.mean(norm))) < 1e-9


def test_zscore_normalize_std_one():
    """zscore_normalize scales the signal to unit standard deviation."""
    rng = np.random.default_rng(7)
    x = rng.normal(5.0, 2.0, size=1000)
    norm, _, _ = zscore_normalize(x)
    assert abs(float(np.std(norm)) - 1.0) < 1e-6


def test_compute_irregular_gap_fraction_clean():
    """Uniform 8 ms spacing yields a near-zero irregular gap fraction."""
    n = 500
    delta = np.full(n - 1, 8, dtype=np.int64)
    frac = compute_irregular_gap_fraction(delta)
    assert frac < 0.01


def test_compute_irregular_gap_fraction_gappy():
    """Large timestamp gaps increase the irregular gap fraction above 5%."""
    n = 500
    delta = np.full(n - 1, 8, dtype=np.int64)
    gap_positions = np.linspace(10, n - 12, 30, dtype=int)
    delta[gap_positions] = 200
    frac = compute_irregular_gap_fraction(delta)
    assert frac > 0.05


def test_segment_into_windows_count(tmp_path):
    """segment_into_windows writes one file per non-overlapping window."""
    n = 1250
    ts = (1_700_000_000_000 + np.arange(n) * 8).astype(np.int64)
    cleaned = np.sin(np.linspace(0, 20 * np.pi, n)).astype(np.float64)
    normalized, _, _ = zscore_normalize(cleaned)
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    count = segment_into_windows(
        seg_dir,
        timestamps_ms=ts,
        cleaned=cleaned,
        normalized=normalized,
        sampling_rate_hz=125.0,
        segment_seconds=5.0,
    )
    assert count == 2


def test_segment_into_windows_content(tmp_path):
    """Each segment CSV has 625 rows and expected column names."""
    n = 1250
    ts = (1_700_000_000_000 + np.arange(n) * 8).astype(np.int64)
    cleaned = np.sin(np.linspace(0, 20 * np.pi, n)).astype(np.float64)
    normalized, _, _ = zscore_normalize(cleaned)
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    segment_into_windows(
        seg_dir,
        timestamps_ms=ts,
        cleaned=cleaned,
        normalized=normalized,
        sampling_rate_hz=125.0,
        segment_seconds=5.0,
    )
    seg_path = seg_dir / "segment_001.csv"
    df = pd.read_csv(seg_path)
    assert len(df) == 625
    assert list(df.columns) == ["timestamp_ms", "clean_ecg", "ecg_z"]
