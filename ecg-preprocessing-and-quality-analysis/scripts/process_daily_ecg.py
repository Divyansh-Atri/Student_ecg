"""process_daily_ecg.py

Production-style daily ECG preprocessing + dataset organization pipeline.

This script is the batch/orchestration counterpart of `clean_ecg.py`.
It *reuses* the robust loader/sanitizer/filter/noise helpers from `clean_ecg.py`
and adds the daily pipeline requirements needed to create an ML-ready dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import neurokit2 as nk

# Reuse the robust core helpers from scripts.clean_ecg (updated scoring).
# Import all helpers, including recalibrated quality scoring
from scripts.beat_segmentation import (
    beat_similarity_scores,
    compute_beat_template,
    save_beat_template_plot,
    segment_by_beats,
)
from scripts.clean_ecg import (
    SamplingInfo,
    SignalStats,
    PeakMetrics,
    baseline_drift_peak_to_peak,
    compute_stats,
    detect_noise,
    detect_rpeaks_robust,
    estimate_sampling_rate,
    estimate_snr_improvement,
    filter_ecg,
    hampel_filter,
    load_ecg,
    resample_to_uniform,
    robust_clip,
    sanitize_ecg,
    QUALITY_LABELS,
    QUALITY_THRESHOLDS,
    score_ecg_quality,
)

DATE_RE_DDMMYYYY = re.compile(r"(\d{2}_\d{2}_\d{4})")
DATE_RE_YYYYMMDD = re.compile(r"(\d{4}-\d{2}-\d{2})")


def safe_int(val: object, default: int = 0) -> int:
    """Best-effort conversion to int for loosely-typed metadata values."""

    try:
        if isinstance(val, (int, np.integer)):
            return int(val)
        if isinstance(val, (float, np.floating)):
            return int(val)
        return int(float(str(val)))
    except Exception:
        return default


@dataclass(frozen=True)
class PipelineConfig:
    target_fs_hz: float = 125.0
    powerline_hz: float = 50.0
    segment_seconds: float = 5.0


@dataclass
class FileResult:
    filename: str
    recording_timestamp_ms: Optional[int]
    patch_id: Optional[str]
    group_name: str
    success: bool
    error_message: Optional[str] = None

    duration_s: float = float("nan")
    sampling_rate_est_hz: float = float("nan")
    irregular_gap_fraction: float = float("nan")

    quality_score_0_100: float = float("nan")
    quality_label: str = ""

    flatline_count: int = 0
    spike_count: int = 0
    jump_count: int = 0

    variance_clean: float = float("nan")
    heart_rate_bpm: float = float("nan")
    rpeak_count: int = 0


# ============================================================================
# LOGGING
# ============================================================================


def build_console_logger() -> logging.Logger:
    logger = logging.getLogger("ecg_daily")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    return logger


def attach_group_file_logger(logger: logging.Logger, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler):
            logger.removeHandler(h)

    fh = logging.FileHandler(str(log_path), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(funcName)s:%(lineno)d | %(message)s",
            "%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(fh)


# ============================================================================
# DISCOVERY / NAMING
# ============================================================================


def parse_ecg_filename(filename: str) -> Tuple[int, str]:
    """Parse ECG filename -> (recording_timestamp_ms, patch_id)."""

    stem = Path(filename).stem
    stem = re.sub(r"\s*\(\d+\)$", "", stem)  # handles "... (1).csv"

    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Filename does not match expected pattern: {filename}")

    ts_str = parts[0]
    patch_id = parts[1]

    if not ts_str.isdigit():
        raise ValueError(f"Invalid timestamp in filename: {filename}")

    return int(ts_str), patch_id


def infer_date_token(input_folder: Path, fallback_timestamp_ms: int) -> str:
    """Infer the date token used in DATE_PATCHID."""

    name = input_folder.name
    m1 = DATE_RE_DDMMYYYY.search(name)
    if m1:
        return m1.group(1)

    m2 = DATE_RE_YYYYMMDD.search(name)
    if m2:
        return m2.group(1)

    dt = datetime.fromtimestamp(fallback_timestamp_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d")


def group_name_for_file(input_folder: Path, recording_timestamp_ms: int, patch_id: str) -> str:
    return f"{infer_date_token(input_folder, recording_timestamp_ms)}_{patch_id}"


def list_ecg_csv_files(input_folder: Path) -> List[Path]:
    out: List[Path] = []
    for p in sorted(input_folder.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".csv":
            continue
        if p.name.startswith("."):
            continue
        low = p.name.lower()
        if "vitals" in low:
            continue
        if "_ecg" not in low:
            continue
        out.append(p)
    return out


# ============================================================================
# OUTPUT PATHS
# ============================================================================


def build_output_paths(output_base: Path, group_name: str, recording_timestamp_ms: int) -> Dict[str, Path]:
    root = output_base / group_name
    daily_report = root / "daily_report"
    corrupted = root / "corrupted"

    ts_dir = root / str(recording_timestamp_ms)
    raw_dir = ts_dir / "raw"
    clean_dir = ts_dir / "clean"
    segments_dir = clean_dir / "segments"

    for d in [daily_report, corrupted, raw_dir, segments_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "root": root,
        "daily_report": daily_report,
        "corrupted": corrupted,
        "timestamp_dir": ts_dir,
        "raw": raw_dir,
        "clean": clean_dir,
        "segments": segments_dir,
    }


def safe_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


# ============================================================================
# SIGNAL UTILITIES
# ============================================================================


def compute_irregular_gap_fraction(delta_ms: np.ndarray) -> float:
    """Fraction of positive deltas that are much larger than the median."""

    if delta_ms.size == 0:
        return float("nan")

    positive = delta_ms[delta_ms > 0]
    if positive.size < 3:
        return 0.0

    med = float(np.median(positive))
    if not np.isfinite(med) or med <= 0:
        return float("nan")

    return float(np.mean(positive > 2.5 * med))


def resample_to_fixed_frequency(timestamps_ms: np.ndarray, signal: np.ndarray, target_fs_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample to a fixed sampling rate using linear interpolation."""

    ts = timestamps_ms.astype(np.int64)
    x = signal.astype(np.float64)
    if ts.size < 2:
        return ts, x

    t0 = float(ts[0])
    t_s = (ts.astype(np.float64) - t0) / 1000.0
    duration_s = float(t_s[-1])
    if not np.isfinite(duration_s) or duration_s <= 0:
        return ts, x

    dt = 1.0 / float(target_fs_hz)
    t_uniform = np.arange(0.0, duration_s + 1e-12, dt, dtype=np.float64)
    x_uniform = np.interp(t_uniform, t_s, x).astype(np.float64)
    ts_uniform = (t0 + np.round(t_uniform * 1000.0)).astype(np.int64)
    return ts_uniform, x_uniform


def zscore_normalize(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = x.astype(np.float64)
    mean = float(np.mean(x)) if x.size else 0.0
    std = float(np.std(x)) if x.size else 1.0
    if not np.isfinite(std) or std < 1e-12:
        std = 1.0
    return (x - mean) / std, mean, std


def compute_fft_magnitude(x: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float64)
    n = x.size
    if n == 0 or not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)

    nfft = 1 << int(np.ceil(np.log2(max(1, n))))
    yf = np.fft.rfft(x, n=nfft)
    xf = np.fft.rfftfreq(nfft, d=1.0 / float(sampling_rate_hz))
    mag = np.abs(yf)
    return xf.astype(np.float64), mag.astype(np.float64)


def save_fft_csv(path: Path, signal: np.ndarray, sampling_rate_hz: float) -> None:
    f, m = compute_fft_magnitude(signal, sampling_rate_hz)
    if f.size == 0:
        return
    pd.DataFrame({"frequency_hz": f, "magnitude": m}).to_csv(path, index=False)


# ============================================================================
# PLOTTING
# ============================================================================


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (14, 7),
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "lines.linewidth": 1.0,
        }
    )


def _decimate_for_plot(t: np.ndarray, y: np.ndarray, max_points: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Decimate signal for fast plotting by keeping min/max in chunks."""
    if len(y) <= max_points:
        return t, y
    chunk_size = len(y) // (max_points // 2)
    t_dec = []
    y_dec = []
    for i in range(0, len(y), chunk_size):
        chunk_t = t[i:i + chunk_size]
        chunk_y = y[i:i + chunk_size]
        min_idx = np.argmin(chunk_y)
        max_idx = np.argmax(chunk_y)
        if min_idx < max_idx:
            t_dec.extend([chunk_t[min_idx], chunk_t[max_idx]])
            y_dec.extend([chunk_y[min_idx], chunk_y[max_idx]])
        else:
            t_dec.extend([chunk_t[max_idx], chunk_t[min_idx]])
            y_dec.extend([chunk_y[max_idx], chunk_y[min_idx]])
    return np.array(t_dec), np.array(y_dec)

def save_time_series_plot(path: Path, timestamps_ms: np.ndarray, signal: np.ndarray, *, title: str, color: str) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    ts = timestamps_ms.astype(np.int64)
    x = signal.astype(np.float64)
    if ts.size == 0:
        return

    t_s = (ts - ts[0]).astype(np.float64) / 1000.0
    t_s_plot, x_plot = _decimate_for_plot(t_s, x)

    fig = plt.figure()
    plt.plot(t_s_plot, x_plot, color=color)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_frequency_spectrum_plot(path: Path, raw: np.ndarray, cleaned: np.ndarray, sampling_rate_hz: float, *, title: str) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    f_r, m_r = compute_fft_magnitude(raw, sampling_rate_hz)
    f_c, m_c = compute_fft_magnitude(cleaned, sampling_rate_hz)
    if f_r.size == 0 or f_c.size == 0:
        return

    fig = plt.figure()
    plt.semilogy(f_r, np.maximum(m_r, 1e-12), label="Raw", color="tab:gray", alpha=0.8)
    plt.semilogy(f_c, np.maximum(m_c, 1e-12), label="Cleaned", color="tab:blue")
    plt.xlim(0, min(60.0, 0.5 * float(sampling_rate_hz)))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("FFT magnitude")
    plt.grid(True, alpha=0.25, which="both")
    plt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_required_clean_plots(
    clean_dir: Path,
    *,
    timestamps_ms: np.ndarray,
    raw: np.ndarray,
    cleaned: np.ndarray,
    sampling_rate_hz: float,
    rpeaks_idx: np.ndarray,
) -> None:
    _configure_matplotlib()
    import matplotlib.pyplot as plt

    ts = timestamps_ms.astype(np.int64)
    if ts.size == 0:
        return

    t_s = (ts - ts[0]).astype(np.float64) / 1000.0

    save_time_series_plot(clean_dir / "raw_ecg.png", ts, raw, title="Raw ECG Signal (Resampled)", color="tab:gray")
    save_time_series_plot(clean_dir / "cleaned_ecg.png", ts, cleaned, title="Cleaned ECG Signal (Resampled)", color="tab:blue")

    t_s_raw_plot, raw_plot = _decimate_for_plot(t_s, raw)
    t_s_cln_plot, cln_plot = _decimate_for_plot(t_s, cleaned)

    fig = plt.figure()
    plt.plot(t_s_raw_plot, raw_plot, label="Raw", color="tab:gray", alpha=0.7)
    plt.plot(t_s_cln_plot, cln_plot, label="Cleaned", color="tab:blue")
    plt.title("Raw vs Cleaned ECG (Overlay)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(clean_dir / "raw_vs_cleaned.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    zoom_s = 10.0
    zoom_n = int(min(cleaned.size, max(1, round(zoom_s * float(sampling_rate_hz)))))
    fig = plt.figure()
    plt.plot(t_s[:zoom_n], cleaned[:zoom_n], color="tab:blue")
    plt.title("Zoomed ECG Window (First 10 seconds)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(clean_dir / "zoomed_ecg_window.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    save_frequency_spectrum_plot(
        clean_dir / "frequency_spectrum.png",
        raw,
        cleaned,
        sampling_rate_hz,
        title="Frequency Spectrum (Raw vs Cleaned)",
    )

    fig = plt.figure()
    plt.plot(t_s, cleaned, color="tab:blue", label="Cleaned ECG")
    if rpeaks_idx.size:
        rp = rpeaks_idx[(rpeaks_idx >= 0) & (rpeaks_idx < cleaned.size)]
        if rp.size:
            plt.scatter(t_s[rp], cleaned[rp], s=18, color="tab:red", label="R-peaks", zorder=3)
    plt.title("R-Peak Detection (Cleaned ECG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(clean_dir / "r_peaks_detection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# REPORTS / METADATA
# ============================================================================


def validate_ecg_dataframe(df_loaded: pd.DataFrame, df_sanitized: pd.DataFrame) -> Dict[str, object]:
    out: Dict[str, object] = {
        "rows_loaded": int(len(df_loaded)),
        "rows_sanitized": int(len(df_sanitized)),
    }

    if len(df_loaded) == 0:
        out["duplicate_timestamps_removed"] = 0
        out["non_increasing_removed"] = 0
        return out

    ts_loaded = df_loaded["timestamp_ms"].to_numpy(dtype=np.int64)
    ts_san = df_sanitized["timestamp_ms"].to_numpy(dtype=np.int64)

    dup_loaded = int(pd.Series(ts_loaded).duplicated(keep="first").sum())
    dup_san = int(pd.Series(ts_san).duplicated(keep="first").sum())
    out["duplicate_timestamps_removed"] = int(max(0, dup_loaded - dup_san))

    non_inc_loaded = int(np.sum(np.diff(ts_loaded) <= 0)) if ts_loaded.size >= 2 else 0
    non_inc_san = int(np.sum(np.diff(ts_san) <= 0)) if ts_san.size >= 2 else 0
    out["non_increasing_removed"] = int(max(0, non_inc_loaded - non_inc_san))

    return out


def write_raw_report(
    report_path: Path,
    *,
    source_file: str,
    sampling_est: SamplingInfo,
    stats_raw: SignalStats,
    validation: Dict[str, object],
    irregular_gap_fraction: float,
) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("RAW ECG PREPROCESSING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source file: {source_file}")
    lines.append("")

    lines.append("STRUCTURAL VALIDATION")
    lines.append("-" * 80)
    lines.append(f"Rows loaded:                  {validation.get('rows_loaded', 'n/a')}")
    lines.append(f"Rows after sanitization:      {validation.get('rows_sanitized', 'n/a')}")
    lines.append(f"Duplicate timestamps removed: {validation.get('duplicate_timestamps_removed', 'n/a')}")
    lines.append(f"Non-increasing removed:       {validation.get('non_increasing_removed', 'n/a')}")
    lines.append("")

    lines.append("TIMING")
    lines.append("-" * 80)
    lines.append(f"Estimated sampling rate (Hz): {sampling_est.sampling_rate_hz:.2f}")
    lines.append(f"Median interval (ms):         {sampling_est.avg_interval_ms:.3f}")
    lines.append(f"Irregular gap fraction:       {irregular_gap_fraction:.3f}")
    lines.append("")

    lines.append("RAW SIGNAL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Mean:  {stats_raw.mean:.6f}")
    lines.append(f"Std:   {stats_raw.std:.6f}")
    lines.append(f"Min:   {stats_raw.min:.6f}")
    lines.append(f"Max:   {stats_raw.max:.6f}")
    lines.append(f"Range: {stats_raw.range:.6f}")
    lines.append("")

    lines.append("Note: This report is for data QC only (not diagnosis).")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_clean_report(
    report_path: Path,
    *,
    steps: List[str],
    stats_raw: SignalStats,
    stats_clean: SignalStats,
    quality: Dict[str, object],
    heart_rate_bpm: float,
    rpeak_count: int,
    sampling_rate_target_hz: float,
) -> None:
    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("CLEAN ECG PREPROCESSING REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("PREPROCESSING STEPS")
    lines.append("-" * 80)
    for s in steps:
        lines.append(f"- {s}")
    lines.append("")
    # Present a concise set of high-value QC metrics (ratios, medians)
    lines.append("METRICS")
    lines.append("-" * 80)
    q_score = float(quality.get("score", float("nan")))
    q_label = str(quality.get("label", "n/a"))
    q_metrics = quality.get("metrics", {}) if isinstance(quality.get("metrics", {}), dict) else {}

    def _fmt_pct(x: float, decimals: int = 1) -> str:
        if not np.isfinite(x):
            return "n/a"
        return f"{100.0 * float(x):.{decimals}f}%"

    def _baseline_label(drift: float) -> str:
        if not np.isfinite(drift):
            return "n/a"
        if drift <= 0.18:
            return "stable"
        if drift <= 0.45:
            return "moderate"
        return "high"

    lines.append(f"Target sampling rate:        {sampling_rate_target_hz:.2f} Hz")
    lines.append(f"Quality score:               {q_score:.1f}/100 ({q_label})")
    lines.append(f"Usable segments:             {int(q_metrics.get('usable_segment_count', 0))}/{int(q_metrics.get('segment_count', 0))} ({_fmt_pct(float(q_metrics.get('usable_segment_ratio', float('nan'))))})")
    lines.append(f"Baseline stability:          {_baseline_label(float(q_metrics.get('baseline_drift_ratio_median', float('nan'))))} (drift ratio {_fmt_pct(float(q_metrics.get('baseline_drift_ratio_median', float('nan'))))})")
    lines.append(f"Noise (powerline):           {_fmt_pct(float(q_metrics.get('powerline_ratio_median', float('nan'))))}")
    lines.append(f"Noise (HF/motion/EMG):       {_fmt_pct(float(q_metrics.get('hf_ratio_median', float('nan'))))}")
    lines.append(f"Clipping/saturation ratio:   {_fmt_pct(float(q_metrics.get('clipping_ratio_median', float('nan'))), decimals=2)}")
    lines.append(f"Missing/interpolated ratio:  {_fmt_pct(float(q_metrics.get('missing_ratio_median', float('nan'))), decimals=2)}")
    lines.append("")

    # Recommendations (short, practical)
    recs = quality.get("recommendations", [])
    if isinstance(recs, list) and recs:
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    lines.append("R-PEAK DETECTION")
    lines.append("-" * 80)
    lines.append(f"R-peaks detected:            {int(rpeak_count)}")
    lines.append(f"Estimated heart rate:        {float(heart_rate_bpm):.2f} BPM")
    lines.append("")

    lines.append("SIGNAL STATISTICS")
    lines.append("-" * 80)
    lines.append(f"Raw:     mean={stats_raw.mean:.6f} std={stats_raw.std:.6f} min={stats_raw.min:.6f} max={stats_raw.max:.6f}")
    lines.append(f"Cleaned: mean={stats_clean.mean:.6f} std={stats_clean.std:.6f} min={stats_clean.min:.6f} max={stats_clean.max:.6f}")
    lines.append("")

    lines.append("Note: This report is for data QC only (not diagnosis).")
# Example usage in pipeline (replace old scoring logic):
# quality = score_ecg_quality(cleaned, raw, sampling_rate_hz, rpeaks, timestamps_ms, segment_seconds)
# score = quality["score"]
# label = quality["label"]
# penalties = quality["penalties"]
# recommendations = quality["recommendations"]
# metrics = quality["metrics"]

    report_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# REQUIRED DENOISING + RPEAKS
# ============================================================================


def denoise_ecg_signal(x: np.ndarray, sampling_rate_hz: float, powerline_hz: float) -> Tuple[np.ndarray, List[str]]:
    """Apply the required denoising chain and return (cleaned_signal, steps)."""

    steps: List[str] = []
    sig = x.astype(np.float64)

    sig = robust_clip(sig, k_mad=12.0)
    steps.append("Outlier control: robust clipping (median +/- 12*MAD)")

    win = max(5, int(round(0.2 * float(sampling_rate_hz))))
    sig = hampel_filter(sig, window_size=win, n_sigma=3.0)
    steps.append("Spike suppression: Hampel filter (~200 ms window)")

    # =========================================
    # ECG NOISE REMOVAL / DENOISING
    # =========================================
    filtered = filter_ecg(sig, sampling_rate_hz=sampling_rate_hz, powerline_hz=powerline_hz)
    steps.append("High-pass @ ~0.5 Hz (baseline wander removal)")
    steps.append(f"Notch @ {powerline_hz:.0f} Hz (powerline interference removal)")
    steps.append("Low-pass @ ~40 Hz (high-frequency/EMG noise removal)")

    try:
        cleaned = nk.ecg_clean(filtered, sampling_rate=float(sampling_rate_hz), method="neurokit")
        cleaned = np.asarray(cleaned, dtype=np.float64)
        steps.append("NeuroKit2: nk.ecg_clean(method='neurokit')")
    except Exception:
        cleaned = np.asarray(filtered, dtype=np.float64)
        steps.append("NeuroKit2 ecg_clean failed; used Butterworth+notch result")

    cleaned = cleaned - float(np.median(cleaned))
    return cleaned, steps


def detect_rpeaks_and_hr(cleaned: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, float]:
    """Robustly detect R-peaks and estimate heart rate using the core robust detector."""
    try:
        metrics = detect_rpeaks_robust(cleaned, sampling_rate_hz)
        return metrics.rpeaks, metrics.heart_rate_bpm
    except Exception:
        return np.asarray([], dtype=np.int64), float("nan")


# ============================================================================
# SEGMENTATION
# ============================================================================


def save_clean_csvs(clean_dir: Path, timestamps_ms: np.ndarray, cleaned: np.ndarray, normalized: np.ndarray) -> None:
    pd.DataFrame({"timestamp_ms": timestamps_ms.astype(np.int64), "clean_ecg": cleaned.astype(np.float64)}).to_csv(
        clean_dir / "cleaned_ecg.csv", index=False
    )
    pd.DataFrame({"timestamp_ms": timestamps_ms.astype(np.int64), "ecg_z": normalized.astype(np.float64)}).to_csv(
        clean_dir / "normalized_ecg.csv", index=False
    )


def segment_into_windows(
    segments_dir: Path,
    *,
    timestamps_ms: np.ndarray,
    cleaned: np.ndarray,
    normalized: np.ndarray,
    sampling_rate_hz: float,
    segment_seconds: float,
) -> int:
    """Split signal into non-overlapping windows and save segment_001.csv etc."""

    n_per = int(round(float(segment_seconds) * float(sampling_rate_hz)))
    if n_per <= 0 or cleaned.size < n_per:
        return 0

    count = 0
    for start in range(0, cleaned.size - n_per + 1, n_per):
        end = start + n_per
        count += 1
        seg_path = segments_dir / f"segment_{count:03d}.csv"
        seg = pd.DataFrame(
            {
                "timestamp_ms": timestamps_ms[start:end].astype(np.int64),
                "clean_ecg": cleaned[start:end].astype(np.float64),
                "ecg_z": normalized[start:end].astype(np.float64),
            }
        )
        seg.to_csv(seg_path, index=False)
    return count


# ============================================================================
# PER-FILE PIPELINE
# ============================================================================


def process_one_file(
    csv_path: Path,
    *,
    input_folder: Path,
    output_base: Path,
    cfg: PipelineConfig,
    logger: logging.Logger,
) -> FileResult:
    filename = csv_path.name

    try:
        recording_ts_ms, patch_id = parse_ecg_filename(filename)
    except Exception as exc:
        return FileResult(filename=filename, recording_timestamp_ms=None, patch_id=None, group_name="unparsed", success=False, error_message=str(exc))

    group_name = group_name_for_file(input_folder, recording_ts_ms, patch_id)
    paths = build_output_paths(output_base, group_name, recording_ts_ms)
    attach_group_file_logger(logger, paths["daily_report"] / "processing.log")

    result = FileResult(filename=filename, recording_timestamp_ms=recording_ts_ms, patch_id=patch_id, group_name=group_name, success=False)

    try:
        # -----------------------------
        # 1) ECG STRUCTURAL VALIDATION
        # -----------------------------
        df_loaded = load_ecg(str(csv_path))
        df_sanitized = sanitize_ecg(df_loaded)
        if df_sanitized.empty or len(df_sanitized) < 10:
            raise ValueError("ECG file contains too few usable samples after sanitization")

        validation = validate_ecg_dataframe(df_loaded, df_sanitized)

        timestamps_ms = df_sanitized["timestamp_ms"].to_numpy(dtype=np.int64)
        ecg_value = df_sanitized["ecg_value"].to_numpy(dtype=np.float64)

        sampling_est = estimate_sampling_rate(timestamps_ms)
        if not np.isfinite(sampling_est.sampling_rate_hz) or sampling_est.sampling_rate_hz <= 0:
            raise ValueError("Could not estimate a valid sampling rate from timestamps")

        result.sampling_rate_est_hz = float(sampling_est.sampling_rate_hz)
        result.irregular_gap_fraction = compute_irregular_gap_fraction(sampling_est.delta_t_ms)

        # =========================================
        # ECG RESAMPLING
        # =========================================
        ts_uniform, x_uniform = resample_to_uniform(timestamps_ms, ecg_value, sampling_est.sampling_rate_hz)
        ts_target, raw_target = resample_to_fixed_frequency(ts_uniform, x_uniform, cfg.target_fs_hz)

        if ts_target.size < int(cfg.target_fs_hz * 2):
            raise ValueError("Recording is too short after resampling (need at least ~2 seconds)")

        result.duration_s = float((ts_target[-1] - ts_target[0]) / 1000.0) if ts_target.size >= 2 else float("nan")

        # Save raw outputs
        safe_copy(csv_path, paths["raw"] / "original_ecg.csv")
        save_time_series_plot(paths["raw"] / "raw_ecg_plot.png", ts_target, raw_target, title="Raw ECG (Resampled to 125 Hz)", color="tab:gray")

        stats_raw = compute_stats(raw_target)
        write_raw_report(
            paths["raw"] / "raw_preprocessing_report.txt",
            source_file=filename,
            sampling_est=sampling_est,
            stats_raw=stats_raw,
            validation=validation,
            irregular_gap_fraction=result.irregular_gap_fraction,
        )

        # =========================================
        # ECG NOISE REMOVAL / DENOISING
        # =========================================
        cleaned, steps = denoise_ecg_signal(raw_target, cfg.target_fs_hz, cfg.powerline_hz)

        # =========================================
        # ECG NORMALIZATION
        # =========================================
        normalized, norm_mean, norm_std = zscore_normalize(cleaned)

        # =========================================
        # FOURIER / FREQUENCY DOMAIN ANALYSIS
        # =========================================
        save_fft_csv(paths["clean"] / "fft_raw.csv", raw_target, cfg.target_fs_hz)
        save_fft_csv(paths["clean"] / "fft_clean.csv", cleaned, cfg.target_fs_hz)

        # R-peak detection (required: nk.ecg_process)
        rpeaks_idx, hr_bpm = detect_rpeaks_and_hr(cleaned, cfg.target_fs_hz)
        result.heart_rate_bpm = hr_bpm
        result.rpeak_count = int(rpeaks_idx.size)

        # Noise heuristics + quality
        noise_info = detect_noise(ts_target, raw_target, cfg.target_fs_hz)
        result.flatline_count = safe_int(noise_info.get("flatline_count", 0))
        result.spike_count = safe_int(noise_info.get("spike_count", 0))
        result.jump_count = safe_int(noise_info.get("jump_count", 0))
        result.variance_clean = float(np.var(cleaned)) if cleaned.size else float("nan")

        # --- Use new modular scoring ---
        quality = score_ecg_quality(
            cleaned=cleaned,
            raw=raw_target,
            sampling_rate_hz=cfg.target_fs_hz,
            rpeaks=rpeaks_idx,
            segment_seconds=cfg.segment_seconds,
            powerline_hz=cfg.powerline_hz,
        )
        q_score = float(quality.get("score", float("nan")))
        q_label = str(quality.get("label", "n/a"))
        recommendations = quality.get("recommendations", []) if isinstance(quality.get("recommendations", []), list) else []
        result.quality_score_0_100 = q_score
        result.quality_label = q_label

        # Save clean CSVs
        save_clean_csvs(paths["clean"], ts_target, cleaned, normalized)

        # Required clean plots
        save_required_clean_plots(
            paths["clean"],
            timestamps_ms=ts_target,
            raw=raw_target,
            cleaned=cleaned,
            sampling_rate_hz=cfg.target_fs_hz,
            rpeaks_idx=rpeaks_idx,
        )

        # =========================================
        # ECG SEGMENTATION FOR ML
        # =========================================
        seg_count = segment_into_windows(
            paths["segments"],
            timestamps_ms=ts_target,
            cleaned=cleaned,
            normalized=normalized,
            sampling_rate_hz=cfg.target_fs_hz,
            segment_seconds=cfg.segment_seconds,
        )

        beats_dir = paths["clean"] / "beats"
        beat_count = segment_by_beats(
            beats_dir,
            timestamps_ms=ts_target,
            cleaned=cleaned,
            normalized=normalized,
            rpeaks=rpeaks_idx,
            sampling_rate_hz=cfg.target_fs_hz,
            pre_ms=200.0,
            post_ms=400.0,
        )
        similarity_scores = beat_similarity_scores(
            cleaned,
            rpeaks_idx,
            cfg.target_fs_hz,
            pre_ms=200.0,
            post_ms=400.0,
        )
        template, std_curve = compute_beat_template(
            cleaned,
            rpeaks_idx,
            cfg.target_fs_hz,
            pre_ms=200.0,
            post_ms=400.0,
        )
        if template.size:
            save_beat_template_plot(
                paths["clean"] / "beat_template.png",
                template,
                std_curve,
                cfg.target_fs_hz,
                pre_ms=200.0,
            )

        stats_clean = compute_stats(cleaned)
        baseline_raw_pp = baseline_drift_peak_to_peak(raw_target, cfg.target_fs_hz)
        baseline_clean_pp = baseline_drift_peak_to_peak(cleaned, cfg.target_fs_hz)
        # Write concise, ratio-based QC report (avoid unstable raw-count metrics)
        write_clean_report(
            paths["clean"] / "clean_preprocessing_report.txt",
            steps=steps,
            stats_raw=stats_raw,
            stats_clean=stats_clean,
            quality=quality,
            heart_rate_bpm=hr_bpm,
            rpeak_count=result.rpeak_count,
            sampling_rate_target_hz=cfg.target_fs_hz,
        )

        meta = {
            "source_file": filename,
            "recording_timestamp_ms": recording_ts_ms,
            "patch_id": patch_id,
            "group_name": group_name,
            "target_fs_hz": cfg.target_fs_hz,
            "powerline_hz": cfg.powerline_hz,
            "duration_s": result.duration_s,
            "sampling_rate_est_hz": result.sampling_rate_est_hz,
            "irregular_gap_fraction": result.irregular_gap_fraction,
            "normalization": {"method": "zscore", "mean": norm_mean, "std": norm_std},
            "quality": {"score_0_100": q_score, "label": q_label},
            "noise_counts": {
                "flatline_count": result.flatline_count,
                "spike_count": result.spike_count,
                "jump_count": result.jump_count,
            },
            "rpeaks": {"count": result.rpeak_count, "heart_rate_bpm": hr_bpm},
            "segments": {"segment_seconds": cfg.segment_seconds, "segment_count": seg_count},
            "beat_segments": {
                "pre_ms": 200.0,
                "post_ms": 400.0,
                "beat_count": beat_count,
                "mean_similarity": float(np.nanmean(similarity_scores))
                if similarity_scores.size
                else float("nan"),
            },
            "preprocessing_steps": steps,
        }
        (paths["clean"] / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        result.success = True
        logger.info(
            f"[OK] {filename} | group={group_name} | HR={hr_bpm:.1f} BPM | Quality={q_score:.1f}/100 ({q_label}) | segments={seg_count}"
        )

    except Exception as e:
        result.success = False
        result.error_message = str(e)

        try:
            safe_copy(csv_path, paths["corrupted"] / filename)
            (paths["corrupted"] / f"{Path(filename).stem}_error.txt").write_text(str(e), encoding="utf-8")
        except Exception:
            pass

        logger.error(f"[FAILED] {filename} | group={group_name} | {e}", exc_info=True)

    return result


# ============================================================================
# DAILY SUMMARY
# ============================================================================


def generate_daily_summary(group_name: str, results: List[FileResult], output_base: Path, logger: logging.Logger) -> None:
    root = output_base / group_name
    report_dir = root / "daily_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    attach_group_file_logger(logger, report_dir / "processing.log")

    results_sorted = sorted(
        results,
        key=lambda r: (r.recording_timestamp_ms if r.recording_timestamp_ms is not None else 0),
    )

    df = pd.DataFrame(
        [
            {
                "filename": r.filename,
                "recording_timestamp_ms": r.recording_timestamp_ms,
                "patch_id": r.patch_id,
                "success": r.success,
                "error_message": r.error_message,
                "duration_s": r.duration_s,
                "sampling_rate_est_hz": r.sampling_rate_est_hz,
                "irregular_gap_fraction": r.irregular_gap_fraction,
                "quality_score_0_100": r.quality_score_0_100,
                "quality_label": r.quality_label,
                "flatline_count": r.flatline_count,
                "spike_count": r.spike_count,
                "jump_count": r.jump_count,
                "variance_clean": r.variance_clean,
                "heart_rate_bpm": r.heart_rate_bpm,
                "rpeak_count": r.rpeak_count,
            }
            for r in results_sorted
        ]
    )
    df.to_csv(report_dir / "daily_ecg_summary.csv", index=False)

    ok = [r for r in results_sorted if r.success]
    failed = [r for r in results_sorted if not r.success]

    avg_fs = float(np.nanmean([r.sampling_rate_est_hz for r in ok])) if ok else float("nan")
    avg_q = float(np.nanmean([r.quality_score_0_100 for r in ok])) if ok else float("nan")
    avg_hr = float(np.nanmean([r.heart_rate_bpm for r in ok])) if ok else float("nan")

    noisy_count = int(sum(1 for r in ok if r.quality_label in {"MODERATE", "POOR"}))

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("DAILY ECG SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Group: {group_name}")
    lines.append("")
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Total files processed:   {len(results_sorted)}")
    lines.append(f"Successful:              {len(ok)}")
    lines.append(f"Corrupted/failed:        {len(failed)}")
    lines.append(f"Noisy (MODERATE/POOR):    {noisy_count}")
    lines.append("")
    lines.append("AGGREGATES")
    lines.append("-" * 80)
    lines.append(f"Average sampling rate (estimated): {avg_fs:.2f} Hz")
    lines.append(f"Average quality score:             {avg_q:.1f}/100")
    lines.append(f"Average heart rate estimate:       {avg_hr:.2f} BPM")
    lines.append("")

    if failed:
        lines.append("FAILURES")
        lines.append("-" * 80)
        for r in failed:
            lines.append(f"- {r.filename}: {r.error_message}")
        lines.append("")

    (report_dir / "daily_ecg_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    # Trend plots
    try:
        _configure_matplotlib()
        import matplotlib.pyplot as plt

        if len(ok) >= 2:
            xs = list(range(len(ok)))
            hrs = [r.heart_rate_bpm for r in ok]
            qs = [r.quality_score_0_100 for r in ok]

            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.plot(xs, hrs, marker="o", color="tab:red")
            ax.set_title("Heart Rate Trend")
            ax.set_xlabel("Recording index")
            ax.set_ylabel("BPM")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(report_dir / "heart_rate_trend.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(12, 4.8))
            ax.plot(xs, qs, marker="s", color="tab:green")
            ax.set_title("Quality Score Trend")
            ax.set_xlabel("Recording index")
            ax.set_ylabel("Quality (0-100)")
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            fig.savefig(report_dir / "quality_score_trend.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

    except Exception as e:
        logger.warning(f"Could not create daily trend plots for {group_name}: {e}")


# ============================================================================
# ORCHESTRATION
# ============================================================================


def _worker(kwargs: dict) -> FileResult:
    csv_path = kwargs["csv_path"]
    input_folder = kwargs["input_folder"]
    output_base = kwargs["output_base"]
    cfg = kwargs["cfg"]
    logger = build_console_logger()
    return process_one_file(
        csv_path, input_folder=input_folder, output_base=output_base, cfg=cfg, logger=logger
    )

def process_daily_folder(input_folder: Path, output_base: Path, cfg: PipelineConfig, logger: logging.Logger) -> int:
    files = list_ecg_csv_files(input_folder)
    if not files:
        logger.error("No ECG CSV files found (looking for '*_ecg*.csv' and excluding vitals).")
        return 1

    logger.info(f"Found {len(files)} ECG CSV files")

    by_group: Dict[str, List[FileResult]] = {}

    import concurrent.futures

    kwargs_list = [
        {
            "csv_path": csv_path,
            "input_folder": input_folder,
            "output_base": output_base,
            "cfg": cfg,
        }
        for csv_path in files
    ]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, res in enumerate(executor.map(_worker, kwargs_list), start=1):
            logger.info(f"[{i}/{len(files)}] Finished processing {res.filename}")
            by_group.setdefault(res.group_name, []).append(res)

    for group_name, group_results in by_group.items():
        if group_name == "unparsed":
            continue
        generate_daily_summary(group_name, group_results, output_base, logger)
        logger.info(f"Daily summary written for group: {group_name}")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Daily ECG preprocessing + dataset organization pipeline")
    parser.add_argument("--input", "-i", required=True, help="Folder containing ECG CSV files")
    parser.add_argument("--output", "-o", default="Processed_ECG", help="Output base directory")
    parser.add_argument("--target-fs", type=float, default=125.0, help="Target resampling frequency (Hz)")
    parser.add_argument("--powerline-hz", type=float, default=50.0, help="Powerline frequency for notch filter")
    parser.add_argument("--segment-seconds", type=float, default=5.0, help="Segment length in seconds")

    args = parser.parse_args(argv)

    input_folder = Path(args.input)
    output_base = Path(args.output)
    logger = build_console_logger()

    if not input_folder.exists() or not input_folder.is_dir():
        logger.error(f"Input folder not found or not a directory: {input_folder}")
        return 1

    output_base.mkdir(parents=True, exist_ok=True)

    cfg = PipelineConfig(
        target_fs_hz=float(args.target_fs),
        powerline_hz=float(args.powerline_hz),
        segment_seconds=float(args.segment_seconds),
    )

    logger.info("=" * 80)
    logger.info("DAILY ECG PREPROCESSING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Input:  {input_folder}")
    logger.info(f"Output: {output_base}")
    logger.info(f"Target sampling rate: {cfg.target_fs_hz:.2f} Hz")
    logger.info(f"Powerline notch:      {cfg.powerline_hz:.0f} Hz")
    logger.info(f"Segment length:       {cfg.segment_seconds:.1f} s")

    return process_daily_folder(input_folder, output_base, cfg, logger)


if __name__ == "__main__":
    raise SystemExit(main())