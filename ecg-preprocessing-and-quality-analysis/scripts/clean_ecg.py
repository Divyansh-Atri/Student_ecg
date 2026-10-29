"""clean_ecg.py

Advanced ECG preprocessing + visualization pipeline for AI/ML workflows.

This script is designed to prepare noisy ECG sensor data for downstream ML:
- robust CSV loading (malformed rows tolerated)
- timestamp sanitization and alignment (dedupe, sort, optional resample)
- baseline drift stabilization (high-pass ~0.5 Hz)
- powerline suppression (50 Hz notch)
- high-frequency noise suppression (low-pass ~40 Hz)
- spike/outlier suppression (Hampel-style robust filter)
- reliable R-peak detection (NeuroKit2 + fallbacks + physiological checks)
- professional scientific plots (6 required plots)
- detailed quality metrics report

Supported input formats:
1) Standard 2-column format (with or without header):
     timestamp_ms,value
2) Some sensors export a “wide pair” format (timestamp,value,timestamp,value,...)
     This script auto-detects and parses that as a fallback.

Outputs (saved under an output directory):
- cleaned_ecg.csv
- ecg_report.txt
- PNG plots (publication-style)

Important:
This is a preprocessing utility; it does NOT provide medical diagnosis.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SamplingInfo:
    """Holds derived sampling information from timestamps."""

    avg_interval_ms: float
    sampling_rate_hz: float
    delta_t_ms: np.ndarray


@dataclass
class SignalStats:
    """Basic descriptive statistics used for preprocessing validation."""

    mean: float
    std: float
    min: float
    max: float
    range: float


@dataclass
class RRStats:
    """Summary statistics for RR intervals (seconds)."""

    count: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float


@dataclass
class PeakMetrics:
    """R-peak detection results."""

    rpeaks: np.ndarray  # indices into the (resampled) signal
    heart_rate_bpm: float
    rr_intervals_s: np.ndarray
    rr_stats: RRStats
    rr_stats_valid: RRStats
    rr_invalid_fraction: float


def load_ecg(csv_path: str) -> pd.DataFrame:
    """Load ECG data using pandas and validate required columns.

    Primary supported format:
      timestamp_ms, ecg_value

    Fallback (helpful in real-world pipelines):
    If the required columns are missing, attempt to detect the "wide pair" format:
      1779086243678,20,1779086243686,27,...

    Returns:
        DataFrame with columns: timestamp_ms (int64), ecg_value (float64)

    Raises:
        ValueError if the file cannot be interpreted as ECG time series.
    """

    # Quick first-line sniff to avoid mis-parsing "wide pair" files as 2-column.
    # If the first non-empty line has >2 comma-separated fields (and an even count),
    # it is almost certainly the sensor's (timestamp,value,timestamp,value,...) format.
    first_line_fields: Optional[List[str]] = None
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip("\n\r")
                if line.strip():
                    first_line_fields = [p.strip() for p in line.split(",")]
                    break
    except Exception:
        first_line_fields = None

    is_wide_pair = False
    if first_line_fields is not None and len(first_line_fields) > 2:
        if len(first_line_fields) % 2 == 0:
            is_wide_pair = True

    # Try: header-based structured CSV.
    try:
        sniff = pd.read_csv(csv_path, nrows=50)
    except Exception:
        sniff = None

    # Common column name variants from sensors / exports.
    candidate_ts = ["timestamp_ms", "timestamp", "time_ms", "t_ms"]
    candidate_val = ["ecg_value", "value", "ecg", "raw_ecg", "amplitude"]

    if sniff is not None and sniff.shape[1] >= 2:
        ts_col = next((c for c in candidate_ts if c in sniff.columns), None)
        val_col = next((c for c in candidate_val if c in sniff.columns), None)
        if ts_col and val_col:
            df = pd.read_csv(csv_path, on_bad_lines="skip")
            out = df[[ts_col, val_col]].rename(columns={ts_col: "timestamp_ms", val_col: "ecg_value"}).copy()
            out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
            out["ecg_value"] = pd.to_numeric(out["ecg_value"], errors="coerce")
            out = out.dropna(subset=["timestamp_ms", "ecg_value"]).copy()
            out["timestamp_ms"] = out["timestamp_ms"].astype(np.int64)
            out["ecg_value"] = out["ecg_value"].astype(np.float64)
            return out

    # Try: no-header two-column CSV (timestamp_ms,value).
    # IMPORTANT: Skip this fast path when the file looks like wide-pair,
    # otherwise we only read the first two columns per row.
    if not is_wide_pair:
        try:
            df2 = pd.read_csv(
                csv_path,
                header=None,
                names=["timestamp_ms", "ecg_value"],
                usecols=[0, 1],
                on_bad_lines="skip",
            )
            df2["timestamp_ms"] = pd.to_numeric(df2["timestamp_ms"], errors="coerce")
            df2["ecg_value"] = pd.to_numeric(df2["ecg_value"], errors="coerce")
            df2 = df2.dropna(subset=["timestamp_ms", "ecg_value"]).copy()
            if len(df2) >= 10:
                df2["timestamp_ms"] = df2["timestamp_ms"].astype(np.int64)
                df2["ecg_value"] = df2["ecg_value"].astype(np.float64)
                return df2
        except Exception:
            pass

    # Fallback: treat the entire file as "rows of pair lists" with no header.
    # This allows you to reuse the script directly on the raw format from some sensors.
    # We keep this logic simple and strict to avoid silently mis-parsing unrelated CSVs.
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        timestamps: List[int] = []
        values: List[float] = []

        for row_index, line in enumerate(f, start=1):
            line = line.strip("\n\r")
            if not line.strip():
                continue

            fields = line.split(",")
            if len(fields) % 2 != 0:
                raise ValueError(
                    f"Row {row_index} has odd field count ({len(fields)}). "
                    "This is not valid (timestamp_ms, ecg_value) pair formatting."
                )

            # Parse pairs.
            row_ts: List[int] = []
            row_v: List[float] = []
            for i in range(0, len(fields), 2):
                ts = pd.to_numeric(fields[i].strip(), errors="coerce")
                v = pd.to_numeric(fields[i + 1].strip(), errors="coerce")
                if pd.isna(ts) or pd.isna(v):
                    raise ValueError(
                        f"Non-numeric value found in raw pair-format at row {row_index}."
                    )
                row_ts.append(int(ts))
                row_v.append(float(v))

            timestamps.extend(row_ts)
            values.extend(row_v)

    if not timestamps:
        raise ValueError(
            "Could not find required columns or parse the file as pair-formatted ECG."
        )

    return pd.DataFrame(
        {
            "timestamp_ms": np.array(timestamps, dtype=np.int64),
            "ecg_value": np.array(values, dtype=np.float64),
        }
    )


def sanitize_ecg(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize ECG rows: drop NaNs, dedupe timestamps, sort, and remove non-increasing time.

    This directly addresses common sensor CSV issues:
    - malformed lines -> coerced to NaN and dropped
    - duplicate timestamps -> keep first
    - out-of-order timestamps -> sorted
    - non-increasing steps -> removed (filters assume forward time)
    """

    out = df.copy()
    out["timestamp_ms"] = pd.to_numeric(out["timestamp_ms"], errors="coerce")
    out["ecg_value"] = pd.to_numeric(out["ecg_value"], errors="coerce")
    out = out.dropna(subset=["timestamp_ms", "ecg_value"]).copy()

    # Convert types after dropping NaNs.
    out["timestamp_ms"] = out["timestamp_ms"].astype(np.int64)
    out["ecg_value"] = out["ecg_value"].astype(np.float64)

    # Remove duplicate timestamps (keep first occurrence).
    out = out.drop_duplicates(subset=["timestamp_ms"], keep="first").copy()

    # Sort by time.
    out = out.sort_values("timestamp_ms", kind="mergesort").reset_index(drop=True)

    # Remove any remaining non-increasing timestamps.
    ts = out["timestamp_ms"].to_numpy(dtype=np.int64)
    if ts.size >= 2:
        keep = np.ones(ts.size, dtype=bool)
        keep[1:] = np.diff(ts) > 0
        out = out.loc[keep].reset_index(drop=True)

    return out


def estimate_sampling_rate(timestamps_ms: np.ndarray) -> SamplingInfo:
    """Estimate sampling rate from timestamps.

    Requirement 3:
      delta_t = timestamp[i+1] - timestamp[i]
      sampling_rate = 1000 / average_interval_ms

    Notes:
    - We compute delta_t in milliseconds.
    - We only use positive deltas to compute the average interval, because
      duplicates/out-of-order timestamps are not physically meaningful.
    """

    if timestamps_ms.size < 2:
        return SamplingInfo(avg_interval_ms=float("nan"), sampling_rate_hz=float("nan"), delta_t_ms=np.asarray([], dtype=np.int64))

    delta_t_ms = np.diff(timestamps_ms.astype(np.int64))
    positive = delta_t_ms[delta_t_ms > 0]

    if positive.size == 0:
        return SamplingInfo(avg_interval_ms=float("nan"), sampling_rate_hz=float("nan"), delta_t_ms=delta_t_ms)

    # Use median for robustness against timestamp jitter and rare gaps.
    avg_interval_ms = float(np.median(positive))
    sampling_rate_hz = float("nan")
    if avg_interval_ms > 0:
        sampling_rate_hz = 1000.0 / avg_interval_ms

    return SamplingInfo(avg_interval_ms=avg_interval_ms, sampling_rate_hz=sampling_rate_hz, delta_t_ms=delta_t_ms)


def compute_stats(signal: np.ndarray) -> SignalStats:
    """Compute basic stats required for validation reports."""

    if signal.size == 0:
        return SignalStats(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"), range=float("nan"))

    sig = signal.astype(np.float64)
    min_v = float(np.min(sig))
    max_v = float(np.max(sig))
    return SignalStats(
        mean=float(np.mean(sig)),
        std=float(np.std(sig)),
        min=min_v,
        max=max_v,
        range=float(max_v - min_v),
    )


def _robust_scale(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return robustly standardized signal and (center, scale)."""

    x = x.astype(np.float64)
    center = float(np.median(x))
    mad = float(np.median(np.abs(x - center)))
    scale = mad * 1.4826  # ~std for normal
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(x)) if float(np.std(x)) > 1e-12 else 1.0
    return (x - center) / scale, center, scale


def resample_to_uniform(
    timestamps_ms: np.ndarray,
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an irregularly-timestamped ECG onto a uniform grid.

    Why this matters:
    - Digital filters and most ECG peak detectors assume constant sampling.
    - Real sensors often have timestamp jitter; resampling improves stability.
    """

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Cannot resample without a valid sampling rate.")

    ts = timestamps_ms.astype(np.int64)
    x = signal.astype(np.float64)
    t_s = (ts - ts[0]).astype(np.float64) / 1000.0

    if t_s.size < 2:
        return ts, x

    dt = 1.0 / fs
    t_uniform = np.arange(0.0, float(t_s[-1]), dt, dtype=np.float64)
    if t_uniform.size < 2:
        return ts, x

    x_uniform = np.interp(t_uniform, t_s, x).astype(np.float64)
    ts_uniform = (ts[0] + np.round(t_uniform * 1000.0)).astype(np.int64)
    return ts_uniform, x_uniform


def hampel_filter(x: np.ndarray, window_size: int, n_sigma: float = 3.0) -> np.ndarray:
    """Suppress impulsive spikes without over-smoothing QRS morphology.

    A Hampel filter replaces outliers in a sliding window with the local median.
    It's effective for motion spikes / ADC glitches while preserving sharp QRS.
    Optimized using pandas rolling windows for significant speedups.
    """

    x = x.astype(np.float64)
    n = x.size
    if n == 0 or window_size < 3:
        return x
    if window_size % 2 == 0:
        window_size += 1

    y = x.copy()
    s = pd.Series(x)
    
    # Vectorized rolling median and MAD
    rolling_med = s.rolling(window=window_size, center=True, min_periods=1).median()
    rolling_mad = (s - rolling_med).abs().rolling(window=window_size, center=True, min_periods=1).median()
    
    sigma = rolling_mad * 1.4826
    
    # Identify and replace outliers
    outliers = (sigma >= 1e-12) & (np.abs(s - rolling_med) > n_sigma * sigma)
    y[outliers] = rolling_med[outliers]
    
    return y


def robust_clip(x: np.ndarray, k_mad: float = 12.0) -> np.ndarray:
    """Clip extreme outliers using median +/- k*MAD.

    This is a gentle way to prevent rare extreme motion spikes from dominating
    downstream filtering and peak detection.
    """

    x = x.astype(np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    sigma = mad * 1.4826
    if not np.isfinite(sigma) or sigma < 1e-12:
        return x
    lo = med - float(k_mad) * sigma
    hi = med + float(k_mad) * sigma
    return np.clip(x, lo, hi)


def filter_ecg(
    x: np.ndarray,
    sampling_rate_hz: float,
    powerline_hz: float = 50.0,
) -> np.ndarray:
    """Professional ECG filtering chain (zero-phase): HP(0.5 Hz) -> notch -> LP(40 Hz)."""

    try:
        from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt
    except Exception as e:
        raise RuntimeError("Missing dependency scipy. Install with: pip install scipy") from e

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate.")

    sig = x.astype(np.float64)

    # High-pass: remove baseline wander (respiration, electrode drift).
    hp_hz = 0.5
    hp_hz = min(hp_hz, 0.45 * fs)
    hp_hz = max(hp_hz, 0.05)
    sos_hp = butter(N=4, Wn=hp_hz / (fs / 2.0), btype="high", output="sos")
    sig = sosfiltfilt(sos_hp, sig)

    # Notch: suppress powerline interference (50 Hz), if Nyquist allows.
    if fs > 2.2 * float(powerline_hz):
        q = 30.0
        b_n, a_n = iirnotch(w0=float(powerline_hz) / (fs / 2.0), Q=q)
        sig = filtfilt(b_n, a_n, sig)

    # Low-pass: remove EMG / high-frequency noise while preserving QRS.
    lp_hz = 40.0
    lp_hz = min(lp_hz, 0.45 * fs)
    sos_lp = butter(N=4, Wn=lp_hz / (fs / 2.0), btype="low", output="sos")
    sig = sosfiltfilt(sos_lp, sig)

    return np.asarray(sig, dtype=np.float64)


def clean_signal_advanced(
    timestamps_ms: np.ndarray,
    ecg_signal: np.ndarray,
    sampling_rate_hz: float,
    *,
    resample: bool,
    powerline_hz: float,
    neurokit_clean: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Full cleaning pipeline returning (ts_ms, raw_aligned, cleaned, steps)."""

    steps: List[str] = []
    ts = timestamps_ms.astype(np.int64)
    raw = ecg_signal.astype(np.float64)

    # Optional resampling for stable filtering/peaks.
    if resample:
        ts, raw = resample_to_uniform(ts, raw, sampling_rate_hz)
        steps.append("Timestamp alignment: resampled to uniform grid (improves filtering + peak detection)")
    else:
        steps.append("Timestamp alignment: kept original timestamps (no resampling)")

    # Robust clipping + spike suppression (motion/ADC glitches).
    fs = float(sampling_rate_hz)
    spike_win = max(5, int(round(0.2 * fs)))  # ~200 ms
    raw_clipped = robust_clip(raw, k_mad=12.0)
    steps.append("Outlier control: robust clipping (median +/- 12*MAD) to limit extreme motion spikes")
    raw_despiked = hampel_filter(raw_clipped, window_size=spike_win, n_sigma=3.0)
    steps.append("Motion/spike suppression: Hampel robust outlier filter (~200 ms window, edge-aware)")

    cleaned = filter_ecg(raw_despiked, sampling_rate_hz=fs, powerline_hz=powerline_hz)
    steps.append("Baseline wander removal: 4th-order high-pass Butterworth @ ~0.5 Hz (zero-phase)")
    steps.append(f"Powerline suppression: notch filter @ {powerline_hz:.0f} Hz (Q=30, zero-phase)")
    steps.append("High-frequency noise removal: 4th-order low-pass Butterworth @ ~40 Hz (zero-phase)")

    # Recentering helps downstream peak detection and makes plots easier to interpret.
    cleaned = cleaned - float(np.median(cleaned))

    if neurokit_clean:
        try:
            import neurokit2 as nk

            # NeuroKit2 can further denoise and stabilize the signal.
            cleaned_nk = nk.ecg_clean(cleaned, sampling_rate=fs, method="neurokit")
            cleaned = np.asarray(cleaned_nk, dtype=np.float64)
            steps.append("Optional refinement: NeuroKit2 ecg_clean(method='neurokit') on the filtered signal")
            cleaned = cleaned - float(np.median(cleaned))
        except Exception:
            steps.append("Optional refinement: NeuroKit2 unavailable; skipped")

    return ts, raw, cleaned, steps


def apply_filters(ecg_signal: np.ndarray, sampling_rate_hz: float) -> Dict[str, np.ndarray]:
    """Provide additional filtering examples using scipy.signal.

    Requirement 6:
    - low-pass filter
    - high-pass filter
    - notch filter for 50 Hz interference

    We return these as separate signals so you can compare approaches.
    """

    try:
        from scipy.signal import butter, filtfilt, iirnotch
    except Exception as e:
        raise RuntimeError(
            "Missing dependency scipy. Install with: pip install scipy"
        ) from e

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        fs = 250.0

    x = ecg_signal.astype(np.float64)

    # Low-pass example: keep QRS and remove high-frequency noise.
    # Typical ECG low-pass cutoffs range ~30-45 Hz depending on use.
    low_cut_hz = min(40.0, 0.45 * fs)
    b_lp, a_lp = butter(N=4, Wn=low_cut_hz / (fs / 2.0), btype="low")
    lowpass = filtfilt(b_lp, a_lp, x)

    # High-pass example: remove baseline wander.
    # Typical baseline-wander removal uses ~0.5 Hz cutoff.
    high_cut_hz = min(0.5, 0.45 * fs)
    # If fs is very low, high_cut_hz could be too close to Nyquist; guard it.
    high_cut_hz = max(high_cut_hz, 0.05)
    b_hp, a_hp = butter(N=4, Wn=high_cut_hz / (fs / 2.0), btype="high")
    highpass = filtfilt(b_hp, a_hp, x)

    # 50 Hz notch filter (mains interference), only if Nyquist allows.
    notch = x.copy()
    notch_freq_hz = 50.0
    if fs > 2 * notch_freq_hz:
        q = 30.0  # quality factor; higher = narrower notch
        b_n, a_n = iirnotch(w0=notch_freq_hz / (fs / 2.0), Q=q)
        notch = filtfilt(b_n, a_n, x)

    return {
        "lowpass_40hz": np.asarray(lowpass, dtype=np.float64),
        "highpass_0p5hz": np.asarray(highpass, dtype=np.float64),
        "notch_50hz": np.asarray(notch, dtype=np.float64),
    }


def detect_rpeaks_robust(cleaned: np.ndarray, sampling_rate_hz: float) -> PeakMetrics:
    """Robust R-peak detection with physiological constraints.

    Strategy:
    - Primary: NeuroKit2 peak detection on a robustly scaled signal
    - Secondary (if available): Pan-Tompkins/Hamilton variants via NeuroKit2
    - Fallback: SciPy find_peaks with adaptive prominence
    - Post-filter: enforce minimum peak distance and keep strongest peak
    """

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Invalid sampling rate for R-peak detection.")

    x = np.asarray(cleaned, dtype=np.float64)
    xz, _, _ = _robust_scale(x)

    min_rr_s = 0.30  # 200 bpm
    max_rr_s = 2.00  # 30 bpm
    min_distance = max(1, int(round(min_rr_s * fs)))

    def _pan_tompkins_like(x_in: np.ndarray) -> np.ndarray:
        """Pan–Tompkins-style QRS detection (robust to amplitude scaling and polarity).

        Steps:
        - Bandpass around QRS energy (5–15 Hz)
        - Derivative + squaring
        - Moving-window integration (~150 ms)
        - Peak finding on integrated signal
        - Refine each detection to the strongest local extremum in the cleaned ECG
        """

        try:
            from scipy.signal import butter, sosfiltfilt, find_peaks
        except Exception as e:
            raise RuntimeError("Missing dependency scipy. Install with: pip install scipy") from e

        x_in = np.asarray(x_in, dtype=np.float64)
        if x_in.size < int(2 * fs):
            return np.asarray([], dtype=np.int64)

        # QRS band (typical): ~5-15 Hz
        lo, hi = 5.0, 15.0
        hi = min(hi, 0.45 * fs)
        if hi <= lo:
            return np.asarray([], dtype=np.int64)

        sos = butter(N=3, Wn=[lo / (fs / 2.0), hi / (fs / 2.0)], btype="bandpass", output="sos")
        y = sosfiltfilt(sos, x_in)

        dy = np.diff(y, prepend=y[0]) * fs
        sq = dy * dy

        win = max(1, int(round(0.150 * fs)))
        kernel = np.ones(win, dtype=np.float64) / float(win)
        mwi = np.convolve(sq, kernel, mode="same")

        med = float(np.median(mwi))
        mad = float(np.median(np.abs(mwi - med)))
        sigma = mad * 1.4826
        height = med + 1.5 * sigma
        prom = max(1e-12, 1.0 * sigma)

        cand, _ = find_peaks(mwi, distance=min_distance, height=height, prominence=prom)
        if cand.size == 0:
            return np.asarray([], dtype=np.int64)

        # Refine candidates to local strongest extremum (polarity-agnostic).
        refine_w = int(round(0.06 * fs))  # +/-60ms
        peaks: List[int] = []
        for c in cand:
            s = max(0, int(c) - refine_w)
            e = min(x_in.size, int(c) + refine_w + 1)
            seg = x_in[s:e]
            if seg.size == 0:
                continue
            p = int(s + int(np.argmax(np.abs(seg))))
            peaks.append(p)

        if not peaks:
            return np.asarray([], dtype=np.int64)
        return np.asarray(sorted(set(peaks)), dtype=np.int64)

    def _postfilter_distance(peaks: np.ndarray) -> np.ndarray:
        if peaks.size <= 1:
            return peaks
        # Enforce min distance by keeping higher-amplitude peak when too close.
        kept: List[int] = [int(peaks[0])]
        for p in peaks[1:]:
            p = int(p)
            if p - kept[-1] >= min_distance:
                kept.append(p)
            else:
                # Too close: keep the one with higher absolute amplitude.
                if abs(x[p]) > abs(x[kept[-1]]):
                    kept[-1] = p
        return np.asarray(kept, dtype=np.int64)

    peaks_primary: Optional[np.ndarray] = None
    peaks_secondary: Optional[np.ndarray] = None
    peaks_pt: Optional[np.ndarray] = None

    # --- NeuroKit2 primary (polarity-aware) ---
    try:
        import neurokit2 as nk

        _, info_pos = nk.ecg_peaks(xz, sampling_rate=fs, method="neurokit")
        peaks_pos = np.asarray(info_pos.get("ECG_R_Peaks", []), dtype=np.int64)

        _, info_neg = nk.ecg_peaks(-xz, sampling_rate=fs, method="neurokit")
        peaks_neg = np.asarray(info_neg.get("ECG_R_Peaks", []), dtype=np.int64)

        # Pick the polarity that yields more candidates.
        peaks_primary = peaks_pos if peaks_pos.size >= peaks_neg.size else peaks_neg
    except Exception:
        peaks_primary = None

    # --- NeuroKit2 secondary (more conservative in some cases) ---
    if peaks_primary is not None and peaks_primary.size >= 5:
        try:
            import neurokit2 as nk

            _, info2 = nk.ecg_peaks(xz, sampling_rate=fs, method="pantompkins1985")
            peaks_secondary = np.asarray(info2.get("ECG_R_Peaks", []), dtype=np.int64)
        except Exception:
            peaks_secondary = None

    # --- Pan–Tompkins-like (SciPy) ---
    try:
        peaks_pt = _pan_tompkins_like(x)
    except Exception:
        peaks_pt = None

    # --- SciPy fallback ---
    def _scipy_peaks() -> np.ndarray:
        try:
            from scipy.signal import find_peaks
        except Exception as e:
            raise RuntimeError("Missing dependency scipy. Install with: pip install scipy") from e

        # Adaptive prominence based on robust scale.
        med = float(np.median(xz))
        mad = float(np.median(np.abs(xz - med)))
        prom = max(0.8, 3.0 * mad)
        peaks, _ = find_peaks(xz, distance=min_distance, prominence=prom)
        return np.asarray(peaks, dtype=np.int64)

    def _score(peaks_in: np.ndarray) -> Tuple[float, int, float]:
        """Lower is better. Returns (score, valid_rr_count, invalid_fraction)."""

        peaks_in = np.asarray(peaks_in, dtype=np.int64)
        peaks_in = np.unique(peaks_in)
        peaks_in = np.sort(peaks_in)
        peaks_in = _postfilter_distance(peaks_in)
        if peaks_in.size < 3:
            return (float("inf"), 0, 1.0)
        rr = np.diff(peaks_in).astype(np.float64) / fs
        rr_valid = rr[(rr >= min_rr_s) & (rr <= max_rr_s)]
        invalid_fraction = float(1.0 - (rr_valid.size / rr.size)) if rr.size else 1.0
        hr = float(60.0 / float(np.mean(rr_valid))) if rr_valid.size else float("nan")
        penalty = 0.0
        if np.isfinite(hr) and (hr < 40.0 or hr > 180.0):
            penalty += 0.5
        # Prefer more valid beats and fewer invalid intervals.
        score = invalid_fraction + penalty + 1.0 / float(rr_valid.size + 1)
        return (score, int(rr_valid.size), invalid_fraction)

    # Choose the best candidate set.
    candidates: List[np.ndarray] = []
    if peaks_primary is not None and peaks_primary.size:
        candidates.append(peaks_primary)
    if peaks_secondary is not None and peaks_secondary.size:
        candidates.append(peaks_secondary)
    if peaks_pt is not None and peaks_pt.size:
        candidates.append(peaks_pt)

    if not candidates:
        peaks = _scipy_peaks()
    else:
        scored = [(idx, *_score(p)) for idx, p in enumerate(candidates)]
        # Pick lowest score, break ties by higher valid_rr_count.
        scored.sort(key=lambda t: (t[1], -t[2]))
        peaks = candidates[scored[0][0]]

    peaks = np.unique(peaks)
    peaks = peaks[(peaks >= 0) & (peaks < x.size)]
    peaks = np.sort(peaks)
    peaks = _postfilter_distance(peaks)

    # RR intervals + HR.
    if peaks.size >= 2:
        rr = np.diff(peaks).astype(np.float64) / fs
        # Remove physiologically impossible intervals for summary stats.
        rr_valid = rr[(rr >= min_rr_s) & (rr <= max_rr_s)]
        rr_for_stats = rr_valid if rr_valid.size >= 1 else rr
        hr_bpm = float(60.0 / float(np.mean(rr_for_stats))) if rr_for_stats.size else float("nan")
    else:
        rr = np.asarray([], dtype=np.float64)
        hr_bpm = float("nan")

    rr_valid = rr[(rr >= min_rr_s) & (rr <= max_rr_s)] if rr.size else np.asarray([], dtype=np.float64)
    invalid_fraction = float(1.0 - (rr_valid.size / rr.size)) if rr.size else float("nan")

    rr_stats = RRStats(
        count=int(rr.size),
        mean_s=float(np.mean(rr)) if rr.size else float("nan"),
        std_s=float(np.std(rr)) if rr.size else float("nan"),
        min_s=float(np.min(rr)) if rr.size else float("nan"),
        max_s=float(np.max(rr)) if rr.size else float("nan"),
    )

    rr_stats_valid = RRStats(
        count=int(rr_valid.size),
        mean_s=float(np.mean(rr_valid)) if rr_valid.size else float("nan"),
        std_s=float(np.std(rr_valid)) if rr_valid.size else float("nan"),
        min_s=float(np.min(rr_valid)) if rr_valid.size else float("nan"),
        max_s=float(np.max(rr_valid)) if rr_valid.size else float("nan"),
    )

    return PeakMetrics(
        rpeaks=peaks,
        heart_rate_bpm=hr_bpm,
        rr_intervals_s=rr,
        rr_stats=rr_stats,
        rr_stats_valid=rr_stats_valid,
        rr_invalid_fraction=invalid_fraction,
    )


def estimate_snr_bandpower(raw: np.ndarray, cleaned: np.ndarray, sampling_rate_hz: float) -> Tuple[float, float, float]:
    """Frequency-domain SNR estimate: in-band (0.5-40 Hz) vs out-of-band power."""

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        return float("nan"), float("nan"), float("nan")

    f_r, p_r = welch_psd(raw, fs)
    f_c, p_c = welch_psd(cleaned, fs)

    def _band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
        mask = (f >= lo) & (f <= hi)
        if not np.any(mask):
            return float("nan")
        return float(np.trapezoid(p[mask], f[mask]))

    nyq = 0.5 * fs
    in_lo, in_hi = 0.5, min(40.0, 0.95 * nyq)
    out_hi = min(nyq, 0.95 * nyq)

    in_r = _band_power(f_r, p_r, in_lo, in_hi)
    out_r = _band_power(f_r, p_r, in_hi, out_hi)
    in_c = _band_power(f_c, p_c, in_lo, in_hi)
    out_c = _band_power(f_c, p_c, in_hi, out_hi)

    out_r = out_r if np.isfinite(out_r) and out_r > 1e-18 else 1e-18
    out_c = out_c if np.isfinite(out_c) and out_c > 1e-18 else 1e-18

    snr_r = 10.0 * float(np.log10(in_r / out_r)) if np.isfinite(in_r) and in_r > 1e-18 else float("nan")
    snr_c = 10.0 * float(np.log10(in_c / out_c)) if np.isfinite(in_c) and in_c > 1e-18 else float("nan")
    return snr_r, snr_c, snr_c - snr_r


def estimate_snr_improvement(raw: np.ndarray, cleaned: np.ndarray) -> Tuple[float, float, float]:
    """Estimate SNR improvement (dB) using noise ≈ raw - cleaned."""

    raw = raw.astype(np.float64)
    cleaned = cleaned.astype(np.float64)
    n = min(raw.size, cleaned.size)
    if n < 10:
        return float("nan"), float("nan"), float("nan")

    raw = raw[:n]
    cleaned = cleaned[:n]
    noise = raw - cleaned

    sp_raw = float(np.var(raw))
    sp_clean = float(np.var(cleaned))
    npow = float(np.var(noise))
    npow = npow if npow > 1e-12 else 1e-12

    snr_raw = 10.0 * float(np.log10(sp_raw / npow)) if sp_raw > 1e-12 else float("nan")
    snr_clean = 10.0 * float(np.log10(sp_clean / npow)) if sp_clean > 1e-12 else float("nan")
    return snr_raw, snr_clean, snr_clean - snr_raw


def baseline_drift_peak_to_peak(x: np.ndarray, sampling_rate_hz: float) -> float:
    """Approximate baseline drift magnitude via <0.5 Hz low-pass component (peak-to-peak)."""

    try:
        from scipy.signal import butter, sosfiltfilt
    except Exception as e:
        raise RuntimeError("Missing dependency scipy. Install with: pip install scipy") from e

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        return float("nan")

    cutoff = min(0.5, 0.45 * fs)
    cutoff = max(cutoff, 0.05)
    sos = butter(N=2, Wn=cutoff / (fs / 2.0), btype="low", output="sos")
    baseline = sosfiltfilt(sos, x.astype(np.float64))
    return float(np.max(baseline) - np.min(baseline))


def welch_psd(x: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Welch PSD for frequency-domain comparison."""

    try:
        from scipy.signal import welch
    except Exception as e:
        raise RuntimeError("Missing dependency scipy. Install with: pip install scipy") from e

    fs = float(sampling_rate_hz)
    x = x.astype(np.float64)
    nperseg = 2048
    if x.size < nperseg:
        nperseg = max(256, 2 ** int(np.floor(np.log2(max(256, x.size)))))
    f, pxx = welch(x, fs=fs, nperseg=int(nperseg))
    return np.asarray(f, dtype=np.float64), np.asarray(pxx, dtype=np.float64)


def _segments_from_mask(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Convert a boolean mask into contiguous index segments [start, end]."""

    if mask.size == 0:
        return []

    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0
    for i, flag in enumerate(mask):
        if flag and not in_seg:
            in_seg = True
            start = i
        elif not flag and in_seg:
            segments.append((start, i - 1))
            in_seg = False
    if in_seg:
        segments.append((start, mask.size - 1))

    return segments


def detect_noise(
    timestamps_ms: np.ndarray,
    signal: np.ndarray,
    sampling_rate_hz: float,
) -> Dict[str, object]:
    """Detect simple noisy regions for preprocessing triage.

    Requirement 9:
    - flatline sections
    - extreme spikes
    - abnormal jumps

    Heuristics used (simple + explainable):
    - Flatline: long run of identical values (or near-identical within tolerance)
    - Spikes: values far outside robust distribution (median +/- k * MAD)
    - Abnormal jumps: large sample-to-sample differences (diff MAD-based)

    Returns a dictionary with counts and example segments.
    """

    x = signal.astype(np.float64)
    n = x.size

    result: Dict[str, object] = {
        "flatline_segments": [],
        "spike_indices": [],
        "jump_indices": [],
        "flatline_count": 0,
        "spike_count": 0,
        "jump_count": 0,
    }

    if n < 3:
        return result

    fs = float(sampling_rate_hz)
    if not np.isfinite(fs) or fs <= 0:
        fs = 250.0

    # --- Flatline detection ---
    # Consider it flatline if consecutive differences are ~0 for at least 0.5 seconds.
    min_flat_sec = 0.5
    min_flat_len = max(5, int(min_flat_sec * fs))
    diff = np.abs(np.diff(x))
    flat_mask = diff < 1e-9

    flat_segments = _segments_from_mask(flat_mask)
    # Convert diff segments to sample-index segments (diff is length n-1).
    flat_sample_segments: List[Tuple[int, int]] = []
    for s, e in flat_segments:
        if (e - s + 1) >= min_flat_len:
            # diff index s..e means samples s..e+1 are flat-ish
            flat_sample_segments.append((s, e + 1))

    result["flatline_segments"] = flat_sample_segments
    result["flatline_count"] = int(len(flat_sample_segments))

    # --- Spike detection (robust) ---
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    # Avoid division by 0
    mad = mad if mad > 1e-12 else float(np.std(x))
    spike_k = 10.0
    spike_mask = np.abs(x - med) > spike_k * mad
    spike_indices = np.where(spike_mask)[0]
    result["spike_indices"] = spike_indices[:200].astype(int).tolist()  # cap output
    result["spike_count"] = int(spike_indices.size)

    # --- Abnormal jump detection (robust) ---
    dx = np.diff(x)
    med_dx = float(np.median(dx))
    mad_dx = float(np.median(np.abs(dx - med_dx)))
    mad_dx = mad_dx if mad_dx > 1e-12 else float(np.std(dx))
    jump_k = 12.0
    jump_mask = np.abs(dx - med_dx) > jump_k * mad_dx
    jump_indices = np.where(jump_mask)[0]  # index i means jump from i -> i+1
    result["jump_indices"] = jump_indices[:200].astype(int).tolist()
    result["jump_count"] = int(jump_indices.size)

    # Add a few readable examples with timestamps.
    # (We keep it small so the report is not massive.)
    examples: List[Dict[str, object]] = []
    for i in jump_indices[:10]:
        examples.append(
            {
                "idx": int(i),
                "t0_ms": int(timestamps_ms[i]),
                "t1_ms": int(timestamps_ms[i + 1]),
                "delta_value": float(dx[i]),
            }
        )
    result["jump_examples"] = examples

    return result


# ============================================================================
# QUALITY SCORING (LIGHTWEIGHT, SEGMENT-BASED)
# ============================================================================


QUALITY_LABELS: Tuple[str, ...] = ("EXCELLENT", "GOOD", "FAIR", "POOR", "UNUSABLE")
QUALITY_THRESHOLDS: Tuple[float, ...] = (92.0, 80.0, 65.0, 45.0, 0.0)


# CENTRALIZED QUALITY SCORING (integrated here per project requirement)
# ============================================================================


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _robust_range(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    lo = float(np.nanpercentile(x, 5.0))
    hi = float(np.nanpercentile(x, 95.0))
    rng = hi - lo
    return float(rng) if np.isfinite(rng) and rng > 0 else float(np.std(x)) if x.size else 0.0


def _linear_severity(value: float, *, ok_max: float, bad_min: float) -> float:
    if not np.isfinite(value):
        return 0.0
    if value <= ok_max:
        return 0.0
    if value >= bad_min:
        return 1.0
    return _clamp01((value - ok_max) / (bad_min - ok_max))


def _fft_power_ratios(x: np.ndarray, sampling_rate_hz: float, *, powerline_hz: float, raw_x: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Lightweight FFT-power based ratios. 
    If raw_x is provided, noise (powerline, HF) is measured on raw_x while 
    ECG band power is measured on x (cleaned) for a more accurate noise-to-signal ratio.
    """
    try:
        f, mag = compute_fft_magnitude(x, sampling_rate_hz)
        if raw_x is not None and raw_x.size == x.size:
            f_raw, mag_raw = compute_fft_magnitude(raw_x, sampling_rate_hz)
        else:
            f_raw, mag_raw = f, mag
    except Exception:
        return {"powerline_ratio": float("nan"), "hf_ratio": float("nan")}

    if f.size == 0 or f_raw.size == 0:
        return {"powerline_ratio": float("nan"), "hf_ratio": float("nan")}

    p = (mag ** 2).astype(np.float64)
    p_raw = (mag_raw ** 2).astype(np.float64)

    def band_power(f_arr, p_arr, lo, hi):
        mask = (f_arr >= lo) & (f_arr <= hi)
        return float(np.sum(p_arr[mask])) if np.any(mask) else 0.0

    nyq = 0.5 * float(sampling_rate_hz)
    ecg_lo, ecg_hi = 0.5, min(40.0, 0.95 * nyq)
    ecg_band = band_power(f, p, ecg_lo, ecg_hi)
    ecg_band = ecg_band if ecg_band > 1e-12 else 1e-12

    pl_lo, pl_hi = max(0.0, float(powerline_hz) - 1.0), float(powerline_hz) + 1.0
    powerline = band_power(f_raw, p_raw, pl_lo, pl_hi)

    hf_lo, hf_hi = 40.0, min(60.0, 0.95 * nyq)
    hf = band_power(f_raw, p_raw, hf_lo, hf_hi) if hf_hi > hf_lo else 0.0

    return {"powerline_ratio": float(powerline / ecg_band), "hf_ratio": float(hf / ecg_band)}


def _estimate_clipping_ratio(raw: np.ndarray) -> float:
    if raw.size < 8:
        return 0.0
    x = raw.astype(np.float64)
    high_thr = float(np.nanpercentile(x, 99.5))
    low_thr = float(np.nanpercentile(x, 0.5))
    dx = np.abs(np.diff(x))
    med_dx = float(np.nanmedian(dx)) if dx.size else 0.0
    eps = max(1e-9, 0.05 * med_dx)
    plateau = (((x[:-1] >= high_thr) | (x[:-1] <= low_thr)) & (dx <= eps))
    return float(np.mean(plateau))


def _rr_cv_from_rpeaks(seg_rpeaks: np.ndarray, sampling_rate_hz: float) -> float:
    if seg_rpeaks.size < 3:
        return float("nan")
    rr = np.diff(seg_rpeaks).astype(np.float64) / float(sampling_rate_hz)
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    if rr.size < 3:
        return float("nan")
    med = float(np.median(rr))
    if not np.isfinite(med) or med <= 1e-12:
        return float("nan")
    return float(np.std(rr) / med)


def score_ecg_quality(
    *,
    cleaned: np.ndarray,
    raw: Optional[np.ndarray],
    sampling_rate_hz: float,
    rpeaks: Optional[np.ndarray] = None,
    segment_seconds: float = 5.0,
    powerline_hz: float = 50.0,
    return_segment_details: bool = False,
) -> Dict[str, object]:
    """Conservative ECG quality scoring using raw noise and physiological checks."""
    x = np.asarray(cleaned, dtype=np.float64)
    fs = float(sampling_rate_hz)
    
    if x.size < 8 or not np.isfinite(fs) or fs <= 0:
        return {"score": 0.0, "label": "UNUSABLE", "metrics": {"reason": "insufficient_data"}, "recommendations": ["Insufficient data"]}

    seg_len = max(1, int(round(segment_seconds * fs)))
    total_segs = 0
    seg_scores: List[float] = []
    seg_labels: List[str] = []
    seg_details: List[Dict[str, object]] = []

    # Penalty weights (conservative/strict)
    W = {
        "drift": 35.0,
        "powerline": 45.0,
        "hf": 45.0,
        "clipping": 60.0,
        "missing": 50.0,
        "flatline": 100.0,
        "rr": 55.0,        # Increased weight for heart rate stability
        "similarity": 40.0  # New: penalty for inconsistent beat shapes
    }

    # Compute beat similarity if rpeaks are available
    sim_scores = np.asarray([])
    if rpeaks is not None and rpeaks.size >= 3:
        try:
            sim_scores = beat_similarity_scores(cleaned, rpeaks, fs)
        except Exception:
            sim_scores = np.asarray([])

    for start in range(0, x.size, seg_len):
        end = min(x.size, start + seg_len)
        if end - start < max(8, int(0.5 * seg_len)):
            continue
        total_segs += 1
        seg = x[start:end]
        raw_seg = raw[start:end] if (raw is not None and raw.size >= end) else None

        seg_rng = _robust_range(seg)
        seg_rng = seg_rng if seg_rng > 1e-12 else float(np.std(seg)) if seg.size else 1.0

        # Baseline drift measured on cleaned (baseline-removal effectiveness)
        try:
            drift_pp = baseline_drift_peak_to_peak(seg, fs)
        except Exception:
            drift_pp = float(np.max(seg) - np.min(seg))
        drift_ratio = float(drift_pp / seg_rng) if seg_rng > 0 else 0.0

        # Noise ratios MEASURED ON RAW SEGMENT if available
        ratios = _fft_power_ratios(seg, fs, powerline_hz=powerline_hz, raw_x=raw_seg)
        pl_ratio = float(ratios.get("powerline_ratio", float("nan")))
        hf_ratio = float(ratios.get("hf_ratio", float("nan")))

        clip_ratio = _estimate_clipping_ratio(raw_seg) if raw_seg is not None else 0.0
        miss_ratio = float(np.mean(~np.isfinite(raw_seg))) if raw_seg is not None else 0.0

        dx = np.abs(np.diff(seg)) if seg.size >= 2 else np.asarray([])
        flatline_ratio = float(np.mean(dx < 1e-9)) if dx.size else 0.0

        # RR CV and Peak Consistency
        rr_cv = float("nan")
        n_peaks = 0
        seg_sim = float("nan")
        if rpeaks is not None and rpeaks.size:
            p_mask = (rpeaks >= start) & (rpeaks < end)
            rp_seg = rpeaks[p_mask] - start
            n_peaks = int(rp_seg.size)
            rr_cv = _rr_cv_from_rpeaks(rp_seg.astype(np.int64), fs)
            
            if sim_scores.size > 0:
                # Extract similarity for peaks in this segment
                seg_sim_vals = sim_scores[p_mask]
                seg_sim_vals = seg_sim_vals[np.isfinite(seg_sim_vals)]
                if seg_sim_vals.size > 0:
                    seg_sim = float(np.mean(seg_sim_vals))

        # Severity mappings
        drift_sev = _linear_severity(drift_ratio, ok_max=0.12, bad_min=0.45)
        pl_sev = _linear_severity(pl_ratio, ok_max=0.005, bad_min=0.04) if np.isfinite(pl_ratio) else 0.25
        hf_sev = _linear_severity(hf_ratio, ok_max=0.05, bad_min=0.20) if np.isfinite(hf_ratio) else 0.25
        clip_sev = _linear_severity(clip_ratio, ok_max=0.0005, bad_min=0.005)
        miss_sev = _linear_severity(miss_ratio, ok_max=0.001, bad_min=0.01)
        fl_sev = 1.0 if flatline_ratio > 0.3 else (0.5 if flatline_ratio > 0.10 else 0.0)

        # RR severity: STRICT PENALTY for missing beats
        rr_sev = 0.0
        if n_peaks < 4:
            # Major penalty: Heartbeats are not clearly distinguishable from noise
            rr_sev = 0.90
        else:
            # cv ranges: <0.08 stable, 0.08-0.18 normal, >0.25 unstable
            rr_sev = _linear_severity(rr_cv, ok_max=0.08, bad_min=0.25) if np.isfinite(rr_cv) else 0.35

        # Similarity severity
        sim_sev = 0.0
        if np.isfinite(seg_sim):
            # Correlation < 0.85 is poor for a template
            sim_sev = _linear_severity(1.0 - seg_sim, ok_max=0.05, bad_min=0.20)
        elif n_peaks >= 4:
            # Peaks detected but similarity could not be computed -> moderate concern
            sim_sev = 0.45

        def scale_sev(s: float) -> float:
            return float(_clamp01(0.5 * s + 0.5 * (s ** 2)))

        penalties = {
            "drift": W["drift"] * scale_sev(drift_sev),
            "powerline": W["powerline"] * scale_sev(pl_sev),
            "hf": W["hf"] * scale_sev(hf_sev),
            "clipping": W["clipping"] * scale_sev(clip_sev),
            "missing": W["missing"] * scale_sev(miss_sev),
            "rr": W["rr"] * scale_sev(rr_sev),
            "flatline": W["flatline"] * scale_sev(fl_sev),
            "similarity": W["similarity"] * scale_sev(sim_sev)
        }

        seg_penalty = float(sum(penalties.values()))
        seg_score = float(max(0.0, min(100.0, 100.0 - seg_penalty)))

        # Reject if obviously unusable
        is_unusable = (fl_sev >= 1.0 or miss_ratio >= 0.15 or clip_ratio >= 0.02 or hf_sev >= 0.9 or (n_peaks < 2 and segment_seconds >= 3))
        seg_label = "UNUSABLE" if is_unusable else next((lab for thr, lab in zip(QUALITY_THRESHOLDS, QUALITY_LABELS) if seg_score >= thr), "UNUSABLE")

        seg_scores.append(seg_score)
        seg_labels.append(seg_label)
        seg_details.append({
            "start": int(start),
            "end": int(end),
            "score": seg_score,
            "label": seg_label,
            "metrics": {
                "drift_ratio": drift_ratio, 
                "powerline_ratio": pl_ratio, 
                "hf_ratio": hf_ratio, 
                "clipping_ratio": clip_ratio, 
                "missing_ratio": miss_ratio, 
                "flatline_ratio": flatline_ratio, 
                "rr_cv": rr_cv,
                "similarity": seg_sim
            },
            "penalties": penalties,
        })

    usable = [s for s, l in zip(seg_scores, seg_labels) if l != "UNUSABLE"]
    usable_count = len(usable)
    usable_ratio = float(usable_count / max(1, total_segs))

    if usable_count == 0:
        file_score = 0.0
    else:
        arr = np.asarray(usable, dtype=np.float64)
        # Conservative aggregate: weighted average of median and lower quartile
        file_score = 0.6 * float(np.median(arr)) + 0.4 * float(np.percentile(arr, 25.0))
        # Coverage penalty
        file_score *= float(0.4 + 0.6 * usable_ratio)

    file_label = next((lab for thr, lab in zip(QUALITY_THRESHOLDS, QUALITY_LABELS) if file_score >= thr), "UNUSABLE")

    metrics = {
        "segment_seconds": float(segment_seconds),
        "segment_count": int(total_segs),
        "usable_segment_count": int(usable_count),
        "usable_segment_ratio": float(usable_ratio),
        "baseline_drift_ratio_median": float(np.median([d["metrics"]["drift_ratio"] for d in seg_details])) if seg_details else 0.0,
        "powerline_ratio_median": float(np.median([d["metrics"]["powerline_ratio"] for d in seg_details if np.isfinite(d["metrics"]["powerline_ratio"])])) if seg_details else 0.0,
        "hf_ratio_median": float(np.median([d["metrics"]["hf_ratio"] for d in seg_details if np.isfinite(d["metrics"]["hf_ratio"])])) if seg_details else 0.0,
        "clipping_ratio_median": float(np.median([d["metrics"]["clipping_ratio"] for d in seg_details])) if seg_details else 0.0,
        "missing_ratio_median": float(np.median([d["metrics"]["missing_ratio"] for d in seg_details])) if seg_details else 0.0,
        "rr_cv_median": float(np.median([d["metrics"]["rr_cv"] for d in seg_details if np.isfinite(d["metrics"]["rr_cv"])])) if seg_details else 0.0,
        "similarity_median": float(np.median([d["metrics"]["similarity"] for d in seg_details if np.isfinite(d["metrics"]["similarity"])])) if seg_details else 0.0,
    }

    recs: List[str] = []
    if file_label in ("EXCELLENT", "GOOD"):
        recs.append("Signal high quality, suitable for ML.")
    if usable_ratio < 0.7:
        recs.append("Significant portions of the signal are unusable due to noise.")
    if metrics["hf_ratio_median"] > 0.15:
        recs.append("High HF noise detected (EMG/Motion). Check electrode placement.")
    if metrics["powerline_ratio_median"] > 0.04:
        recs.append("Strong powerline interference. Ensure grounding.")

    out = {
        "score": float(file_score),
        "label": str(file_label),
        "metrics": metrics,
        "penalties": {"weights": W},
        "recommendations": recs,
        "usable_segment_percentage": float(usable_ratio),
    }
    if return_segment_details:
        out.update({"segment_scores": seg_scores, "segment_labels": seg_labels, "segment_details": seg_details})

    return out


def plot_ecg(
    timestamps_ms: np.ndarray,
    raw: np.ndarray,
    cleaned: np.ndarray,
    sampling_rate_hz: float,
    plot_samples: int,
) -> None:
    """Plot raw and cleaned ECG for quick visual validation."""

    import matplotlib.pyplot as plt

    n = int(min(plot_samples, raw.size, cleaned.size))
    if n <= 0:
        print("No samples to plot.")
        return

    t = timestamps_ms[:n].astype(np.float64)
    # Convert absolute epoch ms -> relative seconds for nicer plots
    t = (t - t[0]) / 1000.0

    plt.figure(figsize=(12, 5))
    plt.plot(t, raw[:n], label="Raw ECG", linewidth=0.8, alpha=0.75)
    plt.plot(t, cleaned[:n], label="NeuroKit2 cleaned ECG", linewidth=0.9)

    plt.title("ECG waveform (raw vs cleaned)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (raw units)")
    if np.isfinite(sampling_rate_hz):
        plt.suptitle(f"Estimated sampling rate: {sampling_rate_hz:.2f} Hz", y=0.98, fontsize=10)

    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.figsize": (14, 8),
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 1.25,
        }
    )


def save_plots(
    out_dir: str,
    base_name: str,
    timestamps_ms: np.ndarray,
    raw: np.ndarray,
    cleaned: np.ndarray,
    sampling_rate_hz: float,
    rpeaks: np.ndarray,
) -> List[str]:
    """Generate and save the 6 required scientific plots as PNG.
    
    Args:
        out_dir: Output directory path.
        base_name: Base filename (without extension) extracted from input CSV.
                   Used to prefix all output plots, ensuring no file overwrites
                   when batch processing multiple ECG files.
        timestamps_ms: ECG timestamps in milliseconds.
        raw: Raw ECG signal.
        cleaned: Cleaned/filtered ECG signal.
        sampling_rate_hz: Estimated sampling rate.
        rpeaks: Detected R-peak indices.
    
    Returns:
        List of saved plot file paths.
    """

    _configure_matplotlib()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    n = min(raw.size, cleaned.size, timestamps_ms.size)
    ts = timestamps_ms[:n].astype(np.int64)
    raw = raw[:n].astype(np.float64)
    cleaned = cleaned[:n].astype(np.float64)
    t_s = (ts - ts[0]).astype(np.float64) / 1000.0

    saved: List[str] = []

    def _save(fig: "plt.Figure", suffix: str) -> None:
        """Save figure with base_name prefix to prevent file collisions in batch processing."""
        filename = f"{base_name}_{suffix}"
        path = os.path.join(out_dir, filename)
        fig.tight_layout()
        fig.savefig(path, dpi=300)
        saved.append(path)
        plt.close(fig)

    # A) Raw full
    fig = plt.figure()
    plt.plot(t_s, raw, color="tab:gray")
    plt.grid(True, alpha=0.25)
    plt.title("Raw ECG Signal (Full)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (sensor units)")
    _save(fig, "raw_ecg.png")

    # B) Cleaned full
    fig = plt.figure()
    plt.plot(t_s, cleaned, color="tab:blue")
    plt.grid(True, alpha=0.25)
    plt.title("Cleaned ECG Signal (Full)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (sensor units)")
    _save(fig, "cleaned_ecg.png")

    # C) Raw vs Cleaned overlay
    fig = plt.figure()
    plt.plot(t_s, raw, label="Raw", color="tab:gray", alpha=0.65)
    plt.plot(t_s, cleaned, label="Cleaned", color="tab:blue")
    plt.grid(True, alpha=0.25)
    plt.title("Raw vs Cleaned ECG (Overlay)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (sensor units)")
    plt.legend()
    _save(fig, "raw_vs_cleaned.png")

    # D) Zoomed window (first 10 seconds)
    zoom_s = 10.0
    zoom_n = int(min(n, max(1, round(zoom_s * float(sampling_rate_hz)))))
    fig = plt.figure()
    plt.plot(t_s[:zoom_n], cleaned[:zoom_n], color="tab:blue")
    plt.grid(True, alpha=0.25)
    plt.title("Zoomed ECG Window (First 10 seconds, Cleaned)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (sensor units)")
    _save(fig, "zoomed_ecg_window.png")

    # E) Frequency spectrum (Welch PSD)
    f_raw, p_raw = welch_psd(raw, sampling_rate_hz)
    f_cl, p_cl = welch_psd(cleaned, sampling_rate_hz)
    fig = plt.figure()
    plt.semilogy(f_raw, p_raw, label="Raw", color="tab:gray", alpha=0.8)
    plt.semilogy(f_cl, p_cl, label="Cleaned", color="tab:blue")
    plt.xlim(0, min(60.0, 0.5 * float(sampling_rate_hz)))
    plt.grid(True, alpha=0.25)
    plt.title("Frequency Spectrum (Welch PSD)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    _save(fig, "frequency_spectrum.png")

    # F) R-peak detection plot (cleaned + markers)
    fig = plt.figure()
    plt.plot(t_s, cleaned, color="tab:blue", label="Cleaned ECG")
    if rpeaks.size:
        rp = rpeaks[(rpeaks >= 0) & (rpeaks < n)]
        plt.scatter(t_s[rp], cleaned[rp], s=18, color="tab:red", label="R-peaks", zorder=3)
    plt.grid(True, alpha=0.25)
    plt.title("R-Peak Detection (Cleaned ECG)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (sensor units)")
    plt.legend()
    _save(fig, "r_peaks_detection.png")

    return saved


def save_cleaned_csv(
    out_path: str,
    timestamps_ms: np.ndarray,
    cleaned: np.ndarray,
) -> None:
    """Save cleaned ECG for downstream ML pipelines."""

    ts = timestamps_ms.astype(np.int64)
    t_s = (ts - ts[0]).astype(np.float64) / 1000.0
    z, _, _ = _robust_scale(cleaned.astype(np.float64))
    out_df = pd.DataFrame(
        {
            "timestamp_ms": ts,
            "timestamp_s": t_s,
            "clean_ecg": cleaned.astype(np.float64),
            "clean_ecg_z": z.astype(np.float64),
        }
    )
    out_df.to_csv(out_path, index=False)


def save_report(
    out_path: str,
    input_path: str,
    sampling: SamplingInfo,
    stats_raw: SignalStats,
    stats_clean: SignalStats,
    filters_used: List[str],
    noise: Dict[str, object],
    duration_s: float,
    snr: Tuple[float, float, float],
    snr_band: Tuple[float, float, float],
    baseline_raw_pp: float,
    baseline_clean_pp: float,
    peak_metrics: PeakMetrics,
    plots_saved: List[str],
) -> None:
    """Write a professional ECG preprocessing + quality report."""

    lines: List[str] = []

    lines.append("ECG cleaning report")
    lines.append("===================")
    lines.append("")
    lines.append(f"Input file: {os.path.basename(input_path)}")
    lines.append("")

    lines.append("Sampling")
    lines.append("--------")
    lines.append(f"Average interval (ms): {sampling.avg_interval_ms:.4f}")
    lines.append(f"Estimated sampling rate (Hz): {sampling.sampling_rate_hz:.2f}")
    lines.append(f"Signal duration (s): {duration_s:.2f}")
    lines.append("")

    lines.append("Cleaning / filtering methods")
    lines.append("----------------------------")
    for m in filters_used:
        lines.append(f"- {m}")
    lines.append("")

    def _stats_block(title: str, s: SignalStats) -> List[str]:
        return [
            title,
            "-" * len(title),
            f"Mean:  {s.mean:.4f}",
            f"Std:   {s.std:.4f}",
            f"Min:   {s.min:.4f}",
            f"Max:   {s.max:.4f}",
            f"Range: {s.range:.4f}",
            "",
        ]

    lines.extend(_stats_block("Signal statistics (raw)", stats_raw))
    lines.extend(_stats_block("Signal statistics (cleaned)", stats_clean))

    lines.append("Quality metrics")
    lines.append("--------------")
    snr_raw, snr_clean, snr_impr = snr
    snrbr, snrbc, snrbd = snr_band
    lines.append(f"Baseline drift (raw, <0.5 Hz component peak-to-peak):    {baseline_raw_pp:.4f}")
    lines.append(f"Baseline drift (cleaned, <0.5 Hz component peak-to-peak): {baseline_clean_pp:.4f}")
    if np.isfinite(snr_impr):
        lines.append(f"Estimated SNR raw (dB):     {snr_raw:.2f}")
        lines.append(f"Estimated SNR cleaned (dB): {snr_clean:.2f}")
        lines.append(f"Estimated SNR improvement:  {snr_impr:+.2f} dB")
    else:
        lines.append("Estimated SNR improvement:  (insufficient data)")
    if np.isfinite(snrbd):
        lines.append(f"Freq-domain SNR raw (dB, 0.5-40 Hz vs >40 Hz):     {snrbr:.2f}")
        lines.append(f"Freq-domain SNR cleaned (dB, 0.5-40 Hz vs >40 Hz): {snrbc:.2f}")
        lines.append(f"Freq-domain SNR improvement:                         {snrbd:+.2f} dB")
    lines.append("")

    lines.append("R-peak detection")
    lines.append("----------------")
    lines.append(f"Detected R-peaks: {int(peak_metrics.rpeaks.size)}")
    lines.append(f"Estimated average heart rate (BPM): {peak_metrics.heart_rate_bpm:.2f}")
    rr = peak_metrics.rr_stats
    rrv = peak_metrics.rr_stats_valid
    lines.append(f"RR interval count: {rr.count}")
    lines.append(f"RR mean (s): {rr.mean_s:.4f}")
    lines.append(f"RR std (s):  {rr.std_s:.4f}")
    lines.append(f"RR min (s):  {rr.min_s:.4f}")
    lines.append(f"RR max (s):  {rr.max_s:.4f}")
    lines.append(f"RR invalid fraction (outside 0.30-2.00 s): {peak_metrics.rr_invalid_fraction:.2%}" if np.isfinite(peak_metrics.rr_invalid_fraction) else "RR invalid fraction: (n/a)")
    lines.append("")
    lines.append("RR stats (valid only, 0.30-2.00 s)")
    lines.append("--------------------------------")
    lines.append(f"Valid RR count: {rrv.count}")
    lines.append(f"Valid RR mean (s): {rrv.mean_s:.4f}")
    lines.append(f"Valid RR std (s):  {rrv.std_s:.4f}")
    lines.append(f"Valid RR min (s):  {rrv.min_s:.4f}")
    lines.append(f"Valid RR max (s):  {rrv.max_s:.4f}")
    lines.append("")

    lines.append("Noise checks (simple heuristics)")
    lines.append("-------------------------------")
    lines.append(f"Flatline segments: {int(noise.get('flatline_count', 0))}")
    lines.append(f"Spike samples:     {int(noise.get('spike_count', 0))}")
    lines.append(f"Abnormal jumps:    {int(noise.get('jump_count', 0))}")

    flat_segments = noise.get("flatline_segments", [])
    if flat_segments:
        lines.append("")
        lines.append("Flatline examples (up to 5 segments):")
        for s, e in flat_segments[:5]:
            lines.append(
                f"- idx {s}..{e} (t={int(s)}..{int(e)} samples)"
            )

    jump_examples = noise.get("jump_examples", [])
    if jump_examples:
        lines.append("")
        lines.append("Jump examples (first 10):")
        for ex in jump_examples:
            lines.append(
                f"- idx={ex['idx']} t0={ex['t0_ms']} t1={ex['t1_ms']} delta_value={ex['delta_value']:.2f}"
            )

    lines.append("")
    lines.append("Preprocessing summary")
    lines.append("---------------------")

    lines.append("Why these steps matter (short explanation)")
    lines.append("-----------------------------------------")
    lines.append("- High-pass (~0.5 Hz): removes baseline wander from breathing/motion/electrodes.")
    lines.append("- Notch (50 Hz): removes mains hum (powerline interference) without affecting QRS.")
    lines.append("- Low-pass (~40 Hz): suppresses high-frequency noise/EMG while preserving QRS morphology.")
    lines.append("- Hampel spike filter: suppresses impulsive spikes (motion/ADC glitches) without smoothing the entire waveform.")
    lines.append("- Resampling (optional): converts jittery timestamps into constant sampling, improving filters and peak detectors.")
    lines.append("")

    structurally_ok = np.isfinite(sampling.sampling_rate_hz) and sampling.sampling_rate_hz > 0
    if structurally_ok:
        lines.append("- Sampling rate estimated successfully.")
    else:
        lines.append("- WARNING: sampling rate could not be estimated reliably.")

    if int(noise.get("flatline_count", 0)) > 0 or int(noise.get("spike_count", 0)) > 0:
        lines.append("- WARNING: noisy sections detected; consider trimming or additional denoising.")
    else:
        lines.append("- No major flatlines/spikes detected by simple heuristics.")

    lines.append("- Plots saved:")
    for p in plots_saved:
        lines.append(f"  - {os.path.basename(p)}")

    lines.append("")
    lines.append("Conclusion: This report is for preprocessing validation, not medical diagnosis.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Advanced ECG preprocessing + visualization pipeline (outputs cleaned ECG, report, and plots)."
    )
    parser.add_argument(
        "csv_path",
        help="Input ECG CSV (timestamp_ms, ecg_value).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show an interactive raw vs cleaned plot window (optional).",
    )
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=10000,
        help="How many samples to plot (default: 10000).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs (default: ./output next to this script).",
    )

    parser.add_argument(
        "--powerline-hz",
        type=float,
        default=50.0,
        help="Powerline frequency for notch filter (default: 50 Hz).",
    )

    parser.add_argument(
        "--no-resample",
        action="store_true",
        help="Disable timestamp resampling to a uniform grid (not recommended).",
    )

    parser.add_argument(
        "--neurokit-clean",
        action="store_true",
        help="Enable optional NeuroKit2 ecg_clean refinement step (may over-smooth for some sensors).",
    )

    # Backward-compatible flag (previous versions used --no-neurokit).
    parser.add_argument(
        "--no-neurokit",
        action="store_true",
        help="(Deprecated) Disable NeuroKit2 refinement. Use SciPy filters only.",
    )

    args = parser.parse_args(argv)

    csv_path = args.csv_path
    if not os.path.exists(csv_path):
        print(f"ERROR: file not found: {csv_path}")
        return 2

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(script_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    # Extract base filename (without extension) from input CSV.
    # This ensures all output files use the same prefix, preventing overwrites
    # when batch processing multiple ECG files with the same script.
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    cleaned_csv_path = os.path.join(out_dir, f"{base_name}_cleaned_ecg.csv")
    report_path = os.path.join(out_dir, f"{base_name}_ecg_report.txt")

    # 1) Load + sanitize
    df = sanitize_ecg(load_ecg(csv_path))

    timestamps_ms = df["timestamp_ms"].to_numpy(dtype=np.int64)
    ecg_value = df["ecg_value"].to_numpy(dtype=np.float64)

    # 2) Estimate sampling
    sampling = estimate_sampling_rate(timestamps_ms)
    if not np.isfinite(sampling.sampling_rate_hz) or sampling.sampling_rate_hz <= 0:
        print("ERROR: Could not estimate sampling rate from timestamps.")
        return 3

    # 3) Cleaning pipeline
    resample = not bool(args.no_resample)
    neurokit_clean = bool(args.neurokit_clean) and (not bool(args.no_neurokit))
    ts_aligned, raw_aligned, cleaned, steps = clean_signal_advanced(
        timestamps_ms,
        ecg_value,
        sampling.sampling_rate_hz,
        resample=resample,
        powerline_hz=float(args.powerline_hz),
        neurokit_clean=neurokit_clean,
    )

    # 4) Stats before / after
    stats_raw = compute_stats(raw_aligned)
    stats_clean = compute_stats(cleaned)

    # 5) Noise detection (run on raw aligned)
    noise = detect_noise(ts_aligned, raw_aligned, sampling.sampling_rate_hz)

    # 6) R-peaks (robust)
    peak_metrics = detect_rpeaks_robust(cleaned, sampling.sampling_rate_hz)

    # 7) Quality metrics
    duration_s = float((ts_aligned[-1] - ts_aligned[0]) / 1000.0) if ts_aligned.size >= 2 else float("nan")
    snr = estimate_snr_improvement(raw_aligned, cleaned)
    snr_band = estimate_snr_bandpower(raw_aligned, cleaned, sampling.sampling_rate_hz)
    baseline_raw_pp = baseline_drift_peak_to_peak(raw_aligned, sampling.sampling_rate_hz)
    baseline_clean_pp = baseline_drift_peak_to_peak(cleaned, sampling.sampling_rate_hz)

    # 8) Plot raw and cleaned (interactive optional)
    if args.plot:
        plot_ecg(ts_aligned, raw_aligned, cleaned, sampling.sampling_rate_hz, plot_samples=args.plot_samples)

    # 9) Save cleaned output
    save_cleaned_csv(cleaned_csv_path, ts_aligned, cleaned)

    # 10) Save required plots
    plots_saved = save_plots(
        out_dir=out_dir,
        base_name=base_name,
        timestamps_ms=ts_aligned,
        raw=raw_aligned,
        cleaned=cleaned,
        sampling_rate_hz=sampling.sampling_rate_hz,
        rpeaks=peak_metrics.rpeaks,
    )

    # 11) Save report
    filters_used = steps
    save_report(
        report_path,
        input_path=csv_path,
        sampling=sampling,
        stats_raw=stats_raw,
        stats_clean=stats_clean,
        filters_used=filters_used,
        noise=noise,
        duration_s=duration_s,
        snr=snr,
        snr_band=snr_band,
        baseline_raw_pp=baseline_raw_pp,
        baseline_clean_pp=baseline_clean_pp,
        peak_metrics=peak_metrics,
        plots_saved=plots_saved,
    )

    print("\nECG preprocessing complete")
    print("--------------------------")
    print(f"Output directory:     {out_dir}")
    print(f"Cleaned ECG saved to: {cleaned_csv_path}")
    print(f"Report saved to:      {report_path}")
    print(f"Sampling rate (Hz):   {sampling.sampling_rate_hz:.2f}")
    print(f"Duration (s):         {duration_s:.2f}")
    print("")
    print("Signal stats")
    print("-----------")
    print(f"Raw mean/std:     {stats_raw.mean:.3f} / {stats_raw.std:.3f}")
    print(f"Clean mean/std:   {stats_clean.mean:.3f} / {stats_clean.std:.3f}")
    print(f"Baseline drift p2p (raw -> clean): {baseline_raw_pp:.3f} -> {baseline_clean_pp:.3f}")
    if np.isfinite(snr_band[2]):
        print(f"Freq-domain SNR improvement (0.5–40 Hz vs >40 Hz): {snr_band[2]:+.2f} dB")
    print("")
    print(f"R-peaks detected:     {int(peak_metrics.rpeaks.size)}")
    print(f"Avg heart rate (BPM): {peak_metrics.heart_rate_bpm:.2f}")
    if np.isfinite(peak_metrics.rr_invalid_fraction):
        print(f"RR invalid fraction:  {peak_metrics.rr_invalid_fraction:.2%}")
    rrv = peak_metrics.rr_stats_valid
    if rrv.count > 0:
        print(f"Valid RR mean/std (s): {rrv.mean_s:.4f} / {rrv.std_s:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
