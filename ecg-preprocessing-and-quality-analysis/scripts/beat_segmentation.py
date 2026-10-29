"""Beat-aligned ECG segmentation and template analysis.

Extracts fixed windows around R-peaks for beat-level ML features, builds an
average beat template, and scores per-beat similarity to that template.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def segment_by_beats(
    output_dir: Path,
    *,
    timestamps_ms: np.ndarray,
    cleaned: np.ndarray,
    normalized: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate_hz: float,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> int:
    """Save one CSV with all aligned clean and z-scored ECG snippets.

    Args:
        output_dir: Directory where all_beats.csv will be created.
        timestamps_ms: Millisecond timestamps aligned with cleaned/normalized.
        cleaned: Filtered ECG amplitude array.
        normalized: Z-scored ECG array (ecg_z).
        rpeaks: Sample indices of R-peaks.
        sampling_rate_hz: Sampling rate used to convert pre/post ms to samples.
        pre_ms: Window length before each R-peak in milliseconds.
        post_ms: Window length after each R-peak in milliseconds.

    Returns:
        Number of beats written to the combined CSV file.

    Notes:
        Skips beats whose window would extend outside the signal bounds.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fs = float(sampling_rate_hz)
    pre_samples = int(round(pre_ms * fs / 1000.0))
    post_samples = int(round(post_ms * fs / 1000.0))
    window_len = pre_samples + post_samples

    ts = timestamps_ms.astype(np.int64)
    clean = cleaned.astype(np.float64)
    norm = normalized.astype(np.float64)
    peaks = np.asarray(rpeaks, dtype=np.int64)

    saved = 0
    beats_data = []
    offsets = np.arange(-pre_samples, post_samples, dtype=np.int64)

    for rpeak in peaks:
        start = int(rpeak) - pre_samples
        end = int(rpeak) + post_samples
        if start < 0 or end > clean.size:
            continue

        saved += 1
        beats_data.append(
            pd.DataFrame(
                {
                    "beat_id": saved,
                    "timestamp_ms": ts[start:end],
                    "clean_ecg": clean[start:end],
                    "ecg_z": norm[start:end],
                    "sample_offset": offsets,
                }
            )
        )

    if beats_data:
        pd.concat(beats_data, ignore_index=True).to_csv(output_dir / "all_beats.csv", index=False)

    return saved


def compute_beat_template(
    cleaned: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate_hz: float,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute median beat template and per-sample standard deviation.

    Args:
        cleaned: Filtered ECG signal.
        rpeaks: R-peak sample indices.
        sampling_rate_hz: Sampling rate in Hz.
        pre_ms: Pre-R window in milliseconds.
        post_ms: Post-R window in milliseconds.

    Returns:
        Tuple (template, std_curve). Empty arrays if no valid beats exist.
    """
    fs = float(sampling_rate_hz)
    pre_samples = int(round(pre_ms * fs / 1000.0))
    post_samples = int(round(post_ms * fs / 1000.0))

    clean = cleaned.astype(np.float64)
    peaks = np.asarray(rpeaks, dtype=np.int64)
    windows: list[np.ndarray] = []

    for rpeak in peaks:
        start = int(rpeak) - pre_samples
        end = int(rpeak) + post_samples
        if start < 0 or end > clean.size:
            continue
        windows.append(clean[start:end])

    if not windows:
        return np.array([]), np.array([])

    stack = np.vstack(windows)
    template = np.median(stack, axis=0)
    std_curve = np.std(stack, axis=0)
    return template.astype(np.float64), std_curve.astype(np.float64)


def beat_similarity_scores(
    cleaned: np.ndarray,
    rpeaks: np.ndarray,
    sampling_rate_hz: float,
    pre_ms: float = 200.0,
    post_ms: float = 400.0,
) -> np.ndarray:
    """Pearson correlation between each valid beat and the median template.

    Args:
        cleaned: Filtered ECG signal.
        rpeaks: R-peak sample indices.
        sampling_rate_hz: Sampling rate in Hz.
        pre_ms: Pre-R window in milliseconds.
        post_ms: Post-R window in milliseconds.

    Returns:
        1D array of correlation coefficients with same size as rpeaks. 
        Empty if fewer than 3 valid beats are found.
    """
    template, _ = compute_beat_template(
        cleaned, rpeaks, sampling_rate_hz, pre_ms=pre_ms, post_ms=post_ms
    )
    if template.size == 0:
        return np.array([])

    fs = float(sampling_rate_hz)
    pre_samples = int(round(pre_ms * fs / 1000.0))
    post_samples = int(round(post_ms * fs / 1000.0))

    clean = cleaned.astype(np.float64)
    peaks = np.asarray(rpeaks, dtype=np.int64)
    scores: list[float] = []

    for rpeak in peaks:
        start = int(rpeak) - pre_samples
        end = int(rpeak) + post_samples
        if start < 0 or end > clean.size:
            scores.append(float("nan"))
            continue
        beat = clean[start:end]
        if beat.size != template.size or np.std(beat) < 1e-12 or np.std(template) < 1e-12:
            scores.append(float("nan"))
            continue
        corr = float(np.corrcoef(beat, template)[0, 1])
        scores.append(corr)

    arr = np.asarray(scores, dtype=np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size < 3:
        return np.array([])
    return arr


def save_beat_template_plot(
    output_path: Path,
    template: np.ndarray,
    std_curve: np.ndarray,
    sampling_rate_hz: float,
    pre_ms: float = 200.0,
) -> None:
    """Save a beat template plot with ±1 std shading.

    Args:
        output_path: PNG destination path.
        template: Median beat waveform.
        std_curve: Per-sample standard deviation curve.
        sampling_rate_hz: Sampling rate for time-axis conversion.
        pre_ms: Pre-R window in milliseconds (sets time origin at R-peak).

    Notes:
        Plot title uses the template window length (number of samples per beat).
    """
    import matplotlib.pyplot as plt

    if template.size == 0:
        return

    fs = float(sampling_rate_hz)
    pre_samples = int(round(pre_ms * fs / 1000.0))
    n = template.size
    t_ms = (np.arange(n, dtype=np.float64) - pre_samples) * (1000.0 / fs)

    tpl = template.astype(np.float64)
    std = std_curve.astype(np.float64)
    if std.size != tpl.size:
        std = np.zeros_like(tpl)

    fig, ax = plt.subplots()
    ax.plot(t_ms, tpl, color="tab:blue", linewidth=1.5, label="Template")
    ax.fill_between(
        t_ms,
        tpl - std,
        tpl + std,
        color="tab:blue",
        alpha=0.3,
        label="±1 std",
    )
    ax.set_xlabel("Time relative to R-peak (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Average Beat Template (n={len(tpl)} beats)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
