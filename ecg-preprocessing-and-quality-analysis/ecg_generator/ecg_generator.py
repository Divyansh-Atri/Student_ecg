"""
ecg_generator.py

TWO capabilities:
1. render_ecg_from_csv()     — takes your cleaned_ecg.csv and renders a
                               publication-style ECG strip with PQRST annotations
2. synthesize_pqrst_ecg()   — generates a fully synthetic ECG signal with
                               realistic PQRST morphology, configurable heart rate,
                               noise level, and pathology type
3. augment_dataset()         — generates N synthetic recordings and saves them
                               as pipeline-compatible CSVs for training

Run:
    python ecg_generator.py --mode render --input Processed_ECG/.../cleaned_ecg.csv
    python ecg_generator.py --mode synth  --output DATA/RAW/SYNTHETIC --count 500
    python ecg_generator.py --mode augment --output DATA/RAW/SYNTHETIC --count 1000
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: RENDER EXISTING CSV AS PROPER ECG STRIP
# ═══════════════════════════════════════════════════════════════════════════════

def render_ecg_from_csv(
    csv_path: str,
    output_path: Optional[str] = None,
    fs: float = 125.0,
    duration_s: float = 10.0,
    annotate_rpeaks: bool = True,
    title: str = "ECG Strip",
) -> None:
    """
    Render your cleaned_ecg.csv as a proper ECG strip plot.
    
    What it does:
    - Loads your CSV (timestamp_ms + clean_ecg or ecg_z column)
    - Renders standard ECG paper style (25mm/s, pink grid, mV scale)
    - Optionally detects and marks R-peaks, P-waves, T-waves
    - Saves as high-res PNG you can embed in your dashboard
    
    Args:
        csv_path:    Path to cleaned_ecg.csv or normalized_ecg.csv
        output_path: Where to save the PNG (None = show interactively)
        fs:          Sampling rate in Hz (default 125)
        duration_s:  How many seconds to show (default 10)
        annotate_rpeaks: Whether to mark R-peaks with red triangles
    """
    df = pd.read_csv(csv_path)

    # Accept either column name
    if "clean_ecg" in df.columns:
        signal = df["clean_ecg"].values
    elif "ecg_z" in df.columns:
        signal = df["ecg_z"].values
    elif "ecg_value" in df.columns:
        signal = df["ecg_value"].values
    else:
        signal = df.iloc[:, 1].values

    n_show = min(len(signal), int(duration_s * fs))
    signal = signal[:n_show]
    t = np.arange(n_show) / fs  # time in seconds

    fig, ax = plt.subplots(figsize=(20, 4))

    # ECG paper style
    ax.set_facecolor("#fff8f8")
    # Major grid every 0.2s and 0.5mV
    ax.set_xticks(np.arange(0, t[-1], 0.2), minor=False)
    ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(which="major", color="#f4a0a0", linewidth=0.8, alpha=0.7)
    ax.grid(which="minor", color="#f4a0a0", linewidth=0.3, alpha=0.4)

    # Plot signal
    ax.plot(t, signal, color="black", linewidth=0.8, label="ECG")

    # Annotate R-peaks if requested
    if annotate_rpeaks:
        try:
            from scipy.signal import find_peaks
            # R-peaks are the tallest peaks, at physiological HR range
            min_dist = int(fs * 0.3)  # minimum 300ms between beats
            peaks, props = find_peaks(signal, height=np.percentile(signal, 75), distance=min_dist)
            ax.scatter(t[peaks], signal[peaks], color="red", s=30, zorder=5,
                      marker="v", label=f"R-peaks (n={len(peaks)})")

            # Estimate and show HR
            if len(peaks) >= 2:
                rr_s = np.diff(t[peaks])
                hr = 60.0 / np.mean(rr_s)
                ax.set_title(f"{title}   |   HR ≈ {hr:.0f} bpm   |   Duration: {t[-1]:.1f}s   |   fs={fs:.0f}Hz",
                           fontsize=11, pad=8)
        except ImportError:
            pass

    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_ylabel("Amplitude", fontsize=10)
    if not annotate_rpeaks:
        ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, t[-1])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved ECG strip to: {output_path}")
        plt.close()
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: SYNTHESIZE PQRST WAVEFORM FROM SCRATCH
# ═══════════════════════════════════════════════════════════════════════════════

def _gaussian(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """A single Gaussian bump — used to model ECG wave components."""
    return amplitude * np.exp(-((t - center) ** 2) / (2 * width ** 2))


def synthesize_one_beat(
    fs: float = 125.0,
    rr_s: float = 0.8,
    pathology: str = "normal",
) -> np.ndarray:
    """
    Synthesize one PQRST beat cycle as a numpy array.
    
    The beat is built by summing Gaussian bumps for each wave:
    - P wave:  small, wide, ~160ms before R
    - Q wave:  small negative dip ~40ms before R
    - R wave:  tall positive spike (the QRS complex peak)
    - S wave:  small negative dip ~40ms after R
    - T wave:  medium positive hump ~300ms after R
    
    Args:
        fs:        sampling rate in Hz
        rr_s:      RR interval in seconds (= 60/HR)
        pathology: "normal", "af", "lbbb", "rbbb", "st_elevation", "flat_t"
    
    Returns:
        1D numpy array of length int(rr_s * fs)
    """
    n = int(rr_s * fs)
    t = np.linspace(0, rr_s, n)
    R_pos = rr_s * 0.5   # R-peak is in the middle of the beat window

    beat = np.zeros(n)

    if pathology == "normal":
        beat += _gaussian(t, R_pos - 0.16, 0.025, 0.15)   # P wave
        beat += _gaussian(t, R_pos - 0.04, 0.010, -0.10)  # Q wave
        beat += _gaussian(t, R_pos,         0.015,  1.00)  # R wave
        beat += _gaussian(t, R_pos + 0.04, 0.010, -0.15)  # S wave
        beat += _gaussian(t, R_pos + 0.30, 0.055,  0.35)  # T wave

    elif pathology == "af":
        # AF: no clear P wave, irregular baseline instead
        baseline_noise = 0.05 * np.sin(2 * np.pi * 8.0 * t + np.random.uniform(0, 2*np.pi))
        beat += baseline_noise
        beat += _gaussian(t, R_pos - 0.04, 0.010, -0.10)  # Q wave
        beat += _gaussian(t, R_pos,         0.015,  1.00)  # R wave
        beat += _gaussian(t, R_pos + 0.04, 0.010, -0.15)  # S wave
        beat += _gaussian(t, R_pos + 0.28, 0.060,  0.30)  # T wave (slightly flatter)

    elif pathology == "lbbb":
        # LBBB: wide, notched QRS
        beat += _gaussian(t, R_pos - 0.16, 0.025, 0.15)   # P wave
        beat += _gaussian(t, R_pos - 0.02, 0.025,  0.60)  # Broad R1
        beat += _gaussian(t, R_pos + 0.04, 0.025,  0.40)  # Notch R2
        beat += _gaussian(t, R_pos + 0.08, 0.010, -0.20)  # S wave
        beat += _gaussian(t, R_pos + 0.32, 0.065, -0.25)  # Inverted T (LBBB pattern)

    elif pathology == "st_elevation":
        # ST elevation: elevated baseline between S and T
        beat += _gaussian(t, R_pos - 0.16, 0.025, 0.15)   # P wave
        beat += _gaussian(t, R_pos - 0.04, 0.010, -0.10)  # Q wave
        beat += _gaussian(t, R_pos,         0.015,  1.00)  # R wave
        beat += _gaussian(t, R_pos + 0.04, 0.010, -0.15)  # S wave
        # ST elevation: add a plateau between S and T
        st_mask = (t > R_pos + 0.06) & (t < R_pos + 0.25)
        beat[st_mask] += 0.25
        beat += _gaussian(t, R_pos + 0.32, 0.055,  0.45)  # Tall T wave

    elif pathology == "flat_t":
        # Flat T wave (hypokalemia or ischemia)
        beat += _gaussian(t, R_pos - 0.16, 0.025, 0.15)
        beat += _gaussian(t, R_pos - 0.04, 0.010, -0.10)
        beat += _gaussian(t, R_pos,         0.015,  1.00)
        beat += _gaussian(t, R_pos + 0.04, 0.010, -0.15)
        beat += _gaussian(t, R_pos + 0.30, 0.055,  0.08)  # Nearly flat T

    return beat


def synthesize_pqrst_ecg(
    duration_s: float = 60.0,
    fs: float = 125.0,
    hr_bpm: float = 72.0,
    hr_variability: float = 0.05,
    noise_level: float = 0.03,
    baseline_drift: float = 0.05,
    powerline_hz: float = 0.0,
    powerline_amplitude: float = 0.0,
    pathology: str = "normal",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthesize a full ECG signal with realistic PQRST morphology.
    
    Args:
        duration_s:          Total recording length in seconds
        fs:                  Sampling rate (default 125Hz to match your pipeline)
        hr_bpm:              Mean heart rate in BPM
        hr_variability:      Fractional HR variability (0.05 = 5% variation)
        noise_level:         Gaussian noise amplitude
        baseline_drift:      Amplitude of slow 0.15Hz baseline wander
        powerline_hz:        Add powerline noise at this frequency (0 = off)
        powerline_amplitude: Amplitude of powerline noise
        pathology:           "normal","af","lbbb","rbbb","st_elevation","flat_t"
    
    Returns:
        timestamps_ms: int64 array
        signal:        float64 ECG signal
    """
    n_total = int(duration_s * fs)
    signal  = np.zeros(n_total)
    t_full  = np.arange(n_total) / fs

    # Build beat by beat
    sample_pos = 0
    while sample_pos < n_total:
        # Vary RR interval slightly for realism
        rr_s = (60.0 / hr_bpm) * (1.0 + np.random.normal(0, hr_variability))
        rr_s = np.clip(rr_s, 0.3, 2.0)

        beat = synthesize_one_beat(fs=fs, rr_s=rr_s, pathology=pathology)
        end_pos = min(sample_pos + len(beat), n_total)
        n_copy  = end_pos - sample_pos
        signal[sample_pos:end_pos] += beat[:n_copy]
        sample_pos = end_pos

    # Add baseline drift
    if baseline_drift > 0:
        drift = baseline_drift * np.sin(2 * np.pi * 0.15 * t_full + np.random.uniform(0, 2*np.pi))
        signal += drift

    # Add Gaussian noise
    if noise_level > 0:
        signal += np.random.normal(0, noise_level, n_total)

    # Add powerline noise
    if powerline_hz > 0 and powerline_amplitude > 0:
        signal += powerline_amplitude * np.sin(2 * np.pi * powerline_hz * t_full)

    timestamps_ms = (np.arange(n_total) * 1000.0 / fs).astype(np.int64)
    return timestamps_ms, signal.astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# PART C: BULK AUGMENTATION — generate N synthetic recordings
# ═══════════════════════════════════════════════════════════════════════════════

def augment_dataset(
    output_dir: Path,
    count: int = 500,
    fs: float = 125.0,
    duration_s: float = 60.0,
) -> None:
    """
    Generate N synthetic ECG recordings and save as pipeline-compatible CSVs.
    
    Creates a balanced mix of:
    - Normal sinus rhythm (various HRs)
    - Atrial fibrillation
    - LBBB (wide QRS)
    - ST elevation (ischemia pattern)
    - Flat T waves
    - Various noise levels (to stress-test your quality scorer)
    
    Also saves a labels.csv so you know the ground truth for each file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Mix of pathologies and noise levels to generate
    configs = [
        # (pathology, hr_range,    noise,  drift,  powerline, weight)
        ("normal",       (55, 100),  0.03, 0.05, 0.0,  0.35),  # 35% normal
        ("af",           (60, 120),  0.05, 0.08, 0.0,  0.15),  # 15% AF
        ("lbbb",         (50,  85),  0.03, 0.05, 0.0,  0.10),  # 10% LBBB
        ("st_elevation", (60,  90),  0.03, 0.05, 0.0,  0.10),  # 10% ST elevation
        ("flat_t",       (60, 100),  0.03, 0.05, 0.0,  0.10),  # 10% flat T
        ("normal",       (55, 100),  0.15, 0.20, 50.0, 0.10),  # 10% noisy normal
        ("normal",       (55, 100),  0.40, 0.50, 50.0, 0.10),  # 10% very noisy
    ]
    # Normalize weights
    weights = np.array([c[5] for c in configs])
    weights /= weights.sum()

    label_rows = []
    print(f"Generating {count} synthetic ECG recordings...")

    for i in range(count):
        # Pick config
        cfg_idx = np.random.choice(len(configs), p=weights)
        pathology, hr_range, noise, drift, pl_hz, _ = configs[cfg_idx]
        hr = np.random.uniform(*hr_range)
        pl_amp = 0.08 if pl_hz > 0 else 0.0

        timestamps, signal = synthesize_pqrst_ecg(
            duration_s=duration_s,
            fs=fs,
            hr_bpm=hr,
            noise_level=noise,
            baseline_drift=drift,
            powerline_hz=pl_hz,
            powerline_amplitude=pl_amp,
            pathology=pathology,
        )

        # Use timestamp as record ID (like your real sensor files)
        fake_ts = 1_700_000_000_000 + i * 70_000
        filename = f"{fake_ts}_SYNTH001_synthetic_ecg.csv"
        out_path = output_dir / filename

        pd.DataFrame({
            "timestamp_ms": timestamps,
            "ecg_value":    signal,
        }).to_csv(out_path, index=False)

        label_rows.append({
            "filename":   filename,
            "pathology":  pathology,
            "hr_bpm":     round(hr, 1),
            "noise_level": noise,
            "drift":      drift,
            "powerline_hz": pl_hz,
            "is_noisy":   noise > 0.1,
            "dataset":    "synthetic",
        })

    pd.DataFrame(label_rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"Done. {count} synthetic recordings saved to {output_dir}")
    print(f"Labels saved to {output_dir / 'labels.csv'}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["render","synth","augment"], required=True)
    parser.add_argument("--input",    default=None, help="Input CSV for render mode")
    parser.add_argument("--output",   default="DATA/RAW/SYNTHETIC")
    parser.add_argument("--count",    type=int, default=500)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fs",       type=float, default=125.0)
    args = parser.parse_args()

    if args.mode == "render":
        assert args.input, "Provide --input path to a cleaned_ecg.csv"
        render_ecg_from_csv(
            args.input,
            output_path=args.input.replace(".csv", "_strip.png"),
            fs=args.fs,
        )

    elif args.mode == "synth":
        ts, sig = synthesize_pqrst_ecg(duration_s=args.duration, fs=args.fs)
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"timestamp_ms": ts, "ecg_value": sig}).to_csv(
            out / "synthetic_ecg.csv", index=False
        )
        print(f"Saved synthetic ECG to {out / 'synthetic_ecg.csv'}")

    elif args.mode == "augment":
        augment_dataset(Path(args.output), count=args.count, fs=args.fs, duration_s=args.duration)