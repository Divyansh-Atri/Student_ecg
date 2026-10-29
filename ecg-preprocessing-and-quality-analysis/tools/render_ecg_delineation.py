import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
from pathlib import Path

def render_delineation(csv_path, output_dir, duration_sec=10):
    print(f"Reading ECG from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Ensure columns
    if 'ecg_value' not in df.columns:
        # Try second column if name is different
        val_col = df.columns[1]
    else:
        val_col = 'ecg_value'
        
    signal = df[val_col].values
    
    # Estimate sampling rate
    if 'timestamp_ms' in df.columns:
        diffs = np.diff(df['timestamp_ms'])
        avg_ms = np.median(diffs)
        fs = 1000.0 / avg_ms
    else:
        fs = 125.0 # default
    
    print(f"Estimated Sampling Rate: {fs:.2f} Hz")
    
    # Take a snippet for visualization
    n_samples = int(duration_sec * fs)
    if len(signal) > n_samples:
        signal = signal[:n_samples]
    
    # Clean signal
    cleaned = nk.ecg_clean(signal, sampling_rate=fs)
    
    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    
    # Delineate waves (P, QRS, T)
    print("Delineating P, QRS, and T waves...")
    try:
        _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=fs, method="peak")
        
        # Plotting
        plt.figure(figsize=(15, 8))
        plt.plot(cleaned, label='Cleaned ECG', color='black', alpha=0.5)
        
        # Plot R-peaks
        r_indices = rpeaks['ECG_R_Peaks']
        plt.scatter(r_indices, cleaned[r_indices], color='red', label='R-peaks (QRS)', zorder=5)
        
        # Plot T-peaks
        if 'ECG_T_Peaks' in waves_peak and waves_peak['ECG_T_Peaks']:
            t_indices = [idx for idx in waves_peak['ECG_T_Peaks'] if not np.isnan(idx)]
            plt.scatter(t_indices, cleaned[np.array(t_indices).astype(int)], color='blue', label='T-waves', marker='v')
            
        # Plot P-peaks
        if 'ECG_P_Peaks' in waves_peak and waves_peak['ECG_P_Peaks']:
            p_indices = [idx for idx in waves_peak['ECG_P_Peaks'] if not np.isnan(idx)]
            plt.scatter(p_indices, cleaned[np.array(p_indices).astype(int)], color='green', label='P-waves', marker='^')

        plt.title(f"ECG Wave Delineation (Snippet): {Path(csv_path).name}")
        plt.xlabel("Samples")
        plt.ylabel("Voltage (mV)")
        plt.legend()
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{Path(csv_path).stem}_delineation.png")
        plt.savefig(out_path, dpi=300)
        print(f"Delineation plot saved to: {out_path}")
        
    except Exception as e:
        print(f"Delineation failed: {e}")
        # Fallback to simple plot
        plt.figure(figsize=(15, 5))
        plt.plot(cleaned)
        plt.title("ECG Cleaned (Delineation failed)")
        plt.savefig(os.path.join(output_dir, f"{Path(csv_path).stem}_cleaned_only.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", default="docs/plots", help="Output directory")
    parser.add_argument("--duration", type=float, default=10, help="Seconds to plot")
    args = parser.parse_args()
    
    render_delineation(args.input, args.output, args.duration)
