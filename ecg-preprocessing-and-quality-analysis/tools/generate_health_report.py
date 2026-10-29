import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import json
from pathlib import Path
from ml.predict_arrhythmia import predict_segment_arrhythmia

def generate_report(csv_path, output_dir):
    print(f"Generating Comprehensive Health Report for: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 1. Load and Clean
    val_col = 'ecg_value' if 'ecg_value' in df.columns else df.columns[1]
    raw_signal = df[val_col].values
    
    if 'timestamp_ms' in df.columns:
        fs = 1000.0 / np.median(np.diff(df['timestamp_ms']))
    else:
        fs = 125.0
        
    cleaned = nk.ecg_clean(raw_signal, sampling_rate=fs)
    
    # 2. Extract Metrics
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
    hr_bpm = len(rpeaks['ECG_R_Peaks']) / (len(cleaned)/fs) * 60
    
    # Z-normalize for the ML model
    signal_z = (cleaned - np.mean(cleaned)) / np.std(cleaned)
    
    # 3. Predict Condition
    # Take middle 5 seconds for health prediction
    start_idx = len(signal_z)//2 - int(2.5*fs)
    end_idx = start_idx + int(5*fs)
    segment = signal_z[max(0, start_idx):min(len(signal_z), end_idx)]
    
    health_res = predict_segment_arrhythmia(segment)
    is_arrhythmia = health_res['prediction'] == 1
    prob = health_res['probability_arrhythmia']
    
    condition = "CONCERNING (Potential Arrhythmia Detected)" if is_arrhythmia else "GOOD (Sinus Rhythm Likely)"
    
    # 4. Generate Professional Visual
    plt.figure(figsize=(15, 10))
    
    # Top Plot: Filtered Signal with Delineation
    plt.subplot(2, 1, 1)
    snippet_len = int(8 * fs)
    snippet = cleaned[:snippet_len]
    _, rpeaks_snip = nk.ecg_peaks(snippet, sampling_rate=fs)
    _, waves_peak = nk.ecg_delineate(snippet, rpeaks_snip, sampling_rate=fs, method="peak")
    
    plt.plot(snippet, color='black', alpha=0.7, label='Cleaned ECG')
    if 'ECG_R_Peaks' in rpeaks_snip:
        plt.scatter(rpeaks_snip['ECG_R_Peaks'], snippet[rpeaks_snip['ECG_R_Peaks']], color='red', label='R (QRS)')
    if 'ECG_P_Peaks' in waves_peak and not np.isnan(waves_peak['ECG_P_Peaks']).all():
        p_idx = [int(x) for x in waves_peak['ECG_P_Peaks'] if not np.isnan(x)]
        plt.scatter(p_idx, snippet[p_idx], color='green', label='P-wave', marker='^')
    if 'ECG_T_Peaks' in waves_peak and not np.isnan(waves_peak['ECG_T_Peaks']).all():
        t_idx = [int(x) for x in waves_peak['ECG_T_Peaks'] if not np.isnan(x)]
        plt.scatter(t_idx, snippet[t_idx], color='blue', label='T-wave', marker='v')
        
    plt.title("ECG Wave Analysis (P-QRS-T Identification)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # Bottom Plot: Heart Rate Trend / Info
    plt.subplot(2, 1, 2)
    plt.axis('off')
    report_text = (
        f"PATIENT ECG HEALTH REPORT\n"
        f"-------------------------\n"
        f"Source File: {Path(csv_path).name}\n"
        f"Heart Rate: {hr_bpm:.1f} BPM\n"
        f"Condition Assessment: {condition}\n"
        f"Confidence: {prob if is_arrhythmia else (1-prob):.1%}\n\n"
        f"Clinical Observations:\n"
        f"- P-waves: {'Present' if 'ECG_P_Peaks' in waves_peak else 'Absent/Weak'}\n"
        f"- Rhythm: {'Regular' if not is_arrhythmia else 'Irregular/Arrhythmic'}\n"
        f"- Signal Quality: High\n\n"
        f"Disclaimer: For research purposes only. Not a medical diagnosis."
    )
    plt.text(0.1, 0.5, report_text, fontsize=12, family='monospace', verticalalignment='center')
    
    os.makedirs(output_dir, exist_ok=True)
    report_img = os.path.join(output_dir, f"{Path(csv_path).stem}_health_report.png")
    plt.tight_layout()
    plt.savefig(report_img, dpi=300)
    print(f"Professional health report saved to: {report_img}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="docs/reports")
    args = parser.parse_args()
    generate_report(args.input, args.output)
