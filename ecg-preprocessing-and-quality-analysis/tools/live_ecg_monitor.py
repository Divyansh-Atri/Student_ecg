import pandas as pd
import numpy as np
import time
import sys
import argparse
import neurokit2 as nk
import os
from ml.predict_arrhythmia import predict_segment_arrhythmia

def live_monitor(csv_path, chunk_sec=5):
    # Robust file path handling
    if not csv_path.endswith(".csv"):
        csv_path += ".csv"
        
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        print("Please ensure the file exists in DATA/RAW/SYNTHETIC/ or other data directories.")
        return

    print("Initializing Live ECG Monitoring Simulation...")
    df = pd.read_csv(csv_path)
    val_col = 'ecg_value' if 'ecg_value' in df.columns else df.columns[1]
    raw_signal = df[val_col].values
    
    fs = 125.0
    samples_per_chunk = int(chunk_sec * fs)
    
    print(f"Streaming {csv_path} @ {fs}Hz...")
    print("-" * 60)
    print(f"{'Time':<10} | {'HR (BPM)':<10} | {'Status':<20} | {'Alert'}")
    print("-" * 60)

    for i in range(0, len(raw_signal), samples_per_chunk):
        chunk = raw_signal[i:i+samples_per_chunk]
        if len(chunk) < samples_per_chunk:
            break
            
        # 1. Clean
        try:
            cleaned = nk.ecg_clean(chunk, sampling_rate=fs)
            
            # 2. Heart Rate
            _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=fs)
            hr = len(rpeaks['ECG_R_Peaks']) / chunk_sec * 60
            
            # 3. Predict Condition
            signal_z = (cleaned - np.mean(cleaned)) / (np.std(cleaned) + 1e-9)
            res = predict_segment_arrhythmia(signal_z)
            
            status = "NORMAL" if res['prediction'] == 0 else "ARRHYTHMIA"
            alert = "!!!" if res['prediction'] == 1 or hr > 120 or hr < 50 else ""
        except Exception as e:
            hr = 0
            status = "SIGNAL ERROR"
            alert = "CHECK PROBE"
        
        timestamp = f"{i/fs:.1f}s"
        print(f"{timestamp:<10} | {hr:<10.1f} | {status:<20} | {alert}")
        
        # Simulate real-time delay
        time.sleep(0.5) 

    print("-" * 60)
    print("Monitoring Session Ended.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    live_monitor(args.input)
