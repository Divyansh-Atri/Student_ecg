"""
download_ecg_datasets.py

Downloads PTB-XL, MIT-BIH Arrhythmia, MIT-BIH NST (noise stress test),
and CinC 2017 datasets from PhysioNet and converts them to the two-column
CSV format your pipeline expects: timestamp_ms, ecg_value

Run:
    pip install wfdb requests tqdm pandas numpy
    python load_dataset/download_ecg_datasets.py --dataset ptbxl --output DATA/RAW/PTBXL
    python load_dataset/download_ecg_datasets.py --dataset mitbih --output DATA/RAW/MITBIH
    python load_dataset/download_ecg_datasets.py --dataset nst --output DATA/RAW/NOISE_TEST
    python load_dataset/download_ecg_datasets.py --dataset cinc2017 --output DATA/RAW/CINC2017
    python load_dataset/download_ecg_datasets.py --dataset all --output DATA/RAW/

What this does:
  1. Downloads each dataset from PhysioNet using the WFDB Python library
  2. Reads every record (signal + header)
  3. Extracts lead II (or lead I as fallback) — single channel like your wearable
  4. Converts to timestamp_ms (starting at 0) + ecg_value
  5. Saves as: <output>/<record_id>_ecg.csv
  6. Also saves a labels.csv mapping each file to its diagnosis/rhythm label
"""

import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── helpers ──────────────────────────────────────────────────────────────────

def signal_to_csv(signal: np.ndarray, fs: float, out_path: Path) -> None:
    """Convert a 1D signal array to timestamp_ms,ecg_value CSV."""
    n = len(signal)
    dt_ms = 1000.0 / fs
    timestamps = np.arange(n) * dt_ms
    df = pd.DataFrame({
        "timestamp_ms": timestamps.astype(np.int64),
        "ecg_value":    signal.astype(np.float64),
    })
    # BUG FIX: Drop rows where ECG value is NaN (wfdb can return NaN for bad channels)
    nan_count = df["ecg_value"].isna().sum()
    if nan_count == len(df):
        raise ValueError(f"Signal is entirely NaN — channel unusable ({out_path.name})")
    if nan_count > 0:
        df["ecg_value"] = df["ecg_value"].interpolate(method="linear").ffill().bfill()
    df.to_csv(out_path, index=False)


def pick_lead(record) -> np.ndarray:
    """Pick the best single lead from a multi-lead record (prefers lead II)."""
    # BUG FIX: wfdb can return None p_signal if the record has no readable physical signals
    if record.p_signal is None:
        raise ValueError(f"Record '{record.record_name}' has no physical signal (p_signal is None)")
    sig_names = [s.lower() for s in record.sig_name]
    # Try lead II first, then I, then just take channel 0
    for preferred in ["ii", "lead ii", "ecg ii", "i", "lead i", "ecg i"]:
        if preferred in sig_names:
            idx = sig_names.index(preferred)
            return record.p_signal[:, idx]
    return record.p_signal[:, 0]


def download_ptbxl(output_dir: Path, max_records: int = 2000) -> None:
    """
    PTB-XL: 21,837 12-lead ECG records at 100Hz or 500Hz.
    Labels: 71 SCP codes (NORM, MI, STTC, CD, HYP, etc.)
    We use 100Hz records (records100/) to stay close to your 125Hz target.
    
    PhysioNet page: https://physionet.org/content/ptb-xl/1.0.3/
    """
    import wfdb
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading PTB-XL metadata...")
    # Download the database CSV that has all labels
    db_path = output_dir / "ptbxl_database.csv"
    if not db_path.exists():
        import urllib.request
        url = "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
        urllib.request.urlretrieve(url, db_path)

    df_meta = pd.read_csv(db_path)
    # Keep only the first max_records
    df_meta = df_meta.head(max_records).reset_index(drop=True)

    label_rows = []
    print(f"Downloading {len(df_meta)} PTB-XL records (lead II, 100Hz)...")

    for _, row in tqdm(df_meta.iterrows(), total=len(df_meta)):
        record_name = row["filename_lr"]  # 100Hz version
        record_id   = str(row["ecg_id"])
        scp_codes   = str(row.get("scp_codes", "UNKNOWN"))
        patient_id  = str(row.get("patient_id", "unknown"))

        try:
            record = wfdb.rdrecord(
                record_name,
                pn_dir="ptb-xl/1.0.3/",
            )
            signal = pick_lead(record)
            fs = float(record.fs)

            out_path = output_dir / f"{record_id}_ptbxl_ecg.csv"
            signal_to_csv(signal, fs, out_path)

            label_rows.append({
                "filename":   out_path.name,
                "record_id":  record_id,
                "patient_id": patient_id,
                "fs_hz":      fs,
                "n_samples":  len(signal),
                "scp_codes":  scp_codes,
                "dataset":    "ptbxl",
            })
        except Exception as e:
            print(f"  Skipped {record_id}: {e}")

    pd.DataFrame(label_rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"PTB-XL done. {len(label_rows)} records saved to {output_dir}")


# ── MIT-BIH Arrhythmia ───────────────────────────────────────────────────────

def download_mitbih(output_dir: Path) -> None:
    """
    MIT-BIH Arrhythmia Database: 48 records at 360Hz, ~30 minutes each.
    Beat-by-beat arrhythmia annotations (N, V, A, F, etc.)
    
    PhysioNet: https://physionet.org/content/mitdb/1.0.0/
    """
    import wfdb
    output_dir.mkdir(parents=True, exist_ok=True)

    # All 48 record names in MIT-BIH
    record_names = [
        "100","101","102","103","104","105","106","107","108","109",
        "111","112","113","114","115","116","117","118","119","121",
        "122","123","124","200","201","202","203","205","207","208",
        "209","210","212","213","214","215","217","219","220","221",
        "222","223","228","230","231","232","233","234",
    ]

    label_rows = []
    print(f"Downloading {len(record_names)} MIT-BIH records...")

    for rec_name in tqdm(record_names):
        try:
            record = wfdb.rdrecord(rec_name, pn_dir="mitdb/1.0.0/")
            ann    = wfdb.rdann(rec_name, "atr", pn_dir="mitdb/1.0.0/")

            signal = record.p_signal[:, 0]  # MLII lead
            fs     = float(record.fs)

            out_path = output_dir / f"{rec_name}_mitbih_ecg.csv"
            signal_to_csv(signal, fs, out_path)

            # Count beat types
            symbols = ann.symbol
            beat_counts = pd.Series(symbols).value_counts().to_dict()

            label_rows.append({
                "filename":    out_path.name,
                "record_id":   rec_name,
                "fs_hz":       fs,
                "n_samples":   len(signal),
                "beat_counts": str(beat_counts),
                "dataset":     "mitbih",
            })
        except Exception as e:
            print(f"  Skipped {rec_name}: {e}")

    pd.DataFrame(label_rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"MIT-BIH done. {len(label_rows)} records saved.")


# ── MIT-BIH Noise Stress Test ─────────────────────────────────────────────────

def download_nst(output_dir: Path) -> None:
    """
    MIT-BIH Noise Stress Test Database: 12 clean ECG records with 3 noise
    channels (baseline wander BW, muscle noise MA, electrode motion EM)
    mixed at 6 different SNR levels.
    
    Perfect for validating your quality scorer — you know EXACTLY how noisy
    each recording is because it was constructed that way.
    
    PhysioNet: https://physionet.org/content/nstdb/1.0.0/
    """
    import wfdb
    output_dir.mkdir(parents=True, exist_ok=True)

    record_names = [
        "118e00","118e06","118e12","118e18","118e24","118e_6",
        "119e00","119e06","119e12","119e18","119e24","119e_6",
    ]

    label_rows = []
    print(f"Downloading {len(record_names)} MIT-BIH Noise Stress Test records...")

    for rec_name in tqdm(record_names):
        try:
            record = wfdb.rdrecord(rec_name, pn_dir="nstdb/1.0.0/")
            signal = record.p_signal[:, 0]
            fs     = float(record.fs)

            # Parse SNR from name: e00=0dB, e06=6dB, e12=12dB etc.
            snr_label = rec_name.split("e")[-1].replace("_", "-")

            out_path = output_dir / f"{rec_name}_nst_ecg.csv"
            signal_to_csv(signal, fs, out_path)

            label_rows.append({
                "filename":  out_path.name,
                "record_id": rec_name,
                "fs_hz":     fs,
                "n_samples": len(signal),
                "snr_db":    snr_label,
                "dataset":   "nst",
                "note":      "Higher dB = cleaner signal",
            })
        except Exception as e:
            print(f"  Skipped {rec_name}: {e}")

    pd.DataFrame(label_rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"NST done. {len(label_rows)} records saved.")


# ── CinC 2017 (AF Detection + Noisy label) ────────────────────────────────────

def download_cinc2017(output_dir: Path) -> None:
    """
    CinC Challenge 2017: 8,528 short single-lead ECG recordings (9-60 seconds).
    Labels: N=Normal, A=AF, O=Other, ~=Noisy
    
    The '~' (Noisy) label is extremely valuable for training your quality scorer —
    these are real recordings that cardiologists labeled as too noisy to classify.
    
    PhysioNet: https://physionet.org/content/challenge-2017/1.0.0/
    """
    import wfdb
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching CinC 2017 record list...")

    # Download the reference label file
    import urllib.request
    label_url = "https://physionet.org/files/challenge-2017/1.0.0/REFERENCE-v3.csv"
    label_file = output_dir / "REFERENCE-v3.csv"
    if not label_file.exists():
        urllib.request.urlretrieve(label_url, label_file)

    ref = pd.read_csv(label_file, header=None, names=["record_id","label"])
    print(f"  {len(ref)} records: {ref['label'].value_counts().to_dict()}")

    label_rows = []
    for _, row in tqdm(ref.iterrows(), total=len(ref)):
        # BUG FIX: strip whitespace — REFERENCE-v3.csv sometimes has trailing spaces
        rec_name = str(row["record_id"]).strip()
        label    = str(row["label"]).strip()
        try:
            record = wfdb.rdrecord(
                rec_name,
                pn_dir="challenge-2017/1.0.0/training2017/",
            )
            signal = record.p_signal[:, 0]
            fs     = float(record.fs)

            out_path = output_dir / f"{rec_name}_cinc2017_ecg.csv"
            signal_to_csv(signal, fs, out_path)

            label_rows.append({
                "filename":   out_path.name,
                "record_id":  rec_name,
                "fs_hz":      fs,
                "n_samples":  len(signal),
                "label":      label,
                # Map to human-readable
                "label_text": {"N":"Normal","A":"AF","O":"Other","~":"Noisy"}.get(label, label),
                "dataset":    "cinc2017",
            })
        except Exception as e:
            print(f"  Skipped {rec_name}: {e}")

    pd.DataFrame(label_rows).to_csv(output_dir / "labels.csv", index=False)
    print(f"CinC 2017 done. {len(label_rows)} records saved.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ECG datasets and convert to pipeline CSV format")
    parser.add_argument("--dataset", choices=["ptbxl","mitbih","nst","cinc2017","all"], default="all")
    parser.add_argument("--output",  default="DATA/RAW", help="Base output folder")
    parser.add_argument("--max-ptbxl", type=int, default=2000, help="Max PTB-XL records to download (default 2000, max 21837)")
    args = parser.parse_args()

    base = Path(args.output)

    if args.dataset in ("mitbih", "all"):
        download_mitbih(base / "MITBIH_ARRHYTHMIA")

    if args.dataset in ("nst", "all"):
        download_nst(base / "NOISE_STRESS_TEST")

    if args.dataset in ("cinc2017", "all"):
        download_cinc2017(base / "CINC2017")

    if args.dataset in ("ptbxl", "all"):
        download_ptbxl(base / "PTBXL", max_records=args.max_ptbxl)

    print("\nAll done. Now run your pipeline:")
    print("  python -m scripts.process_daily_ecg --input DATA/RAW/PTBXL --output Processed_ECG --target-fs 125 --powerline-hz 50")