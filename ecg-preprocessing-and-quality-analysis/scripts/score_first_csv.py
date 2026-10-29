import glob
import json
import os
import numpy as np

import importlib.util

# load scripts/clean_ecg.py as module to avoid package import issues when
# running this helper script from inside the scripts/ folder.
here = os.path.dirname(__file__)
clean_path = os.path.join(here, "clean_ecg.py")
spec = importlib.util.spec_from_file_location("clean_ecg", clean_path)
clean_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clean_mod)
score_ecg_quality = clean_mod.score_ecg_quality

# find first CSV under DATA/RAW
base = os.path.join(os.path.dirname(__file__), "..")
root = os.path.abspath(os.path.join(base, "DATA", "RAW"))
paths = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
if not paths:
    print("No CSV files found under DATA/RAW")
    raise SystemExit(2)

path = paths[0]
print(f"Using file: {path}")

# read file as text and parse comma-separated timestamp,value sequence
with open(path, "r", encoding="utf-8", errors="ignore") as f:
    txt = f.read()

parts = [p for p in txt.replace('\n', ',').split(',') if p.strip()!='']
vals = []
for p in parts:
    try:
        vals.append(float(p))
    except Exception:
        pass
if len(vals) < 2:
    print("Could not parse numeric values from CSV")
    raise SystemExit(2)

# assume even indices are timestamps (ms) and odd indices are values
timestamps = np.array(vals[0::2], dtype=np.float64)
ecg = np.array(vals[1::2], dtype=np.float64)

# basic sanity
n = min(timestamps.size, ecg.size)
timestamps = timestamps[:n]
ecg = ecg[:n]

# estimate sampling rate from median diff (timestamps in ms)
if n >= 2:
    diffs = np.diff(timestamps)
    # convert ms -> seconds
    median_dt_s = float(np.median(diffs)) / 1000.0 if np.any(np.isfinite(diffs)) else 0.0
    fs = 1.0 / median_dt_s if median_dt_s > 0 else 250.0
else:
    fs = 250.0

print(f"Samples: {n}, estimated fs={fs:.2f} Hz")

res = score_ecg_quality(cleaned=ecg, raw=ecg, sampling_rate_hz=fs, rpeaks=None, return_segment_details=True)

# Print concise summary + aggregation debug
out = {
    'file': os.path.relpath(path),
    'score': res.get('score'),
    'label': res.get('label'),
    'metrics': res.get('metrics'),
}
print(json.dumps(out, indent=2))

if 'debug' in res:
    dbg = res['debug']
    agg = dbg.get('aggregation', {})
    print("\nAggregation debug:")
    print(json.dumps(agg, indent=2))
    print('\nFirst 6 per-segment penalties:')
    for i, s in enumerate(dbg.get('per_segment_severity', [])[:6]):
        print(i, json.dumps(s, indent=2))

print('\nDone')
