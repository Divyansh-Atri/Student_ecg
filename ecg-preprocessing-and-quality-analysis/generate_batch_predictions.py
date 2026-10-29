import json
import glob
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tools.medgemma_bridge import build_clinical_context

# Get the first 5 files from the Noise Stress Test dataset
files = glob.glob("DATA/RAW/NOISE_STRESS_TEST/*_nst_ecg.csv")[:5]

batch_predictions = []
print(f"Processing {len(files)} files for batch prediction...")

for f in files:
    try:
        context = build_clinical_context(f)
        batch_predictions.append(context)
        print(f"Processed: {os.path.basename(f)}")
    except Exception as e:
        print(f"Failed {f}: {e}")

with open("batch_predictions.json", "w") as out_f:
    json.dump(batch_predictions, out_f, indent=4)

print("Batch predictions saved to: ecg-preprocessing-and-quality-analysis/batch_predictions.json")
