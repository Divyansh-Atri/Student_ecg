import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from tools.medgemma_bridge import build_clinical_context

context = build_clinical_context("DATA/RAW/NOISE_STRESS_TEST/118e_6_nst_ecg.csv")
with open("ecg_prediction.json", "w") as f:
    json.dump(context, f, indent=4)
print("Prediction JSON generated at: ecg-preprocessing-and-quality-analysis/ecg_prediction.json")
