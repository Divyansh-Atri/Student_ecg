"""MedGemma Bridge: Connects ECG analysis pipeline to MedGemma Clinical LLM.

This script takes the results from our signal processing and ML models,
formats them into a clinical context, and queries MedGemma for an automated
clinical assessment.
"""

import argparse
import json
import os
import requests
from pathlib import Path

# Try to import our ML modules
try:
    from ml.predict_quality import predict_file_quality
    from ml.predict_arrhythmia import predict_file_arrhythmia
except ImportError:
    print("Warning: ML modules not found. Ensure you are running from the project root.")
    predict_file_quality = None
    predict_file_arrhythmia = None

def build_clinical_context(input_csv):
    """Aggregates metrics from various models into a single context object."""
    import pandas as pd
    import numpy as np

    input_path = Path(input_csv)
    cleaned_csv = input_csv
    normalized_csv = input_csv
    temp_files = []

    try:
        df = pd.read_csv(input_path)
        col_clean = "clean_ecg" if "clean_ecg" in df.columns else ("ecg_value" if "ecg_value" in df.columns else None)
        has_clean = col_clean is not None
        has_normalized = "ecg_z" in df.columns

        if has_clean and has_normalized:
            cleaned_csv = input_csv
            normalized_csv = input_csv
        else:
            # Attempt to resolve sister files or generate on the fly
            if "cleaned" in input_path.name or has_clean:
                # Look for normalized sister
                norm_name = input_path.name.replace("cleaned_ecg", "normalized_ecg").replace("clean", "normalized")
                candidate = input_path.parent / norm_name
                if candidate.exists() and candidate.resolve() != input_path.resolve():
                    normalized_csv = str(candidate)
                else:
                    # Dynamically generate normalized version
                    val_col = col_clean if col_clean else df.columns[1]
                    df["ecg_z"] = (df[val_col] - df[val_col].mean()) / (df[val_col].std() + 1e-9)
                    tmp_path = input_path.parent / f"temp_normalized_{input_path.name}"
                    df[["timestamp_ms", "ecg_z"]].to_csv(tmp_path, index=False)
                    normalized_csv = str(tmp_path)
                    temp_files.append(tmp_path)
            elif "normalized" in input_path.name or has_normalized:
                # Look for cleaned sister
                clean_name = input_path.name.replace("normalized_ecg", "cleaned_ecg").replace("normalized", "clean")
                candidate = input_path.parent / clean_name
                if candidate.exists() and candidate.resolve() != input_path.resolve():
                    cleaned_csv = str(candidate)
                else:
                    # Dynamically generate cleaned version
                    df["clean_ecg"] = df["ecg_z"]
                    tmp_path = input_path.parent / f"temp_cleaned_{input_path.name}"
                    df[["timestamp_ms", "clean_ecg"]].to_csv(tmp_path, index=False)
                    cleaned_csv = str(tmp_path)
                    temp_files.append(tmp_path)

        # 1. Get Quality Assessment
        if predict_file_quality:
            q_res = predict_file_quality(cleaned_csv)
            quality_label = q_res.get('file_label', 'UNKNOWN')
            quality_score = q_res.get('file_score', 0.0)
        else:
            quality_label, quality_score = "N/A", 0.0

        # 2. Get Arrhythmia Assessment
        if predict_file_arrhythmia:
            a_res = predict_file_arrhythmia(normalized_csv)
            arr_prob = a_res.get('probability_arrhythmia', 0.0)
            burden = a_res.get('arrhythmia_burden', 0.0)
            flag = a_res.get('overall_flag', False)
        else:
            arr_prob, burden, flag = 0.0, 0.0, False

    finally:
        # Clean up any temporary files generated on the fly
        for temp_file in temp_files:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass

    # 3. Construct Context for MedGemma
    context = {
        "modality": "ECG",
        "metadata": {
            "source_file": input_path.name,
            "pipeline_version": "1.0.0"
        },
        "quality_metrics": {
            "label": quality_label,
            "score_pct": quality_score
        },
        "clinical_features": {
            "arrhythmia_detected": flag,
            "probability": arr_prob,
            "burden_pct": burden * 100
        },
        "prompt_override": (
            f"The patient's ECG signal has {quality_label} quality. "
            f"The analysis detected an arrhythmia burden of {burden*100:.1f}%. "
            "Please provide a detailed clinical interpretation."
        )
    }
    
    return context

def main():
    parser = argparse.ArgumentParser(description="MedGemma ECG Analysis Bridge")
    parser.add_argument("--input", required=True, help="Path to cleaned ECG CSV")
    parser.add_argument("--api_url", default="http://localhost:8000/api/v1/ecg/analyze", help="MedGemma Agent API URL")
    parser.add_argument("--api_key", help="MedGemma API Key (can also set MEDGEMMA_API_KEY env var)")
    parser.add_argument("--patient_id", default="ECG_TEST_001", help="Patient ID for the record")
    parser.add_argument("--output", help="Path to save the MedGemma response")
    
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("MEDGEMMA_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}

    print(f"[*] Extracting clinical context from: {args.input}")
    context = build_clinical_context(args.input)
    context["patient_id"] = args.patient_id
    
    print(f"[*] Sending context to MedGemma API at {args.api_url}...")
    try:
        resp = requests.post(args.api_url, json=context, headers=headers)
        resp.raise_for_status()
        response = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"Error calling MedGemma API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Server replied: {e.response.text}")
        return
    
    print("\n--- MedGemma Clinical Assessment ---")
    print(json.dumps(response, indent=2))
    print("------------------------------------\n")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(response, f, indent=2)
        print(f"[*] Report saved to: {args.output}")

if __name__ == "__main__":
    main()
