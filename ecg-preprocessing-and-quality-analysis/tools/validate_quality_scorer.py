import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from pathlib import Path

def validate():
    manifest_path = "ml/segment_manifest_enriched.csv"
    output_plot = "docs/quality_vs_snr.png"
    
    if not os.path.exists(manifest_path):
        print(f"Error: {manifest_path} not found.")
        return

    df = pd.read_csv(manifest_path)
    
    # Filter to NST dataset
    # In my prepare_training_manifest.py, I used 'other' for things not containing 'nst' in path
    # Let's fix the filtering logic here just in case
    nst_df = df[df['segment_path'].str.contains('NOISE_STRESS_TEST', case=False, na=False)].copy()
    
    if nst_df.empty:
        # Try group_name if segment_path doesn't have it (though Task 3 should have put it in DATA/RAW/NOISE_STRESS_TEST)
        print("No segments found with 'NOISE_STRESS_TEST' in path. Checking source_dataset...")
        nst_df = df[df['source_dataset'] == 'nst'].copy()

    if nst_df.empty:
        print("Error: No NST segments found for validation.")
        return

    print(f"Found {len(nst_df)} NST segments.")

    def extract_snr(path):
        # Filenames/Groups: 1970-01-01_118e, 1970-01-01_118e00, 1970-01-01_118e06
        # SNR is the number after 'e'. 
        # If 'e' is at the end of the patient ID part (e.g. 118e), it's -6dB.
        
        match = re.search(r'11[89]e(\d*)', path)
        if match:
            val = match.group(1)
            if val == '':
                return -6
            return int(val)
        return None

    nst_df['SNR_dB'] = nst_df['segment_path'].apply(extract_snr)
    nst_df = nst_df[nst_df['SNR_dB'].notna()]

    # Group and compute mean quality_score
    summary = nst_df.groupby('SNR_dB')['quality_score'].agg(['mean', 'count']).reset_index()
    summary.rename(columns={'mean': 'mean_quality_score'}, inplace=True)
    summary = summary.sort_values('SNR_dB')

    print("\n--- Validation Results ---")
    print(summary.to_string(index=False))

    # Assert monotonicity (mean_quality_score should increase with SNR_dB)
    scores = summary['mean_quality_score'].tolist()
    is_monotonic = all(x <= y for x, y in zip(scores, scores[1:]))
    
    if is_monotonic:
        print("\nSUCCESS: Quality score increases monotonically with SNR.")
    else:
        print("\nWARNING: Quality score is NOT strictly monotonic with SNR.")
        # Check for small variations
        diffs = [y - x for x, y in zip(scores, scores[1:])]
        print(f"Differences: {diffs}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(summary['SNR_dB'].astype(str), summary['mean_quality_score'], color='skyblue', edgecolor='navy')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Mean Quality Score (0-100)')
    plt.title('Quality Scorer Validation: Quality vs Known Noise (NST)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot, dpi=300)
    print(f"\nValidation plot saved to: {output_plot}")

if __name__ == "__main__":
    validate()
