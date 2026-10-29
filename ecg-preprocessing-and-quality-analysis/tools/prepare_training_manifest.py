import pandas as pd
import json
import os
from pathlib import Path

def prepare_manifest():
    project_root = Path(".")
    manifest_path = project_root / "ml/segment_manifest.csv"
    output_path = project_root / "ml/segment_manifest_enriched.csv"

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found.")
        return

    print(f"Loading manifest from {manifest_path}...")
    df = pd.read_csv(manifest_path)
    
    enriched_rows = []
    
    print("Enriching segments with metadata...")
    for idx, row in df.iterrows():
        segment_path_str = row['segment_path']
        
        # IMPROVED source dataset detection
        low_path = segment_path_str.lower()
        if 'nst' in low_path or 'noise_stress_test' in low_path or '118e' in low_path or '119e' in low_path:
            source = 'nst'
        elif 'synthetic' in low_path or 'synth' in low_path:
            source = 'synthetic'
        elif 'mitbih' in low_path:
            source = 'mitbih'
        elif 'cinc' in low_path:
            source = 'cinc2017'
        elif 'ptbxl' in low_path:
            source = 'ptbxl'
        else:
            source = 'other'
            
        new_row = row.to_dict()
        new_row.update({
            'auto_quality_label': row.get('quality_label', 'UNKNOWN'),
            'auto_quality_score': row.get('quality_score', 0.0),
            'source_dataset': source
        })
        enriched_rows.append(new_row)

    enriched_df = pd.DataFrame(enriched_rows)
    
    # Pathologies (from labels.csv in source folders)
    print("Joining pathology labels...")
    source_labels = {}
    raw_dir = project_root / "DATA/RAW"
    if raw_dir.exists():
        for folder in raw_dir.iterdir():
            if folder.is_dir():
                labels_file = folder / "labels.csv"
                if labels_file.exists():
                    try:
                        ldf = pd.read_csv(labels_file)
                        if 'filename' in ldf.columns and 'pathology' in ldf.columns:
                            for _, lrow in ldf.iterrows():
                                source_labels[lrow['filename']] = lrow['pathology']
                    except Exception as e:
                        print(f"Error reading {labels_file}: {e}")

    def map_pathology(row):
        gn = str(row['group_name'])
        # Match NST especially
        if row['source_dataset'] == 'nst':
            # gn looks like 1970-01-01_118e00
            snr_match = gn.split('_')[-1] # e.g. 118e00
            for key in source_labels:
                if snr_match in key:
                    return source_labels[key]
        
        for key in source_labels:
            if gn.split('_')[-1] in key:
                return source_labels[key]
        return 'unknown'

    enriched_df['pathology_label'] = enriched_df.apply(map_pathology, axis=1)

    print(f"Saving enriched manifest to {output_path}...")
    enriched_df.to_csv(output_path, index=False)
    
    print("\n--- Summary ---")
    print(f"Total segments: {len(enriched_df)}")
    print("\nBreakdown by source_dataset:")
    print(enriched_df['source_dataset'].value_counts())
    print("\nBreakdown by auto_quality_label:")
    print(enriched_df['quality_label'].value_counts())

if __name__ == "__main__":
    prepare_manifest()
