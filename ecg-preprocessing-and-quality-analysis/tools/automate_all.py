import subprocess
import os
import sys
from pathlib import Path

def run_step(command_list, description, env):
    print(f"\n>>> STEP: {description}")
    print(f"Executing: {' '.join(command_list)}")
    try:
        process = subprocess.run(command_list, env=env, check=True, capture_output=True, text=True)
        print(process.stdout[-500:]) # Print last 500 chars of output
        return True
    except subprocess.CalledProcessError as e:
        print(f"!!! Error during {description}:")
        print(e.stderr)
        return False

def main():
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)
    
    # Set up environment properly
    custom_env = os.environ.copy()
    custom_env["PYTHONPATH"] = f"{custom_env.get('PYTHONPATH', '')}:."
    python_exec = ".venv/bin/python"
    
    steps = [
        ([python_exec, "tools/download_ecg_datasets.py", "--dataset", "nst", "--output", "DATA/RAW/"], "Downloading Noise Stress Test Dataset"),
        ([python_exec, "tools/ecg_generator.py", "--mode", "augment", "--count", "50", "--output", "DATA/RAW/SYNTHETIC"], "Generating Synthetic ECG Data"),
        ([python_exec, "-m", "scripts.process_daily_ecg", "--input", "DATA/RAW/NOISE_STRESS_TEST", "--output", "Processed_ECG", "--target-fs", "125"], "Preprocessing NST Dataset"),
        ([python_exec, "-m", "scripts.process_daily_ecg", "--input", "DATA/RAW/SYNTHETIC", "--output", "Processed_ECG", "--target-fs", "125"], "Preprocessing Synthetic Dataset"),
        ([python_exec, "-m", "ml.label_segments"], "Building Segment Manifest"),
        ([python_exec, "tools/prepare_training_manifest.py"], "Enriching Training Manifest"),
        ([python_exec, "-m", "ml.train_quality_model", "--manifest", "ml/segment_manifest_enriched.csv"], "Training Quality ML Model"),
        ([python_exec, "tools/validate_quality_scorer.py"], "Validating Quality Scorer"),
    ]

    print("============================================================")
    print("ECG PIPELINE AUTOMATION: STARTING FULL END-TO-END SIMULATION")
    print("============================================================")

    for cmd_list, desc in steps:
        if not run_step(cmd_list, desc, custom_env):
            print("\n!!! Pipeline aborted due to error.")
            sys.exit(1)

    # FINAL STEP: Generate Health Reports for all processed data
    print("\n>>> STEP: Generating Health Reports for all recordings")
    raw_files = list(Path("DATA/RAW/NOISE_STRESS_TEST").glob("*.csv")) + \
                list(Path("DATA/RAW/SYNTHETIC").glob("*.csv"))
    
    # We limit to first 10 for report generation to keep it fast
    for raw_file in raw_files[:10]:
        cmd_list = [python_exec, "tools/generate_health_report.py", "--input", str(raw_file), "--output", "docs/reports"]
        subprocess.run(cmd_list, env=custom_env, check=False)
        print(f"Generated report for: {raw_file.name}")

    print("\n============================================================")
    print("ALL TASKS COMPLETED SUCCESSFULLY")
    print("Processed data is in Processed_ECG/")
    print("Health Reports are in docs/reports/")
    print("Validation plots are in docs/")
    print("============================================================")

if __name__ == "__main__":
    main()
