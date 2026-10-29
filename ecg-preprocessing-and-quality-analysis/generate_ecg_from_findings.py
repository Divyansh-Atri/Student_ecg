#!/usr/bin/env python3
"""
generate_ecg_from_findings.py

A robust CLI tool to generate a clinically identical ECG waveform from 
machine-generated clinical findings text.

It uses the MedGemma LLM via Ollama to intelligently parse the text into structured 
synthesis parameters, falling back to deterministic heuristic rules if the LLM 
is unavailable. The resulting parameters are then fed into the core DSP 
engine (ecg_generator.py) to synthesize the waveform.
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error
from pathlib import Path

# Add local path so we can import the synthesis engine
sys.path.append(str(Path(__file__).parent / "ecg_generator"))
from ecg_generator import synthesize_pqrst_ecg, render_ecg_from_csv
import pandas as pd

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "medgemma"

def parse_findings_with_heuristics(findings_text: str) -> dict:
    """
    Fallback deterministic parser using regex rules to extract clinical
    parameters if the LLM is down.
    """
    text = findings_text.lower()
    
    # Defaults
    params = {
        "hr_bpm": 72.0,
        "pathology": "normal",
        "noise_level": 0.03,
        "baseline_drift": 0.05
    }
    
    # 1. Pathology mapping
    if "atrial fibrillation" in text or "af" in text or "arrhythmia" in text:
        params["pathology"] = "af"
        if "burden" in text:
            # extract burden % to adjust noise/variability if desired
            pass
    elif "left bundle branch block" in text or "lbbb" in text:
        params["pathology"] = "lbbb"
    elif "st elevation" in text or "ischemia" in text:
        params["pathology"] = "st_elevation"
    elif "flat t" in text or "hypokalemia" in text:
        params["pathology"] = "flat_t"
        
    # 2. Heart rate extraction
    hr_match = re.search(r'hr\s*(?:of|is|≈|=)?\s*(\d+)', text)
    if not hr_match:
        hr_match = re.search(r'rate\s*(?:of|is|≈|=)?\s*(\d+)', text)
    if hr_match:
        params["hr_bpm"] = float(hr_match.group(1))
        
    # 3. Quality / Noise extraction
    if "poor quality" in text or "noisy" in text:
        params["noise_level"] = 0.15
        params["baseline_drift"] = 0.20
    elif "excellent" in text or "clean" in text:
        params["noise_level"] = 0.01
        params["baseline_drift"] = 0.02
        
    return params

def parse_findings_with_llm(findings_text: str) -> dict:
    """
    Query the local MedGemma LLM to extract structured features.
    """
    system_prompt = """You are an expert cardiology parameter extraction system.
Analyze the clinical findings text and output a valid JSON object matching this schema EXACTLY, with no markdown formatting or extra text:
{
  "hr_bpm": <float, default 72.0>,
  "pathology": <string: one of "normal", "af", "lbbb", "st_elevation", "flat_t">,
  "noise_level": <float, default 0.03, higher if poor quality>,
  "baseline_drift": <float, default 0.05>
}
Extract the parameters that best match the findings."""

    payload = {
        "model": MODEL_NAME,
        "prompt": f"FINDINGS:\n{findings_text}\n\nExtract JSON:",
        "system": system_prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            extracted_json_str = result.get("response", "{}")
            params = json.loads(extracted_json_str)
            # Ensure safe defaults if LLM misses a field
            return {
                "hr_bpm": float(params.get("hr_bpm", 72.0)),
                "pathology": str(params.get("pathology", "normal")),
                "noise_level": float(params.get("noise_level", 0.03)),
                "baseline_drift": float(params.get("baseline_drift", 0.05))
            }
    except Exception as e:
        print(f"[!] LLM parsing failed ({e}). Falling back to heuristics...")
        return parse_findings_with_heuristics(findings_text)

def generate_ecg(findings_text: str, output_dir: str, duration_s: float = 10.0, render: bool = True):
    print("--- Clinical Findings Analysis ---")
    print(findings_text)
    print("----------------------------------\n")
    
    print("1. Parsing findings into synthesis parameters...")
    params = parse_findings_with_llm(findings_text)
    print("Extracted Parameters:")
    print(json.dumps(params, indent=2))
    
    print("\n2. Synthesizing ECG waveform...")
    ts, sig = synthesize_pqrst_ecg(
        duration_s=duration_s,
        fs=125.0,
        hr_bpm=params["hr_bpm"],
        noise_level=params["noise_level"],
        baseline_drift=params["baseline_drift"],
        pathology=params["pathology"]
    )
    
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = out_dir_path / "generated_ecg.csv"
    pd.DataFrame({"timestamp_ms": ts, "ecg_value": sig}).to_csv(csv_path, index=False)
    print(f"[*] Saved raw waveform to: {csv_path}")
    
    if render:
        png_path = out_dir_path / "generated_ecg_strip.png"
        render_ecg_from_csv(
            str(csv_path),
            output_path=str(png_path),
            fs=125.0,
            duration_s=min(duration_s, 10.0),
            title=f"Synthetic ECG - Pathology: {params['pathology'].upper()}"
        )
        print(f"[*] Rendered clinical strip to: {png_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ECG from Clinical Findings")
    parser.add_argument("--findings", type=str, help="Raw text of the machine-generated findings", required=False)
    parser.add_argument("--findings_file", type=str, help="Path to text file containing findings", required=False)
    parser.add_argument("--output", type=str, default="Processed_ECG", help="Output directory")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of generated ECG in seconds")
    parser.add_argument("--no_render", action="store_true", help="Do not render the PNG image")
    
    args = parser.parse_args()
    
    if args.findings_file:
        with open(args.findings_file, "r") as f:
            text = f.read()
    elif args.findings:
        text = args.findings
    else:
        text = "The patient's ECG signal has POOR quality. The analysis detected an arrhythmia burden of 94.7%. Please provide a detailed clinical interpretation."
        print("[!] No findings provided, using default fallback test string.\n")
        
    generate_ecg(text, args.output, args.duration, not args.no_render)
