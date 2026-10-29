
ECG_SYSTEM_PROMPT = """You are MedGemma, an expert clinical AI assistant specializing in cardiology and ECG interpretation.
You will receive preprocessed ECG metrics, including signal quality and arrhythmia detection results.
Provide a concise clinical summary, a recommendation for next steps, and your confidence level in this assessment.
Always format your response as a valid JSON object matching the requested schema."""

def build_ecg_user_prompt(payload: dict) -> str:
    metadata = payload.get('metadata', {})
    quality = payload.get('quality_metrics', {})
    features = payload.get('clinical_features', {})
    
    return f"""Patient ECG Analysis Results:
- Modality: {payload.get('modality', 'ECG')}
- Patient ID: {payload.get('patient_id', 'UNKNOWN')}
- Source File: {metadata.get('source_file', 'Unknown')}
- Signal Quality: {quality.get('label', 'Unknown')} ({quality.get('score_pct', 0.0):.1f}%)
- Arrhythmia Detected: {features.get('arrhythmia_detected', False)}
- Arrhythmia Probability: {features.get('probability', 0.0):.2f}
- Arrhythmia Burden: {features.get('burden_pct', 0.0):.1f}%

Please analyze these findings and provide a clinical summary, primary recommendation, and your confidence score."""
