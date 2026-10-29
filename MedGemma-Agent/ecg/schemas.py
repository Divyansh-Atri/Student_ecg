from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ECGMetadata(BaseModel):
    source_file: str
    pipeline_version: str

class ECGQualityMetrics(BaseModel):
    label: str
    score_pct: float

class ECGClinicalFeatures(BaseModel):
    arrhythmia_detected: bool
    probability: float
    burden_pct: float

class ECGAnalyzeRequest(BaseModel):
    patient_id: str = Field(default="UNKNOWN_PATIENT")
    modality: str = "ECG"
    metadata: ECGMetadata
    quality_metrics: ECGQualityMetrics
    clinical_features: ECGClinicalFeatures
    prompt_override: Optional[str] = None

class ECGAnalyzeResponse(BaseModel):
    id: str
    patient_id: str
    clinical_summary: str
    recommendation: str
    confidence: float
    created_at: datetime
