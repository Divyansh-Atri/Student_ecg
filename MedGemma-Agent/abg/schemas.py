"""
Pydantic schemas for the ABG module.
Completely independent of vitals/schemas.py.
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class ABGAlertLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    NORMAL   = "NORMAL"


# ── Input ──────────────────────────────────────────────────────────────────────

class ABGInput(BaseModel):
    """Incoming ABG values for interpretation."""

    patient_id: str = Field(..., description="Patient identifier")

    # Core required values
    ph: float = Field(..., alias="pH", description="Arterial pH (7.0–7.8)")
    pao2_mmhg: float = Field(..., description="PaO2 in mmHg (20–700)")
    paco2_mmhg: float = Field(..., description="PaCO2 in mmHg (10–120)")
    hco3: float = Field(..., description="Bicarbonate in mmol/L (5–60)")
    base_excess: float = Field(..., description="Base excess in mmol/L (-30 to +30)")
    lactate: float = Field(..., description="Lactate in mmol/L (0.1–30)")

    # Optional electrolytes (needed for anion gap)
    na: Optional[float] = Field(None, description="Sodium in mmol/L (optional)")
    cl: Optional[float] = Field(None, description="Chloride in mmol/L (optional)")
    albumin: Optional[float] = Field(None, description="Albumin in g/dL (optional, for corrected AG)")
    glucose: Optional[float] = Field(None, description="Glucose in mmol/L (optional)")
    ketones: Optional[bool] = Field(None, description="Ketones present (optional)")

    # Context
    fio2: float = Field(0.21, description="FiO2 as fraction (0.21–1.0)")
    chronic_copd: bool = Field(False, description="Patient has known chronic COPD")
    snapshot_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = {"populate_by_name": True}

    @field_validator("ph")
    @classmethod
    def validate_ph(cls, v: float) -> float:
        if not (6.8 <= v <= 7.8):
            raise ValueError(f"pH {v} is outside plausible range (6.8–7.8)")
        return v

    @field_validator("pao2_mmhg")
    @classmethod
    def validate_pao2(cls, v: float) -> float:
        if not (20 <= v <= 700):
            raise ValueError(f"PaO2 {v} mmHg is outside plausible range (20–700)")
        return v

    @field_validator("paco2_mmhg")
    @classmethod
    def validate_paco2(cls, v: float) -> float:
        if not (10 <= v <= 120):
            raise ValueError(f"PaCO2 {v} mmHg is outside plausible range (10–120)")
        return v

    @field_validator("hco3")
    @classmethod
    def validate_hco3(cls, v: float) -> float:
        if not (5 <= v <= 60):
            raise ValueError(f"HCO3 {v} mmol/L is outside plausible range (5–60)")
        return v

    @field_validator("base_excess")
    @classmethod
    def validate_be(cls, v: float) -> float:
        if not (-30 <= v <= 30):
            raise ValueError(f"Base excess {v} is outside plausible range (-30 to +30)")
        return v

    @field_validator("lactate")
    @classmethod
    def validate_lactate(cls, v: float) -> float:
        if not (0.1 <= v <= 30):
            raise ValueError(f"Lactate {v} mmol/L is outside plausible range (0.1–30)")
        return v

    @field_validator("fio2")
    @classmethod
    def validate_fio2(cls, v: float) -> float:
        if not (0.21 <= v <= 1.0):
            raise ValueError(f"FiO2 {v} must be between 0.21 and 1.0")
        return v


# ── Output sub-models ──────────────────────────────────────────────────────────

class RespiratoryResult(BaseModel):
    hypoxia: bool
    type: str
    pf_ratio: Optional[float] = None
    explanation: List[str]


class AcidBaseResult(BaseModel):
    status: str
    primary_disorder: str
    acute_vs_chronic: Optional[str] = None
    explanation: List[str]


class CompensationResult(BaseModel):
    compensated: Optional[str] = None
    expected_value: Optional[float] = None
    measured_value: Optional[float] = None
    explanation: List[str]


class MetabolicResult(BaseModel):
    anion_gap: Optional[float] = None
    anion_gap_corrected: Optional[float] = None
    gap_type: Optional[str] = None
    delta_delta: Optional[float] = None
    likely_causes: List[str]
    explanation: List[str]


# ── Full response ──────────────────────────────────────────────────────────────

class ABGResponse(BaseModel):
    abg_id: str
    patient_id: str
    alert_level: ABGAlertLevel
    summary: str
    clinical_alerts: List[str]

    respiratory: RespiratoryResult
    acid_base: AcidBaseResult
    compensation: CompensationResult
    metabolic: MetabolicResult

    # MedGemma reasoning (None if LLM unavailable)
    clinical_significance: Optional[str] = None
    primary_concern: Optional[str] = None
    contributing_factors: Optional[List[str]] = None
    llm_confidence: Optional[float] = None
    llm_limitations: Optional[str] = None
    rule_based_only: bool = False

    disclaimer: str = (
        "CLINICAL DECISION SUPPORT ONLY. This output is generated by an AI system "
        "and must not replace the clinical judgment of a qualified healthcare professional. "
        "All ABG interpretations require clinical review before action is taken."
    )
    created_at: datetime


class ABGListItem(BaseModel):
    abg_id: str
    patient_id: str
    alert_level: ABGAlertLevel
    primary_disorder: str
    summary: str
    rule_based_only: bool
    created_at: datetime
