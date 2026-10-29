import uuid
from datetime import datetime
from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    Boolean, DateTime, Text, ForeignKey, Enum as SAEnum
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import enum

from config.settings import get_settings

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Enums ─────────────────────────────────────────────────────────────────────

class AlertLevel(str, enum.Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    NORMAL = "NORMAL"
    REJECTED = "REJECTED"  # input guardrail rejection


class RecommendedAction(str, enum.Enum):
    IMMEDIATE_RESPONSE = "immediate_response"
    URGENT_REVIEW = "urgent_review"
    CLINICIAN_REVIEW = "clinician_review"
    OBSERVATION = "observation"


class AuditEventType(str, enum.Enum):
    VITALS_SNAPSHOT_RECEIVED = "VITALS_SNAPSHOT_RECEIVED"
    ALERT_RAISED = "ALERT_RAISED"
    EHR_INGESTED = "EHR_INGESTED"
    EHR_QUERIED = "EHR_QUERIED"
    PATIENT_ACCESSED = "PATIENT_ACCESSED"
    AUDIT_LOG_ACCESSED = "AUDIT_LOG_ACCESSED"
    INPUT_REJECTED = "INPUT_REJECTED"
    LLM_FALLBACK_TRIGGERED = "LLM_FALLBACK_TRIGGERED"
    BREAK_GLASS_ACCESS = "BREAK_GLASS_ACCESS"


# ── Models ────────────────────────────────────────────────────────────────────

class Patient(Base):
    __tablename__ = "patients"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, unique=True, nullable=False, index=True)
    # PHI fields stored encrypted — plain values never persisted
    name_encrypted = Column(Text, nullable=True)
    age = Column(Integer, nullable=True)
    gender = Column(String(16), nullable=True)
    ehr_ingested = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    snapshots = relationship("VitalsSnapshot", back_populates="patient", cascade="all, delete-orphan")
    alerts = relationship("Alert", back_populates="patient", cascade="all, delete-orphan")


class VitalsSnapshot(Base):
    __tablename__ = "vitals_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False, index=True)
    systolic_bp = Column(Float, nullable=False)
    diastolic_bp = Column(Float, nullable=False)
    heart_rate = Column(Float, nullable=False)
    spo2 = Column(Float, nullable=False)
    snapshot_timestamp = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="snapshots")
    alert = relationship("Alert", back_populates="snapshot", uselist=False)


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, ForeignKey("patients.patient_id"), nullable=False, index=True)
    snapshot_id = Column(String, ForeignKey("vitals_snapshots.id"), nullable=False)

    alert_level = Column(SAEnum(AlertLevel), nullable=False)
    news2_score = Column(Integer, nullable=False)
    qsofa_score = Column(Integer, nullable=False)
    sepsis_risk_flag = Column(Boolean, default=False)

    # LLM output — nullable in case of fallback
    llm_reasoning = Column(Text, nullable=True)
    llm_confidence = Column(Float, nullable=True)
    llm_contributing_factors = Column(Text, nullable=True)  # JSON array as string
    llm_limitations = Column(Text, nullable=True)

    recommended_action = Column(SAEnum(RecommendedAction), nullable=False)
    rule_based_only = Column(Boolean, default=False)  # True when LLM was unavailable

    disclaimer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("Patient", back_populates="alerts")
    snapshot = relationship("VitalsSnapshot", back_populates="alert")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type = Column(SAEnum(AuditEventType), nullable=False)
    patient_id = Column(String, nullable=True, index=True)  # nullable for non-patient events
    actor_role = Column(String, nullable=False)   # admin / clinician / readonly / system
    actor_key_hint = Column(String, nullable=True)  # last 4 chars of API key
    endpoint = Column(String, nullable=True)
    event_data = Column(Text, nullable=True)       # JSON string
    prev_hash = Column(String(64), nullable=True)  # SHA-256 of previous log entry
    entry_hash = Column(String(64), nullable=False)  # SHA-256 of this entry
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class ABGResult(Base):
    """Persisted ABG interpretation result — separate from vitals pipeline."""
    __tablename__ = "abg_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, nullable=False, index=True)

    # Input values (stored for audit trail)
    ph = Column(Float, nullable=False)
    pao2_mmhg = Column(Float, nullable=False)
    paco2_mmhg = Column(Float, nullable=False)
    hco3 = Column(Float, nullable=False)
    base_excess = Column(Float, nullable=False)
    lactate = Column(Float, nullable=False)
    fio2 = Column(Float, nullable=False)

    # Rule-based results
    alert_level = Column(String(16), nullable=False)
    primary_disorder = Column(String(64), nullable=False)
    summary = Column(Text, nullable=False)
    clinical_alerts = Column(Text, nullable=True)    # JSON array as string
    respiratory_json = Column(Text, nullable=True)   # JSON
    acid_base_json = Column(Text, nullable=True)     # JSON
    compensation_json = Column(Text, nullable=True)  # JSON
    metabolic_json = Column(Text, nullable=True)     # JSON

    # LLM output
    llm_clinical_significance = Column(Text, nullable=True)
    llm_primary_concern = Column(Text, nullable=True)
    llm_contributing_factors = Column(Text, nullable=True)  # JSON array as string
    llm_confidence = Column(Float, nullable=True)
    llm_limitations = Column(Text, nullable=True)
    rule_based_only = Column(Boolean, default=False)

    disclaimer = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ECGResult(Base):
    """Persisted ECG interpretation result."""
    __tablename__ = "ecg_results"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id = Column(String, nullable=False, index=True)

    source_file = Column(String, nullable=False)
    quality_label = Column(String, nullable=False)
    quality_score = Column(Float, nullable=False)
    arrhythmia_detected = Column(Boolean, nullable=False)
    arrhythmia_probability = Column(Float, nullable=False)
    arrhythmia_burden = Column(Float, nullable=False)

    # LLM output
    llm_clinical_summary = Column(Text, nullable=True)
    llm_recommendation = Column(Text, nullable=True)
    llm_confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)
