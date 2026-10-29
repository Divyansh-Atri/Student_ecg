from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db, Patient, Alert, VitalsSnapshot
from vitals.schemas import PatientSummary
from api.middleware.auth import require_any

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get(
    "/{patient_id}",
    response_model=PatientSummary,
    summary="Get patient summary",
    description="Returns EHR ingestion status and analysis history summary for a patient.",
)
def get_patient(
    patient_id: str,
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found.")

    total_snapshots = db.query(VitalsSnapshot).filter(VitalsSnapshot.patient_id == patient_id).count()
    alerts = (
        db.query(Alert)
        .filter(Alert.patient_id == patient_id)
        .order_by(Alert.created_at.desc())
        .all()
    )
    last_snapshot = (
        db.query(VitalsSnapshot)
        .filter(VitalsSnapshot.patient_id == patient_id)
        .order_by(VitalsSnapshot.snapshot_timestamp.desc())
        .first()
    )

    return PatientSummary(
        patient_id=patient.patient_id,
        age=patient.age,
        gender=patient.gender,
        ehr_ingested=patient.ehr_ingested,
        total_snapshots=total_snapshots,
        total_alerts=len(alerts),
        last_alert_level=alerts[0].alert_level if alerts else None,
        last_snapshot_at=last_snapshot.snapshot_timestamp if last_snapshot else None,
    )
