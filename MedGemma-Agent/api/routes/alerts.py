from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from db.database import get_db, Alert, AlertLevel
from vitals.schemas import AlertResponse, AlertListItem
from api.middleware.auth import require_any

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get(
    "",
    response_model=list[AlertListItem],
    summary="List alerts",
    description="Returns alerts filtered by patient, severity, and time range.",
)
def list_alerts(
    patient_id: Optional[str] = Query(None),
    severity: Optional[AlertLevel] = Query(None),
    from_dt: Optional[datetime] = Query(None),
    to_dt: Optional[datetime] = Query(None),
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    query = db.query(Alert)
    if patient_id:
        query = query.filter(Alert.patient_id == patient_id)
    if severity:
        query = query.filter(Alert.alert_level == severity)
    if from_dt:
        query = query.filter(Alert.created_at >= from_dt)
    if to_dt:
        query = query.filter(Alert.created_at <= to_dt)
    alerts = query.order_by(Alert.created_at.desc()).limit(limit).all()
    return [
        AlertListItem(
            alert_id=a.id,
            patient_id=a.patient_id,
            alert_level=a.alert_level,
            news2_score=a.news2_score,
            qsofa_score=a.qsofa_score,
            sepsis_risk_flag=a.sepsis_risk_flag,
            recommended_action=a.recommended_action,
            rule_based_only=a.rule_based_only,
            created_at=a.created_at,
        )
        for a in alerts
    ]


@router.get(
    "/{alert_id}",
    response_model=AlertResponse,
    summary="Get full alert detail",
    description="Returns the full alert including LLM reasoning, rule scores, and EHR contributing factors.",
)
def get_alert(
    alert_id: str,
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")
    # Phase 4: full AlertResponse assembly here
    raise HTTPException(status_code=501, detail="Full alert response not yet implemented (Phase 4).")
