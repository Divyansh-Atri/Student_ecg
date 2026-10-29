from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from db.database import get_db, AuditLog, AuditEventType
from api.middleware.auth import require_admin

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get(
    "",
    summary="Query audit trail",
    description=(
        "Admin-only. Returns the immutable HIPAA-compliant audit log. "
        "Each entry includes a SHA-256 hash chain for tamper detection."
    ),
)
def query_audit_log(
    patient_id: Optional[str] = Query(None),
    event_type: Optional[AuditEventType] = Query(None),
    from_dt: Optional[datetime] = Query(None),
    to_dt: Optional[datetime] = Query(None),
    limit: int = Query(100, le=1000),
    db: Session = Depends(get_db),
    role: str = Depends(require_admin),
):
    query = db.query(AuditLog)
    if patient_id:
        query = query.filter(AuditLog.patient_id == patient_id)
    if event_type:
        query = query.filter(AuditLog.event_type == event_type)
    if from_dt:
        query = query.filter(AuditLog.created_at >= from_dt)
    if to_dt:
        query = query.filter(AuditLog.created_at <= to_dt)
    logs = query.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": log.id,
            "event_type": log.event_type,
            "patient_id": log.patient_id,
            "actor_role": log.actor_role,
            "actor_key_hint": log.actor_key_hint,
            "endpoint": log.endpoint,
            "event_data": log.event_data,
            "entry_hash": log.entry_hash,
            "prev_hash": log.prev_hash,
            "created_at": log.created_at,
        }
        for log in logs
    ]
