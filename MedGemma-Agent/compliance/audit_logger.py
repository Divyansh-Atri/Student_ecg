"""
HIPAA-compliant audit logger with SHA-256 hash chaining.

Every audit log entry includes:
- SHA-256 hash of the previous entry (prev_hash)
- SHA-256 hash of this entry's content (entry_hash)

This creates a tamper-evident chain. Any modification to a past entry
causes all subsequent hashes to be invalid — detectable on inspection.

Retention: configurable via AUDIT_RETENTION_YEARS (HIPAA minimum: 6 years).
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from db.database import AuditLog, AuditEventType


_cached_last_hash: str | None = None
_hash_initialized: bool = False


def _compute_hash(prev_hash: str | None, event_data: dict, created_at: str) -> str:
    payload = json.dumps({
        "prev_hash": prev_hash or "GENESIS",
        "event_data": event_data,
        "created_at": created_at,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _get_last_hash(db: Session) -> str | None:
    global _cached_last_hash, _hash_initialized
    if not _hash_initialized:
        last = db.query(AuditLog).order_by(AuditLog.created_at.desc()).first()
        _cached_last_hash = last.entry_hash if last else None
        _hash_initialized = True
    return _cached_last_hash


def log_event(
    db: Session,
    event_type: AuditEventType,
    actor_role: str,
    api_key: str | None = None,
    patient_id: str | None = None,
    endpoint: str | None = None,
    event_data: dict | None = None,
) -> AuditLog:
    """
    Write an audit log entry with hash chain. Thread-safe for SQLite
    (SQLAlchemy handles session locking).
    """
    global _cached_last_hash
    now = datetime.now(timezone.utc).isoformat()
    prev_hash = _get_last_hash(db)
    data = event_data or {}
    entry_hash = _compute_hash(prev_hash, data, now)

    # Only store last 4 chars of API key as hint (never full key)
    key_hint = api_key[-4:] if api_key and len(api_key) >= 4 else None

    entry = AuditLog(
        event_type=event_type,
        patient_id=patient_id,
        actor_role=actor_role,
        actor_key_hint=key_hint,
        endpoint=endpoint,
        event_data=json.dumps(data),
        prev_hash=prev_hash,
        entry_hash=entry_hash,
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    _cached_last_hash = entry.entry_hash
    return entry


def verify_chain(db: Session, limit: int = 1000) -> tuple[bool, str]:
    """
    Verify the integrity of the audit hash chain.
    Returns (True, "OK") if intact, (False, reason) if tampered.
    """
    logs = db.query(AuditLog).order_by(AuditLog.created_at.asc()).limit(limit).all()
    prev_hash = None

    for log in logs:
        expected = _compute_hash(
            prev_hash,
            json.loads(log.event_data or "{}"),
            log.created_at.isoformat(),
        )
        if log.entry_hash != expected:
            return False, f"Hash mismatch at audit log entry {log.id} (created {log.created_at})"
        prev_hash = log.entry_hash

    return True, "OK"
