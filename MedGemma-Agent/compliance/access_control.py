"""
RBAC access control helpers.

Roles and their permissions:
  admin      — full access: read/write patients, alerts, audit log, EHR ingest
  clinician  — read/write patients and alerts, EHR ingest; no audit log
  readonly   — read patients and alerts only; no EHR ingest, no audit log
"""

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
        "submit_vitals", "read_alerts", "read_patients",
        "ingest_ehr", "read_audit", "run_mock", "read_health",
    },
    "clinician": {
        "submit_vitals", "read_alerts", "read_patients",
        "ingest_ehr", "run_mock", "read_health",
    },
    "readonly": {
        "read_alerts", "read_patients", "read_health",
    },
}


def has_permission(role: str, permission: str) -> bool:
    return permission in ROLE_PERMISSIONS.get(role, set())
