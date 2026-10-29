from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from db.database import get_db, Patient, AuditEventType
from api.middleware.auth import require_any
from vitals.mock_producer import list_test_cases, get_test_case_snapshot, get_test_case_ehr_path
from rag.ehr_processor import chunk_ehr
from rag.vector_store import upsert_chunks
from compliance.phi_handler import deidentify
from compliance.audit_logger import log_event

router = APIRouter(prefix="/mock", tags=["Mock / Testing"])


@router.get(
    "/test-cases",
    summary="List all predefined test cases with vitals and EHR metadata",
)
def list_mock_test_cases(role: str = Depends(require_any)):
    return list_test_cases()


@router.post(
    "/run/{case_id}",
    summary="Run a predefined test case end-to-end",
)
async def run_test_case(
    case_id: str,
    request: Request,
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    # Build snapshot (may raise ValidationError for guardrail-rejection test cases)
    try:
        snapshot = get_test_case_snapshot(case_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValidationError as e:
        # TC-08 and similar: schema-level guardrail rejection
        first_error = e.errors()[0]
        reason = first_error.get("msg", str(e))
        log_event(
            db=db,
            event_type=AuditEventType.INPUT_REJECTED,
            actor_role=role,
            api_key=request.headers.get("X-API-Key", ""),
            patient_id=None,
            endpoint=f"/api/v1/mock/run/{case_id}",
            event_data={"reason": reason, "case_id": case_id},
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input rejected by guardrail: {reason}",
        )

    # Auto-ingest EHR if this test case has one and it hasn't been ingested yet
    ehr_path = get_test_case_ehr_path(case_id)
    if ehr_path is not None:
        patient = db.query(Patient).filter(Patient.patient_id == snapshot.patient_id).first()
        if not patient or not patient.ehr_ingested:
            chunks = chunk_ehr(snapshot.patient_id, str(ehr_path))
            for chunk in chunks:
                chunk.text = deidentify(chunk.text)
            upsert_chunks(snapshot.patient_id, chunks)
            if not patient:
                patient = Patient(patient_id=snapshot.patient_id)
                db.add(patient)
            patient.ehr_ingested = True
            db.commit()

    # Delegate to the real vitals endpoint logic
    from api.routes.vitals import submit_vitals_snapshot
    return await submit_vitals_snapshot(
        payload=snapshot,
        request=request,
        db=db,
        role=role,
    )
