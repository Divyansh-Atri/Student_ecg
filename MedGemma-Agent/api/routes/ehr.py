import tempfile, os
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Request, status
from sqlalchemy.orm import Session

from db.database import get_db, Patient, AuditEventType
from api.middleware.auth import require_clinician_or_above
from rag.ehr_processor import chunk_ehr
from rag.vector_store import upsert_chunks
from compliance.phi_handler import deidentify, encrypt_phi
from compliance.audit_logger import log_event

router = APIRouter(prefix="/ehr", tags=["EHR"])


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Ingest an EHR PDF into the RAG store",
)
async def ingest_ehr(
    request: Request,
    patient_id: str = Form(...),
    age: int = Form(None),
    gender: str = Form(None),
    name: str = Form(None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    role: str = Depends(require_clinician_or_above),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    api_key = request.headers.get("X-API-Key", "")

    # Upsert patient record
    patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
    if not patient:
        patient = Patient(patient_id=patient_id, age=age, gender=gender)
        db.add(patient)
    else:
        if age:
            patient.age = age
        if gender:
            patient.gender = gender

    # Encrypt PHI name field before persisting
    if name:
        patient.name_encrypted = encrypt_phi(name)

    # Write PDF to temp file and process
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        chunks = chunk_ehr(patient_id, tmp_path)
        # De-identify chunk text before embedding/storing
        for chunk in chunks:
            chunk.text = deidentify(chunk.text)
        upsert_chunks(patient_id, chunks)
    finally:
        os.unlink(tmp_path)

    patient.ehr_ingested = True
    db.commit()

    log_event(
        db=db,
        event_type=AuditEventType.EHR_INGESTED,
        actor_role=role,
        api_key=api_key,
        patient_id=patient_id,
        endpoint="/api/v1/ehr/ingest",
        event_data={"chunks_created": len(chunks), "filename": file.filename},
    )

    return {
        "status": "accepted",
        "patient_id": patient_id,
        "chunks_created": len(chunks),
        "message": f"EHR ingested and indexed. {len(chunks)} chunks stored in RAG.",
    }
