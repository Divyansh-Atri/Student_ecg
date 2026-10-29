"""
ABG interpretation API routes.

Completely independent of the vitals monitoring pipeline.
Endpoints:
  POST /abg/upload         — extract ABG values from a PDF lab report
  POST /abg/analyze        — manual ABG entry + MedGemma reasoning
  GET  /abg/test-cases     — list predefined test cases
  POST /abg/run/{case_id}  — run a named test case end-to-end
  GET  /abg/history        — list past ABG results
"""

from __future__ import annotations
import json
import re
import uuid
import tempfile
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form, status
from pydantic import ValidationError
from sqlalchemy.orm import Session

from db.database import get_db, ABGResult
from api.middleware.auth import require_any, require_clinician_or_above
from abg.schemas import ABGInput, ABGResponse, ABGListItem, ABGAlertLevel
from abg.pdf_parser import parse_abg_pdf
from abg.interpreter import ABGInterpreter
from abg.prompts import ABG_SYSTEM_PROMPT, build_abg_user_prompt
from abg.test_case_loader import list_abg_test_cases, get_abg_test_case_input
from config.settings import get_settings

settings = get_settings()

router = APIRouter(prefix="/abg", tags=["ABG Interpreter"])

# ── LLM helpers (standalone — no LangGraph) ───────────────────────────────────

_ABG_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "clinical_significance": {"type": "string"},
        "primary_concern":       {"type": "string"},
        "contributing_factors":  {"type": "array", "items": {"type": "string"}},
        "confidence":            {"type": "number"},
        "limitations":           {"type": "string"},
        "disclaimer":            {"type": "string"},
    },
    "required": ["clinical_significance", "primary_concern", "contributing_factors",
                 "confidence", "limitations", "disclaimer"],
}


def _strip_think(text: str) -> str:
    text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    return re.sub(r"<unused\d+>", "", text).strip()


def _extract_json(text: str) -> dict | None:
    text = _strip_think(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _call_medgemma_abg(
    payload: ABGInput,
    interp: dict,
    ehr_context: str,
) -> tuple[dict | None, bool, str | None]:
    """
    Call MedGemma for ABG reasoning.
    Returns (parsed_output, llm_available, error_message).
    """
    user_prompt = build_abg_user_prompt(
        patient_id=payload.patient_id,
        pH=payload.ph,
        pao2_mmhg=payload.pao2_mmhg,
        paco2_mmhg=payload.paco2_mmhg,
        hco3=payload.hco3,
        base_excess=payload.base_excess,
        lactate=payload.lactate,
        fio2=payload.fio2,
        na=payload.na,
        cl=payload.cl,
        chronic_copd=payload.chronic_copd,
        interp=interp,
        ehr_context=ehr_context,
    )
    try:
        with httpx.Client(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": ABG_SYSTEM_PROMPT},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "stream": False,
                    "format": _ABG_LLM_SCHEMA,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            parsed = _extract_json(raw)
            if parsed is None:
                return None, True, "Could not parse JSON from LLM response"
            return parsed, True, None
    except Exception as exc:
        return None, False, str(exc)[:200]


# ── EHR retrieval (reuse existing RAG if available) ───────────────────────────

def _get_ehr_context(patient_id: str) -> str:
    try:
        from rag.retriever import retrieve_patient_context
        ctx = retrieve_patient_context(patient_id)
        return ctx or "No EHR data available."
    except Exception:
        return "No EHR data available."


# ── Core analysis logic ────────────────────────────────────────────────────────

async def _analyze_abg(
    payload: ABGInput,
    db: Session,
) -> ABGResponse:
    # Step 1 — run rule-based interpretation
    interpreter = ABGInterpreter(
        pH=payload.ph,
        pao2_mmhg=payload.pao2_mmhg,
        paco2_mmhg=payload.paco2_mmhg,
        hco3=payload.hco3,
        base_excess=payload.base_excess,
        lactate=payload.lactate,
        fio2=payload.fio2,
        chronic_copd=payload.chronic_copd,
        na=payload.na,
        cl=payload.cl,
        albumin=payload.albumin,
        glucose=payload.glucose,
        ketones=payload.ketones,
    )
    interp = interpreter.interpret()

    # Step 2 — retrieve EHR context
    ehr_context = _get_ehr_context(payload.patient_id)

    # Step 3 — call MedGemma
    llm_out, llm_available, llm_error = _call_medgemma_abg(payload, interp, ehr_context)

    # Step 4 — build response
    abg_id = str(uuid.uuid4())
    now    = datetime.utcnow()

    rule_based_only = llm_out is None

    resp = ABGResponse(
        abg_id=abg_id,
        patient_id=payload.patient_id,
        alert_level=ABGAlertLevel(interp["alert_level"].value),
        summary=interp["summary"],
        clinical_alerts=interp["clinical_alerts"],
        respiratory=interp["respiratory"],
        acid_base=interp["acid_base"],
        compensation=interp["compensation"],
        metabolic=interp["metabolic"],
        clinical_significance=llm_out.get("clinical_significance") if llm_out else None,
        primary_concern=llm_out.get("primary_concern") if llm_out else None,
        contributing_factors=llm_out.get("contributing_factors") if llm_out else None,
        llm_confidence=llm_out.get("confidence") if llm_out else None,
        llm_limitations=llm_out.get("limitations") if llm_out else None,
        rule_based_only=rule_based_only,
        created_at=now,
    )

    # Step 5 — persist to DB
    ab  = interp["acid_base"]
    rec = ABGResult(
        id=abg_id,
        patient_id=payload.patient_id,
        ph=payload.ph,
        pao2_mmhg=payload.pao2_mmhg,
        paco2_mmhg=payload.paco2_mmhg,
        hco3=payload.hco3,
        base_excess=payload.base_excess,
        lactate=payload.lactate,
        fio2=payload.fio2,
        alert_level=interp["alert_level"].value,
        primary_disorder=ab["primary_disorder"].value if hasattr(ab["primary_disorder"], "value") else str(ab["primary_disorder"]),
        summary=interp["summary"],
        clinical_alerts=json.dumps(interp["clinical_alerts"]),
        respiratory_json=json.dumps({
            "hypoxia": interp["respiratory"]["hypoxia"],
            "type": interp["respiratory"]["type"].value,
            "pf_ratio": interp["respiratory"].get("pf_ratio"),
            "explanation": interp["respiratory"]["explanation"],
        }),
        acid_base_json=json.dumps({
            "status": ab["status"].value if hasattr(ab["status"], "value") else str(ab["status"]),
            "primary_disorder": ab["primary_disorder"].value if hasattr(ab["primary_disorder"], "value") else str(ab["primary_disorder"]),
            "acute_vs_chronic": ab.get("acute_vs_chronic"),
            "explanation": ab["explanation"],
        }),
        compensation_json=json.dumps({
            "compensated": interp["compensation"].get("compensated"),
            "expected_value": interp["compensation"].get("expected_value"),
            "measured_value": interp["compensation"].get("measured_value"),
            "explanation": interp["compensation"]["explanation"],
        }),
        metabolic_json=json.dumps({
            "anion_gap": interp["metabolic"].get("anion_gap"),
            "anion_gap_corrected": interp["metabolic"].get("anion_gap_corrected"),
            "gap_type": interp["metabolic"].get("gap_type"),
            "delta_delta": interp["metabolic"].get("delta_delta"),
            "likely_causes": interp["metabolic"]["likely_causes"],
            "explanation": interp["metabolic"]["explanation"],
        }),
        llm_clinical_significance=llm_out.get("clinical_significance") if llm_out else None,
        llm_primary_concern=llm_out.get("primary_concern") if llm_out else None,
        llm_contributing_factors=json.dumps(llm_out.get("contributing_factors", [])) if llm_out else None,
        llm_confidence=llm_out.get("confidence") if llm_out else None,
        llm_limitations=llm_out.get("limitations") if llm_out else None,
        rule_based_only=rule_based_only,
        disclaimer=resp.disclaimer,
        created_at=now,
    )
    db.add(rec)
    db.commit()

    return resp


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    summary="Extract ABG values from a PDF lab report for form pre-fill",
)
async def upload_abg_pdf(
    file: UploadFile = File(..., description="PDF lab report"),
    role: str = Depends(require_any),
):
    """
    Parse an ABG lab report PDF and return extracted values.
    Values are returned for UI pre-fill — the user reviews before submitting to /analyze.
    Fields that could not be extracted are null.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    # Write upload to a temp file for pdfplumber
    content = await file.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="PDF too large (max 10 MB)")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        parsed = parse_abg_pdf(tmp_path)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not extract text from PDF: {str(exc)[:200]}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Remove raw_text from response (not needed by UI, keeps payload small)
    parsed.pop("raw_text", None)

    core_fields = {"ph", "pao2_mmhg", "paco2_mmhg", "hco3", "base_excess", "lactate"}
    extracted   = set(parsed.get("extracted_fields", []))
    missing     = [f for f in core_fields if f not in extracted]

    return {
        **parsed,
        "missing_core_fields": missing,
        "parse_success": len(missing) == 0,
        "filename": file.filename,
    }


@router.post(
    "/analyze",
    response_model=ABGResponse,
    summary="Analyze ABG values — rule-based interpretation + MedGemma reasoning",
)
async def analyze_abg(
    payload: ABGInput,
    db: Session = Depends(get_db),
    role: str = Depends(require_clinician_or_above),
):
    return await _analyze_abg(payload, db)


@router.get(
    "/test-cases",
    summary="List predefined ABG test cases",
)
def list_test_cases(role: str = Depends(require_any)):
    return list_abg_test_cases()


@router.post(
    "/run/{case_id}",
    response_model=ABGResponse,
    summary="Run a predefined ABG test case end-to-end",
)
async def run_abg_test_case(
    case_id: str,
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    try:
        payload = get_abg_test_case_input(case_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValidationError as exc:
        first = exc.errors()[0]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Input rejected by guardrail: {first.get('msg', str(exc))}",
        )
    return await _analyze_abg(payload, db)


@router.get(
    "/history",
    response_model=list[ABGListItem],
    summary="List past ABG interpretations",
)
def abg_history(
    patient_id: str | None = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    role: str = Depends(require_any),
):
    q = db.query(ABGResult).order_by(ABGResult.created_at.desc())
    if patient_id:
        q = q.filter(ABGResult.patient_id == patient_id)
    rows = q.limit(limit).all()
    return [
        ABGListItem(
            abg_id=r.id,
            patient_id=r.patient_id,
            alert_level=ABGAlertLevel(r.alert_level),
            primary_disorder=r.primary_disorder,
            summary=r.summary,
            rule_based_only=r.rule_based_only,
            created_at=r.created_at,
        )
        for r in rows
    ]
