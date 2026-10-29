import uuid
import httpx
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
import json
import re

from db.database import get_db, ECGResult
from ecg.schemas import ECGAnalyzeRequest, ECGAnalyzeResponse
from ecg.prompts import ECG_SYSTEM_PROMPT, build_ecg_user_prompt
from config.settings import get_settings
from api.middleware.auth import require_any

settings = get_settings()
router = APIRouter(prefix="/ecg", tags=["ECG Analysis"])

_ECG_LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "clinical_summary": {"type": "string"},
        "recommendation": {"type": "string"},
        "confidence": {"type": "number"}
    },
    "required": ["clinical_summary", "recommendation", "confidence"]
}

def _strip_think(text: str) -> str:
    text = re.sub(r"<unused94>.*?<unused95>", "", text, flags=re.DOTALL)
    return re.sub(r"<unused\d+>", "", text).strip()

@router.post(
    "/analyze",
    response_model=ECGAnalyzeResponse,
    summary="Analyze preprocessed ECG metrics using MedGemma"
)
async def analyze_ecg(
    payload: ECGAnalyzeRequest,
    db: Session = Depends(get_db),
    role: str = Depends(require_any)
):
    # Allow an external override of the user prompt (used by medgemma_bridge)
    if getattr(payload, "prompt_override", None):
        user_prompt = payload.prompt_override
    else:
        user_prompt = build_ecg_user_prompt(payload.model_dump())
    
    parsed = None
    try:
        async with httpx.AsyncClient(timeout=settings.OLLAMA_TIMEOUT) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": ECG_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": _ECG_LLM_SCHEMA,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            
            text = _strip_think(raw)
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    parsed = json.loads(m.group())
            
            if not parsed:
                if settings.LLM_FALLBACK_ON_TIMEOUT:
                    parsed = {
                        "clinical_summary": "System operating in fallback mode. LLM analysis unavailable.",
                        "recommendation": "Review standard protocol based on detected metrics.",
                        "confidence": 0.0
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Could not parse valid JSON from LLM response"
                    )
                
    except httpx.HTTPError as exc:
        if settings.LLM_FALLBACK_ON_TIMEOUT:
            parsed = {
                "clinical_summary": "System operating in fallback mode. LLM analysis unavailable.",
                "recommendation": "Review standard protocol based on detected metrics.",
                "confidence": 0.0
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach MedGemma model via Ollama: {str(exc)}"
            )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        if settings.LLM_FALLBACK_ON_TIMEOUT:
            parsed = {
                "clinical_summary": "System operating in fallback mode. LLM analysis unavailable.",
                "recommendation": "Review standard protocol based on detected metrics.",
                "confidence": 0.0
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"LLM Error: {str(exc)}"
            )

    ecg_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Save to DB
    ecg_record = ECGResult(
        id=ecg_id,
        patient_id=payload.patient_id,
        source_file=payload.metadata.source_file,
        quality_label=payload.quality_metrics.label,
        quality_score=payload.quality_metrics.score_pct,
        arrhythmia_detected=payload.clinical_features.arrhythmia_detected,
        arrhythmia_probability=payload.clinical_features.probability,
        arrhythmia_burden=payload.clinical_features.burden_pct,
        llm_clinical_summary=parsed.get("clinical_summary"),
        llm_recommendation=parsed.get("recommendation"),
        llm_confidence=parsed.get("confidence"),
        created_at=now
    )
    db.add(ecg_record)
    db.commit()

    return ECGAnalyzeResponse(
        id=ecg_id,
        patient_id=payload.patient_id,
        clinical_summary=parsed.get("clinical_summary", "No summary provided."),
        recommendation=parsed.get("recommendation", "No recommendation provided."),
        confidence=parsed.get("confidence", 0.0),
        created_at=now
    )
