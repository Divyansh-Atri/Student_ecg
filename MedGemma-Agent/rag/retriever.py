"""
EHR retriever — assembles the patient context string passed to MedGemma.

Retrieval query is fixed and clinically motivated: we always ask for
risk factors, medications, surgical history, and prior sepsis/infection,
since those are the EHR signals most relevant to post-operative deterioration.
"""

from __future__ import annotations
from functools import lru_cache
from rag.vector_store import query_patient_ehr, patient_ehr_exists

RETRIEVAL_QUERY = (
    "patient surgical history cardiovascular risk factors prior sepsis infection "
    "current medications allergies post-operative complications comorbidities"
)

NO_EHR_MESSAGE = (
    "No EHR data available for this patient. "
    "Clinical assessment is based on vital signs only."
)


@lru_cache(maxsize=1000)
def retrieve_patient_context(patient_id: str, top_k: int = 5) -> str:
    """
    Retrieve relevant EHR context for a patient and format it as a
    plain-text string for inclusion in the MedGemma prompt.

    Returns NO_EHR_MESSAGE if no EHR has been ingested for this patient.
    """
    if not patient_ehr_exists(patient_id):
        return NO_EHR_MESSAGE

    chunks = query_patient_ehr(patient_id, RETRIEVAL_QUERY, top_k=top_k)
    if not chunks:
        return NO_EHR_MESSAGE

    sections: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        section_label = f"[{chunk['section'].upper()}]" if chunk["section"] else "[EHR EXTRACT]"
        sections.append(f"{section_label}\n{chunk['text']}")

    return "\n\n---\n\n".join(sections)
