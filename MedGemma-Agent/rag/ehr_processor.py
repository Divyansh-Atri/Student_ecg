"""
EHR PDF processor — parses and chunks discharge summaries for RAG ingestion.

Strategy: paragraph-aware chunking that respects clinical section boundaries.
Chunk size ~400 tokens with 50-token overlap between adjacent chunks.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
import pdfplumber


# Clinical section headings commonly found in discharge summaries
SECTION_HEADINGS = re.compile(
    r"^("
    r"chief complaint|history of present illness|past medical history|"
    r"past surgical history|medications|allergies|social history|"
    r"family history|review of systems|physical examination|"
    r"assessment|plan|diagnosis|diagnoses|procedures|"
    r"laboratory|labs|imaging|discharge|follow.?up|"
    r"hospital course|operative note|post.?operative"
    r")\s*[:\-]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

AVG_CHARS_PER_TOKEN = 4      # rough approximation
TARGET_TOKENS = 400
OVERLAP_TOKENS = 50
TARGET_CHARS = TARGET_TOKEN = TARGET_TOKENS * AVG_CHARS_PER_TOKEN
OVERLAP_CHARS = OVERLAP_TOKENS * AVG_CHARS_PER_TOKEN


@dataclass
class EHRChunk:
    patient_id: str
    chunk_index: int
    text: str
    section: str | None      # detected clinical section, if any
    page: int | None
    char_count: int


def extract_text_from_pdf(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text) tuples."""
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append((i, text))
    return pages


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on double newlines or section headings."""
    paragraphs = re.split(r"\n{2,}", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _detect_section(paragraph: str) -> str | None:
    """Return the section heading if the paragraph starts with one."""
    first_line = paragraph.split("\n")[0].strip()
    if SECTION_HEADINGS.match(first_line):
        return first_line.rstrip(":- ").title()
    return None


def chunk_ehr(patient_id: str, pdf_path: str | Path) -> list[EHRChunk]:
    """
    Parse a PDF EHR and return a list of overlapping chunks with metadata.
    """
    pages = extract_text_from_pdf(pdf_path)
    chunks: list[EHRChunk] = []
    chunk_index = 0
    current_section: str | None = None

    for page_num, page_text in pages:
        paragraphs = _split_into_paragraphs(page_text)
        buffer = ""
        buffer_section = current_section

        for para in paragraphs:
            detected = _detect_section(para)
            if detected:
                current_section = detected

            if len(buffer) + len(para) > TARGET_CHARS and buffer:
                chunks.append(EHRChunk(
                    patient_id=patient_id,
                    chunk_index=chunk_index,
                    text=buffer.strip(),
                    section=buffer_section,
                    page=page_num,
                    char_count=len(buffer),
                ))
                chunk_index += 1
                # Overlap: carry last OVERLAP_CHARS of buffer into next chunk
                buffer = buffer[-OVERLAP_CHARS:] + "\n" + para
                buffer_section = current_section
            else:
                buffer += ("\n" if buffer else "") + para

        # Flush remaining buffer for this page
        if buffer.strip():
            chunks.append(EHRChunk(
                patient_id=patient_id,
                chunk_index=chunk_index,
                text=buffer.strip(),
                section=current_section,
                page=page_num,
                char_count=len(buffer),
            ))
            chunk_index += 1

    return chunks
