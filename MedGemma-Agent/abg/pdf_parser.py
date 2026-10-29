"""
ABG PDF Parser — extracts ABG values from lab report PDFs.

Handles common lab report formats:
  - Table-style (label + value on same or adjacent line)
  - Inline report style (label: value pairs)
  - Machine printout style (ABL90, i-STAT, Radiometer, Cobas)

Returns extracted values dict and a per-field confidence flag.
Values that could not be extracted are None — the UI pre-fills what it can
and highlights uncertain fields for clinician review before submission.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import pdfplumber


# ── Regex patterns for each ABG parameter ─────────────────────────────────────
# Each entry: list of (pattern, group_index) tuples, tried in order.
# First match wins. Patterns are case-insensitive.

_PATTERNS: dict[str, list[str]] = {
    "ph": [
        r"pH\s*[:\-=]?\s*([67]\.\d{1,3})",
        r"\bph\s*[:\-=]?\s*([67]\.\d{1,3})",
    ],
    "pao2_mmhg": [
        r"Pa?O[₂2]\s*[:\-=]?\s*([\d.]+)\s*mmHg",
        r"Pa?O[₂2]\s*[:\-=]?\s*([\d.]+)\s*kPa",   # flag for conversion
        r"p\s*O[₂2]\s*[:\-=]?\s*([\d.]+)\s*mmHg",
        r"p\s*O[₂2]\s*[:\-=]?\s*([\d.]+)\s*kPa",
        r"O2\s+Partial\s+Pressure\s*[:\-=]?\s*([\d.]+)\s*mmHg",
        r"Pa?O[₂2]\s*[:\-=]?\s*([\d.]+)",           # no unit — treat as mmHg
        r"p\s*O[₂2]\s*[:\-=]?\s*([\d.]+)",
    ],
    "paco2_mmhg": [
        r"Pa?CO[₂2]\s*[:\-=]?\s*([\d.]+)\s*mmHg",
        r"Pa?CO[₂2]\s*[:\-=]?\s*([\d.]+)\s*kPa",
        r"p\s*CO[₂2]\s*[:\-=]?\s*([\d.]+)\s*mmHg",
        r"p\s*CO[₂2]\s*[:\-=]?\s*([\d.]+)\s*kPa",
        r"Pa?CO[₂2]\s*[:\-=]?\s*([\d.]+)",
        r"p\s*CO[₂2]\s*[:\-=]?\s*([\d.]+)",
    ],
    "hco3": [
        r"HCO[₃3][-\s]*[:\-=]?\s*([\d.]+)",
        r"Bicarbonate\s*[:\-=]?\s*([\d.]+)",
        r"Bicarb\s*[:\-=]?\s*([\d.]+)",
        r"cHCO[₃3]\s*[:\-=]?\s*([\d.]+)",
    ],
    "base_excess": [
        r"BE\s*\(ecf\)\s*[:\=]?\s*([+\-]?[\d.]+)",
        r"BEecf\s*[:\=]?\s*([+\-]?[\d.]+)",
        # "Base Excess (BE) -4.2" — optional parenthetical; use [:\=]? not [:\-=]?
        # so the leading minus of a negative value is not eaten as a separator
        r"Base\s+Excess\s*(?:\([^)]*\))?\s*[:\=]?\s*([+\-]?[\d.]+)",
        r"Base\s+Excess\s*[:\=]?\s*([+\-]?[\d.]+)",
        r"B\.E\.\s*[:\=]?\s*([+\-]?[\d.]+)",
        # "BE" standalone — explicit colon/equals, or explicit sign
        r"\bBE\b\s*[:\=]\s*([+\-]?[\d.]+)",
        r"\bBE\b\s+([+\-][\d.]+)",
    ],
    "lactate": [
        r"Lactate\s*[:\-=]?\s*([\d.]+)",
        r"Lac\b\s*[:\-=]?\s*([\d.]+)",
        r"Lactic\s+Acid\s*[:\-=]?\s*([\d.]+)",
        r"cLac\s*[:\-=]?\s*([\d.]+)",
    ],
    "fio2": [
        r"FiO[₂2]\s*[:\-=]?\s*([\d.]+)\s*%",       # percentage — divide by 100
        r"FIO[₂2]\s*[:\-=]?\s*([\d.]+)\s*%",
        r"FiO[₂2]\s*[:\-=]?\s*(0\.\d+)",            # fraction already
        r"FIO[₂2]\s*[:\-=]?\s*(0\.\d+)",
        r"FiO[₂2]\s*[:\-=]?\s*([\d.]+)",            # ambiguous — infer below
        r"O2\s+fraction\s*[:\-=]?\s*([\d.]+)",
        r"Inspired\s+O2\s*[:\-=]?\s*([\d.]+)\s*%",
    ],
    "na": [
        r"Na\+?\s*[:\-=]?\s*([\d.]+)",
        r"Sodium\s*[:\-=]?\s*([\d.]+)",
        r"cNa\s*[:\-=]?\s*([\d.]+)",
    ],
    "cl": [
        r"Cl[-\s]*[:\-=]?\s*([\d.]+)",
        r"Chloride\s*[:\-=]?\s*([\d.]+)",
        r"cCl\s*[:\-=]?\s*([\d.]+)",
    ],
    "albumin": [
        r"Albumin\s*[:\-=]?\s*([\d.]+)\s*g/dL",
        r"Albumin\s*[:\-=]?\s*([\d.]+)\s*g/L",      # flag for conversion
        r"Alb\s*[:\-=]?\s*([\d.]+)\s*g/dL",
        r"Alb\s*[:\-=]?\s*([\d.]+)",
    ],
    "glucose": [
        r"Glucose\s*[:\-=]?\s*([\d.]+)\s*mmol",
        r"Glc\s*[:\-=]?\s*([\d.]+)",
        r"Glucose\s*[:\-=]?\s*([\d.]+)",
    ],
    "chronic_copd_flag": [
        r"\bCOPD\b",
        r"Chronic\s+obstructive",
        r"CO2\s+retainer",
        r"chronic\s+resp(?:iratory)?\s+failure",
    ],
    "ketones_flag": [
        r"Ketone[s]?\s*[:\-=]?\s*[Pp]os",
        r"Ketone[s]?\s*[:\-=]?\s*\+",
        r"\bKetosis\b",
        r"\bDKA\b",
    ],
    "patient_id": [
        r"Patient\s+ID\s*[:\-=]?\s*([A-Za-z0-9_\-]+)",
        r"MRN\s*[:\-=]?\s*([A-Za-z0-9_\-]+)",
        r"Patient\s+No\.?\s*[:\-=]?\s*([A-Za-z0-9_\-]+)",
        r"Acc(?:ession)?\s*[:\-=]?\s*([A-Za-z0-9_\-]+)",
    ],
}

# Fields where the value might be in kPa and needs mmHg conversion
_KPA_FIELDS = {"pao2_mmhg", "paco2_mmhg"}
_KPA_TO_MMHG = 7.5006


def _extract_text(pdf_path: str | Path) -> str:
    """Extract all text from PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text_parts.append(t)
    return "\n".join(text_parts)


def _match_field(field: str, text: str) -> tuple[Optional[float | bool], bool]:
    """
    Try all patterns for a field against the text.
    Returns (value, is_kpa_or_percent_flag).
    Bool fields (copd/ketones) return (True/False, False).
    """
    patterns = _PATTERNS.get(field, [])

    # Boolean flags — check negation context (up to 60 chars before match on same line)
    _NEGATION = re.compile(r'\b(?:no|not|without|denies|absent|negative|non[-\s])\b', re.IGNORECASE)
    if field in ("chronic_copd_flag", "ketones_flag"):
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                # Get text from start of the same line up to the match
                line_start = text.rfind('\n', 0, m.start()) + 1
                pre_text   = text[line_start:m.start()]
                if not _NEGATION.search(pre_text):
                    return True, False
        return False, False

    for i, pat in enumerate(patterns):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            try:
                val = float(raw)
            except ValueError:
                continue

            # Detect kPa patterns — they include "kPa" in the regex string
            is_kpa = "kPa" in pat

            # Detect albumin in g/L — convert to g/dL
            if field == "albumin" and "g/L" in pat:
                val = val / 10.0

            # Detect FiO2 as percentage
            if field == "fio2" and "%" in pat:
                val = val / 100.0
            elif field == "fio2" and val > 1.0:
                # Ambiguous — looks like a percentage, convert
                val = val / 100.0

            return val, is_kpa

    return None, False


def parse_abg_pdf(pdf_path: str | Path) -> dict:
    """
    Extract ABG values from a PDF lab report.

    Returns a dict with:
      - Extracted numeric values (None if not found)
      - 'extracted_fields': list of field names successfully extracted
      - 'uncertain_fields': list of fields where confidence is lower
      - 'raw_text': full extracted text (for debugging)
    """
    text = _extract_text(pdf_path)

    result: dict = {
        "ph":           None,
        "pao2_mmhg":    None,
        "paco2_mmhg":   None,
        "hco3":         None,
        "base_excess":  None,
        "lactate":      None,
        "fio2":         0.21,   # default room air
        "na":           None,
        "cl":           None,
        "albumin":      None,
        "glucose":      None,
        "ketones":      None,
        "chronic_copd": False,
        "patient_id":   None,
        "extracted_fields": [],
        "uncertain_fields": [],
        "raw_text":     text,
    }

    numeric_fields = ["ph", "pao2_mmhg", "paco2_mmhg", "hco3", "base_excess",
                      "lactate", "fio2", "na", "cl", "albumin", "glucose"]
    core_fields    = {"ph", "pao2_mmhg", "paco2_mmhg", "hco3", "base_excess", "lactate"}

    for field in numeric_fields:
        val, is_kpa = _match_field(field, text)
        if val is not None:
            if is_kpa and field in _KPA_FIELDS:
                val = round(val * _KPA_TO_MMHG, 1)
            result[field] = round(val, 3) if field == "ph" else round(val, 1)
            result["extracted_fields"].append(field)
            # Mark as uncertain if matched by a low-specificity pattern.
            # Only applicable when there are >=4 patterns (last 2 are low-specificity).
            # For fields with <4 patterns all are considered reliable.
            patterns = _PATTERNS.get(field, [])
            if len(patterns) >= 4 and field in core_fields:
                for i, pat in enumerate(patterns):
                    if re.search(pat, text, re.IGNORECASE):
                        if i >= len(patterns) - 2:
                            result["uncertain_fields"].append(field)
                        break

    # Boolean flags
    copd, _    = _match_field("chronic_copd_flag", text)
    ketones, _ = _match_field("ketones_flag", text)
    result["chronic_copd"] = bool(copd)
    result["ketones"]      = bool(ketones) if ketones else None

    # Patient ID
    pid_val, _ = _match_field("patient_id", text)
    if pid_val is None:
        # Try string extraction for patient ID
        m = re.search(
            r"Patient\s+(?:ID|No\.?|Name)?\s*[:\-=]?\s*([A-Za-z0-9_\-]+)",
            text, re.IGNORECASE
        )
        if m:
            result["patient_id"] = m.group(1)
    else:
        result["patient_id"] = str(int(pid_val)) if isinstance(pid_val, float) else str(pid_val)

    # Validate plausibility of core values
    _validate(result)

    return result


def _validate(result: dict) -> None:
    """Zero out values that are physiologically implausible (likely parse errors)."""
    checks = {
        "ph":          (6.5, 7.9),
        "pao2_mmhg":   (20,  700),
        "paco2_mmhg":  (10,  120),
        "hco3":        (5,   60),
        "base_excess": (-30, 30),
        "lactate":     (0.1, 30),
        "fio2":        (0.2, 1.01),
        "na":          (100, 170),
        "cl":          (70,  130),
        "albumin":     (1,   7),
    }
    for field, (lo, hi) in checks.items():
        v = result.get(field)
        if v is not None and not (lo <= v <= hi):
            result[field] = None
            result["extracted_fields"] = [f for f in result["extracted_fields"] if f != field]
            result["uncertain_fields"] = [f for f in result["uncertain_fields"] if f != field]
